"""
基于原生Pytorch的通用分割/分类任务的评价指标模块。

定位：
    仅依赖 torch、不依赖 torchmetrics，适用于：
        1. 自定义指标模块开发的学习与参考（基于混淆矩阵的指标推导范式）；
        2. 无 torchmetrics 依赖环境下对 general.py（MetricCollection）的快速替换。
    其余两个模块的定位：
        general.py     基于 torchmetrics 封装的常用指标集合（开箱即用）；
        metrics_dev.py 开发中的、集成 torchmetrics 特性的指标类实现。

接口风格（与 general.py 的 MetricCollection 对齐）：
    所有指标类均为 nn.Module 子类，累积状态以 buffer / 子模块形式注册，
    随 LightningModule 被 trainer 自动搬运设备；
    update() / compute() / reset() / clone() / metric_keys() /
    confusion_matrix 属性与 MetricCollection 约定一致，
    可在无 torchmetrics 环境下直接替换使用。

分层设计：
    ConfusionCounts / ConfusionMatrixCounts（状态层）
        只负责混淆矩阵的累积（update）、重置（reset）、合并（__add__ / all_reduce），
        并以 property 暴露逐类 one-vs-rest 计数视图（tp/fp/fn/tn 向量），
        不推导任何指标。
    指标计算层
        TorchClassificationMetric  分类任务指标（acc / balanced_acc / precision / recall / f1 / kappa / top-k）
        TorchSegmentationMetric    分割任务指标（pixel_acc / mean_iou / mean_dice / precision / recall / f1 / freq_iou / kappa）
        均持有 ConfusionMatrixCounts，从计数视图推导各自语义正确的指标。

混淆矩阵约定：
    shape 为 (num_classes, num_classes)，
    cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
"""
from __future__ import annotations

from typing import Any, Literal
import math
import copy
import warnings
import logging

import torch
import torch.nn as nn
import torch.distributed as dist


from .utils import fmt_value

logger = logging.getLogger(__name__)



class ConfusionCounts:
    """
    混淆计数状态容器（int64 Tensor 版本）。
    四个计数寄存器以 int64 0-dim 张量存储，支持：
        - 设备跟随（to / cpu / cuda），避免逐 batch 的 GPU→CPU 同步
        - 向量化批量累加（__iadd__ 张量原地加法）
        - DDP 精确聚合（all_reduce 整数无损）
    
    TP：真实正例
    TN：真实反例
    FP：假正例
    FN：假反例
    """
    _NAMES = ("tp", "tn", "fp", "fn")
    def __init__(
        self,
        tp: int | torch.Tensor = 0,
        tn: int | torch.Tensor = 0,
        fp: int | torch.Tensor = 0,
        fn: int | torch.Tensor = 0,
        device: torch.device | None = None,
    ):
        self._device = device or torch.device("cpu")
        vals = [tp, tn, fp, fn]
        # 将输入统一为 int64 0-dim 张量并 stack，避免逐个 .item() 同步
        tensors = []
        for v in vals:
            if isinstance(v, torch.Tensor):
                tensors.append(v.to(self._device).to(torch.int64).reshape(()))
            else:
                tensors.append(torch.tensor(int(v), dtype=torch.int64, device=self._device))
        self._counts = torch.stack(tensors)

    # ------------------------------------------------------------------
    # 设备管理
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> ConfusionCounts:
        """将计数张量迁移到指定设备，返回自身（支持链式调用）。"""
        device = torch.device(device)
        if device != self._device:
            self._counts = self._counts.to(device)
            self._device = device
        return self

    def cpu(self) -> ConfusionCounts:
        return self.to("cpu")

    def cuda(self, device: torch.device | None = None) -> ConfusionCounts:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return self.to(device)

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # 计数访问（返回 0-dim int64 张量，保持计算图一致）
    # ------------------------------------------------------------------

    @property
    def tp(self) -> torch.Tensor:
        return self._counts[0]

    @property
    def tn(self) -> torch.Tensor:
        return self._counts[1]

    @property
    def fp(self) -> torch.Tensor:
        return self._counts[2]

    @property
    def fn(self) -> torch.Tensor:
        return self._counts[3]

    # ------------------------------------------------------------------
    # 派生计数
    # ------------------------------------------------------------------

    @property
    def positive(self) -> torch.Tensor:
        """GT 正样本数 = tp + fn。"""
        return self.tp + self.fn

    @property
    def negative(self) -> torch.Tensor:
        """GT 负样本数 = tn + fp。"""
        return self.tn + self.fp

    @property
    def predicted_positive(self) -> torch.Tensor:
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> torch.Tensor:
        return self.tn + self.fn

    @property
    def total(self) -> torch.Tensor:
        return self._counts.sum()

    # ------------------------------------------------------------------
    # 状态维护：重置 / 合并
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清零全部计数。"""
        self._counts.zero_()

    def __add__(self, other: ConfusionCounts) -> ConfusionCounts:
        """合并两个计数器（适用于 DDP 多卡汇总）。"""
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        merged = ConfusionCounts(device=self._device)
        merged._counts = self._counts + other._counts.to(self._device)
        return merged

    def __iadd__(self, other: ConfusionCounts) -> ConfusionCounts:
        """就地合并计数器。"""
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        if self._counts.shape != other._counts.shape:
            raise ValueError(
                f"计数维度不匹配: {self._counts.shape} vs {other._counts.shape}"
            )
        self._counts += other._counts.to(self._device)
        return self

    # ------------------------------------------------------------------
    # DDP 聚合
    # ------------------------------------------------------------------

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：对四个计数做一次 all_reduce(SUM)。

        整数求和精确无损（int64 打包为 float32 通信，可精确表示 2^24 以内整数，足够覆盖实际计数规模）。
        单机或未初始化进程组时为 no-op，可无条件调用。
        """
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
            return
        t = self._counts.clone().float()  # float32: MPS 不支持 float64 运算
        if dist.get_backend() == "nccl":
            t = t.cuda(torch.cuda.current_device())
        dist.all_reduce(t)
        self._counts.copy_(t.to(self._device).long())

    # ------------------------------------------------------------------
    # 视图 / 序列化
    # ------------------------------------------------------------------

    def confusion_matrix(self) -> torch.Tensor:
        """
        装配二分类 2×2 混淆矩阵（行为真值、列为预测）。

        Returns:
            2×2 整型 torch.Tensor，shape = (2, 2)
        """
        return torch.stack(
            [
                torch.stack([self.tp, self.fn]),
                torch.stack([self.fp, self.tn]),
            ]
        )

    def summary(self) -> dict[str, float | int]:
        """计算完整指标摘要，返回 Python 标量（兼容 report / 日志管线）。"""
        # float32: MPS 后端对 float64 运算支持极其有限
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()
        tn = self.tn.float()

        total = tp + tn + fp + fn
        pos = tp + fn  # positive
        pred_pos = tp + fp  # predicted_positive

        recall = tp / (pos + 1e-10)
        accuracy = (tp + tn) / (total + 1e-10)
        precision = tp / (pred_pos + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            # Raw counts
            "tp": self.tp.item(),
            "tn": self.tn.item(),
            "fp": self.fp.item(),
            "fn": self.fn.item(),
            "positive": pos.item(),
            "negative": self.negative.item(),
            "predicted_positive": pred_pos.item(),
            "predicted_negative": self.predicted_negative.item(),
            "total": total.item(),
            # Derived statistics
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    def copy(self) -> ConfusionCounts:
        new = ConfusionCounts(device=self._device)
        new._counts = self._counts.clone()
        return new

    def as_dict(self) -> dict[str, int]:
        return {
            "tp": self.tp.item(),
            "tn": self.tn.item(),
            "fp": self.fp.item(),
            "fn": self.fn.item(),
            "total": self.total.item(),
        }

    @classmethod
    def from_dict(cls, data: dict, device: torch.device | None = None) -> ConfusionCounts:
        return cls(
            tp=data["tp"],
            tn=data["tn"],
            fp=data["fp"],
            fn=data["fn"],
            device=device,
        )

    def __bool__(self) -> bool:
        return bool(self.total.item() > 0)

    def __repr__(self) -> str:
        return (
            f"ConfusionCounts(tp={self.tp.item()}, tn={self.tn.item()}, "
            f"fp={self.fp.item()}, fn={self.fn.item()}, "
            f"total={self.total.item()})"
        )



class ConfusionMatrixCounts:
    """
    通用混淆矩阵状态容器（int64 Tensor 版本）。

    ConfusionCounts 的多分类推广：以 (C, C) int64 张量为唯一事实源，
    逐类 one-vs-rest 计数（tp / fp / fn / tn）均为派生视图，
    不单独维护，保证与矩阵的一致性。

    设计理念与 ConfusionCounts 对齐：
        - 纯 Python 类（非 nn.Module），自由脱离 Module 体系使用
        - 矩阵为唯一事实源，tp/fp/fn/tn 从矩阵动态推导
        - 手动设备管理（to / cpu / cuda），避免逐 batch 的 GPU→CPU 同步
        - 向量化合并（__iadd__）+ 兼容性校验
        - DDP 精确聚合（all_reduce 整数无损）
        - 序列化支持（as_dict / from_dict）

    混淆矩阵约定：
        matrix[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
        行 = 真值（GT），列 = 预测（Pred）。

    Example:
        >>> cm = ConfusionMatrixCounts(num_classes=5)
        >>> cm += other_cm
        >>> cm.tp                  # 每类正确预测数，shape=(5,) int64
        >>> cm.per_class_metrics() # 逐类 precision / recall / f1
        >>> cm.summary()           # 完整指标摘要
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        device: torch.device | None = None,
    ):
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self.matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=self._device
        )

    # ------------------------------------------------------------------
    # 设备管理
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "ConfusionMatrixCounts":
        """将矩阵迁移到指定设备，返回自身（支持链式调用）。"""
        device = torch.device(device)
        if device != self._device:
            self.matrix = self.matrix.to(device)
            self._device = device
        return self

    def cpu(self) -> "ConfusionMatrixCounts":
        return self.to("cpu")

    def cuda(self, device: torch.device | None = None) -> "ConfusionMatrixCounts":
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return self.to(device)

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def ignore_index(self) -> int | None:
        return self._ignore_index

    @property
    def total(self) -> int:
        """有效样本总数（已排除 ignore_index）。"""
        return int(self.matrix.sum())

    # ------------------------------------------------------------------
    # 派生视图：逐类 one-vs-rest 计数（shape=(num_classes,) int64 向量）
    # 与 ConfusionCounts 的标量 property 对应，这里推广为向量
    # ------------------------------------------------------------------

    @property
    def gt_count(self) -> torch.Tensor:
        """每类真实样本数（行和）。"""
        return self.matrix.sum(dim=1)

    @property
    def pred_count(self) -> torch.Tensor:
        """每类被预测样本数（列和）。"""
        return self.matrix.sum(dim=0)

    @property
    def tp(self) -> torch.Tensor:
        """每类正确预测数（对角线）。"""
        return self.matrix.diagonal()

    @property
    def fp(self) -> torch.Tensor:
        """每类误报数：被预测为该类但真实为其他类。"""
        return self.pred_count - self.tp

    @property
    def fn(self) -> torch.Tensor:
        """每类漏检数：真实为该类但被预测为其他类。"""
        return self.gt_count - self.tp

    @property
    def tn(self) -> torch.Tensor:
        """每类真阴数：one-vs-rest 视角下的其余部分。"""
        return self.total - self.tp - self.fp - self.fn

    # ------------------------------------------------------------------
    # 状态维护：重置 / 合并
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清零混淆矩阵。"""
        self.matrix.zero_()

    def _check_compatible(self, other: "ConfusionMatrixCounts") -> None:
        """合并前校验：类别数与忽略索引必须一致。"""
        if self._num_classes != other._num_classes:
            raise ValueError(
                f"num_classes 不一致，无法合并: {self._num_classes} vs {other._num_classes}"
            )
        if self._ignore_index != other._ignore_index:
            raise ValueError(
                f"ignore_index 不一致，无法合并: {self._ignore_index} vs {other._ignore_index}"
            )

    def __add__(self, other: "ConfusionMatrixCounts") -> "ConfusionMatrixCounts":
        """合并两个矩阵（如多进程各自统计后离线汇总）。"""
        if not isinstance(other, ConfusionMatrixCounts):
            return NotImplemented
        self._check_compatible(other)
        merged = ConfusionMatrixCounts(
            self._num_classes, ignore_index=self._ignore_index, device=self._device
        )
        merged.matrix = self.matrix + other.matrix.to(self._device)
        return merged

    def __iadd__(self, other: "ConfusionMatrixCounts") -> "ConfusionMatrixCounts":
        """就地合并。"""
        if not isinstance(other, ConfusionMatrixCounts):
            return NotImplemented
        self._check_compatible(other)
        self.matrix += other.matrix.to(self._device)
        return self

    # ------------------------------------------------------------------
    # DDP 聚合
    # ------------------------------------------------------------------

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：对矩阵做一次 all_reduce(SUM)。

        整数求和精确无损（int64 打包为 float32 通信，可精确表示 2^24 以内整数，
        足够覆盖实际计数规模）。单机或未初始化进程组时为 no-op，可无条件调用。
        """
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
            return
        mat = self.matrix.clone().float()  # float32: MPS 不支持 float64 运算
        if dist.get_backend() == "nccl":
            mat = mat.cuda(torch.cuda.current_device())
        dist.all_reduce(mat)
        self.matrix.copy_(mat.to(self._device).long())

    # ------------------------------------------------------------------
    # 视图 / 序列化
    # ------------------------------------------------------------------

    def per_class_metrics(self) -> list[dict[str, float]]:
        """
        逐类指标（precision / recall / f1）。

        Returns:
            列表，第 i 个元素为第 i 类的指标字典，值为 Python float。
        """
        eps = 1e-10
        tp_v = self.tp.float()
        gt_v = self.gt_count.float()
        pred_v = self.pred_count.float()

        precision = tp_v / (pred_v + eps)
        recall = tp_v / (gt_v + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return [
            {
                "class": i,
                "tp": int(tp_v[i].item()),
                "fp": int(self.fp[i].item()),
                "fn": int(self.fn[i].item()),
                "tn": int(self.tn[i].item()),
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "f1": float(f1[i].item()),
            }
            for i in range(self._num_classes)
        ]

    def summary(self) -> dict:
        """计算完整指标摘要，返回 Python 标量（兼容 report / 日志管线）。"""
        eps = 1e-10
        tp_v = self.tp.float()
        gt_v = self.gt_count.float()
        pred_v = self.pred_count.float()
        total_f = float(self.total)

        precision = tp_v / (pred_v + eps)
        recall = tp_v / (gt_v + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = float(tp_v.sum()) / (total_f + eps)

        return {
            "num_classes": self._num_classes,
            "ignore_index": self._ignore_index,
            "total": self.total,
            "accuracy": accuracy,
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "per_class": self.per_class_metrics(),
            "matrix": self.matrix.cpu().tolist(),
        }

    def confusion_matrix(self) -> torch.Tensor:
        """返回混淆矩阵的独立副本。"""
        return self.matrix.clone()

    def copy(self) -> "ConfusionMatrixCounts":
        new = ConfusionMatrixCounts(
            self._num_classes, ignore_index=self._ignore_index, device=self._device
        )
        new.matrix = self.matrix.clone()
        return new

    def as_dict(self) -> dict:
        """序列化为字典（矩阵以嵌套列表存储），兼容 JSON / 日志管线。"""
        return {
            "num_classes": self._num_classes,
            "ignore_index": self._ignore_index,
            "matrix": self.matrix.cpu().tolist(),
            "total": self.total,
        }

    @classmethod
    def from_dict(
        cls, data: dict, device: torch.device | None = None
    ) -> "ConfusionMatrixCounts":
        """从 as_dict() 输出反序列化。"""
        num_classes = data["num_classes"]
        ignore_index = data.get("ignore_index")
        instance = cls(num_classes, ignore_index=ignore_index, device=device)
        instance.matrix = torch.tensor(
            data["matrix"], dtype=torch.int64, device=instance._device
        )
        return instance

    def __bool__(self) -> bool:
        return self.total > 0

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrixCounts(num_classes={self._num_classes}, "
            f"ignore_index={self._ignore_index}, total={self.total})"
        )



class TorchClassificationMetric:
    """
    分类任务指标计算层（纯 Python 类）。

    持有 ConfusionMatrixCounts 作为唯一累积状态（纯 Python 对象，手动设备管理），
    所有指标从其计数视图（tp / fp / fn / tn / gt_count / pred_count）推导；
    例外是 Top-k 准确率——它依赖 logits 排序信息，无法从混淆矩阵还原，
    因此单独维护命中/总数两个标量计数。

    compute() 返回汇总指标：
        - acc             总体准确率（多分类下恒等于 micro P/R/F1）
        - balanced_acc    平衡准确率（= macro Recall），类别不均衡时更可靠
        - precision       精确率（聚合方式由 average 参数控制）
        - recall          召回率（聚合方式由 average 参数控制）
        - f1              F1 分数（聚合方式由 average 参数控制）
        - kappa           Cohen's Kappa，扣除随机一致性后的一致度

    按需获取：
        - topk_acc()      Top-k 准确率（仅当 top_k 非空且 update 传入 logits 时）
        - per_class_metrics()  逐类 precision/recall/f1

    注意：
        macro 平均对验证集中未出现的类别计为 0（与 sklearn 默认行为一致）。

    Example:
        >>> metric = ClassificationMetric(num_classes=10, top_k=5)
        >>> for logits, target in dataloader:
        ...     metric.update(logits, target)
        >>> results = metric.compute()
        >>> cm = metric.confusion_matrix()   # 混淆矩阵独立访问

    Args:
        num_classes: 类别总数（>= 2）
        average: 多类别聚合方式，"macro"/"micro"/"weighted"
        top_k: 额外统计 Top-k 准确率，None 表示不统计
        ignore_index: 忽略的标签索引
        prefix: 指标键名前缀（如 'val/'）
        postfix: 指标键名后缀
        monitor: 主监控指标键名，用于早停
        **kwargs: 兼容扩展
    """

    # 随 epoch 变化、且量纲一致（均为 [0, 1] 比例）可同图对比的指标键。
    CURVE_KEYS = ("accuracy", "precision", "recall", "f1")

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted"] = "macro",
        top_k: int | None = None,
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        monitor: str | None = "f1",
        **kwargs: Any,
    ):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"average 需为 'macro'/'micro'/'weighted' 之一，实际得到 '{average}'")
        if top_k is not None and not 1 <= top_k <= num_classes:
            raise ValueError(
                f"top_k 需在 [1, num_classes={num_classes}] 内，实际得到 {top_k}"
            )
        self.num_classes = num_classes
        self.average = average
        self.top_k = top_k
        self.ignore_index = ignore_index
        self._counts = ConfusionMatrixCounts(num_classes, ignore_index=ignore_index)
        self.prefix = prefix or ""
        self.postfix = postfix or ""
        self.monitor = monitor
        if self.monitor is not None and self.monitor not in self.core_metric_keys:
            raise ValueError(f"monitor 必须为 {self.core_metric_keys}，实际得到 {self.monitor}")
        # Top-k 计数器：int64 0-dim 张量，在与矩阵同设备上累积，
        # 避免 int()/item() 造成逐 batch 的 GPU→CPU 同步
        self._topk_correct = torch.zeros((), dtype=torch.int64)
        self._topk_total = torch.zeros((), dtype=torch.int64)
        # top_k 启用但传入类别索引时只警告一次的标志
        self._topk_index_warned = False

    # ------------------------------------------------------------------
    # 状态维护：重置 / 累积
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置全部累积状态（混淆矩阵 + Top-k 计数）。"""
        self._counts.reset()
        self._topk_correct.zero_()
        self._topk_total.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。

        Args:
            preds: 预测结果。
                   - logits/probabilities (N, C, ...) 自动沿 dim=1 取 argmax
                   - 类别索引 (N, ...) 直接使用
            target: 真实标签，与 argmax 后的 preds 同 shape 的整数张量
        """
        # 保存原始 logits 用于 Top-k（在预处理取 argmax 之前）
        raw_logits = preds if self.top_k is not None else None

        # 预处理：logits → 类别索引 + ignore_index 过滤 + 越界校验
        pred_idx, target_flat = self._preprocess_preds(preds, target)

        # 向量化批量统计，返回 ConfusionMatrixCounts 后合并
        counts = self._classify_batch(pred_idx, target_flat)
        self._counts += counts

        # Top-k 需要原始 logits 的排序信息
        if self.top_k is not None and raw_logits is not None:
            if raw_logits.dim() == target.dim() + 1:
                self._update_topk(raw_logits, target)
            elif not self._topk_index_warned:
                warnings.warn(
                    "top_k 已启用，但传入的 preds 是类别索引而非 logits，"
                    "Top-k 准确率将无法统计；请在 update 中传入 (N, C, ...) 形状的 logits"
                )
                self._topk_index_warned = True

    def _preprocess_preds(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预处理预测值：logits → argmax、形状校验、ignore_index 过滤、越界校验。

        Returns:
            (pred_idx, target_flat): 均为 1-D long tensor，已过滤 ignore_index
        """
        p = preds
        g = target

        # logits (N, C, ...) → 类别索引
        if p.dim() == g.dim() + 1:
            if p.shape[1] < 2:
                raise ValueError(
                    f"logits 的通道维需 >= 2，实际得到 {p.shape[1]}；"
                    "单通道输出请先阈值化后传入类别索引"
                )
            p = p.argmax(dim=1)

        if p.shape != g.shape:
            raise ValueError(
                f"preds 与 target 形状不匹配: {tuple(p.shape)} vs {tuple(g.shape)}"
            )

        # 展平
        p = p.reshape(-1).long()
        g = g.reshape(-1).long()

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = g != self.ignore_index
            p, g = p[valid], g[valid]

        # 越界标签校验
        for name, t in (("preds", p), ("target", g)):
            if t.numel() and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {self.num_classes - 1}]"
                )

        return p, g

    def _classify_batch(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> ConfusionMatrixCounts:
        """向量化批量统计，返回 ConfusionMatrixCounts。

        用 bincount 构建混淆矩阵，全程保持 Tensor 形式，避免逐次 .item() 同步。
        """
        C = self.num_classes
        device = self._counts.device

        # 防御性设备对齐
        preds = preds.to(device)
        target = target.to(device)

        # 行=真实类别，列=预测类别；bincount 批量统计
        indices = target * C + preds
        counts = torch.bincount(indices, minlength=C * C)
        matrix = counts.reshape(C, C).to(torch.int64)

        batch_cm = ConfusionMatrixCounts(C, ignore_index=self.ignore_index, device=device)
        batch_cm.matrix = matrix
        return batch_cm

    def _update_topk(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """累积 Top-k 命中统计（需要原始 logits 的排序信息）。"""
        # (N, C, ...) 沿类别维取 top-k 索引 → (N, k, ...)
        topk_idx = logits.topk(self.top_k, dim=1).indices
        hit = (topk_idx == target.unsqueeze(1).long()).any(dim=1)  # (N, ...)
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            hit = hit[valid]
        # 显式构造同设备累加结果，避免 device mismatch
        self._topk_correct += hit.sum().to(self._topk_correct.device)
        self._topk_total += hit.numel()

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, torch.Tensor]:
        """
        从累积状态计算分类指标。

        汇总指标的聚合方式由构造时的 average 参数决定：
            - "macro"   ：逐类计算后等权平均
            - "micro"   ：全局 TP/FP/FN 累加后统一计算（多分类下等价于 acc）
            - "weighted"：逐类计算后按真实类别频率加权平均

        Returns:
            dict 包含：{p}acc{s}, {p}balanced_acc{s}, {p}precision{s},
                      {p}recall{s}, {p}f1{s}, {p}kappa{s}
        """
        eps = 1e-10
        # float32: MPS 后端对 float64 运算支持极其有限
        tp = self._counts.tp.float()
        gt_count = self._counts.gt_count.float()
        pred_count = self._counts.pred_count.float()
        total = self._counts.total

        # ---- 逐类指标 ----
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # ---- 总体指标 ----
        acc = tp.sum() / (total + eps)
        balanced_acc = recall.mean()  # 恒为 macro recall，不受 average 影响

        # Cohen's Kappa: pe 为边缘分布下的随机一致性率
        pe = (gt_count * pred_count).sum() / ((total + eps) ** 2)
        if 1 - pe > eps:
            kappa = (acc - pe) / (1 - pe)
        else:
            kappa = torch.zeros((), dtype=torch.float32, device=tp.device)

        # ---- 按 average 聚合 ----
        freq = gt_count / (total + eps)
        agg_p, agg_r, agg_f = self._aggregate(precision, recall, f1, acc, freq)

        p, s = self.prefix, self.postfix
        return {
            f"{p}acc{s}": acc,
            f"{p}balanced_acc{s}": balanced_acc,
            f"{p}precision{s}": agg_p,
            f"{p}recall{s}": agg_r,
            f"{p}f1{s}": agg_f,
            f"{p}kappa{s}": kappa,
        }

    def _aggregate(
        self,
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1: torch.Tensor,
        acc: torch.Tensor,
        freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按 average 方式聚合逐类指标。"""
        if self.average == "micro":
            return acc, acc, acc
        elif self.average == "weighted":
            return (freq * precision).sum(), (freq * recall).sum(), (freq * f1).sum()
        else:  # macro
            return precision.mean(), recall.mean(), f1.mean()

    # ------------------------------------------------------------------
    # DDP 聚合
    # ------------------------------------------------------------------

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：混淆矩阵 + Top-k 计数一并同步，未初始化时 no-op。

        Note:
            含集合通信，所有 rank 必须同步调用且次数一致；重复调用会重复累加，
            应由调用方保证每轮只汇总一次。
        """
        self._counts.all_reduce()
        if self.top_k is not None:
            if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
                return
            t = torch.stack([self._topk_correct, self._topk_total]).float()
            if dist.get_backend() == "nccl":
                t = t.cuda(torch.cuda.current_device())
            dist.all_reduce(t)
            t = t.long()
            self._topk_correct.copy_(t[0].to(self._topk_correct.device))
            self._topk_total.copy_(t[1].to(self._topk_total.device))

    # ------------------------------------------------------------------
    # 设备管理（与 template.py DetectionMetric 行为对齐）
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> ClassificationMetric:
        """将内部计数张量迁移到指定设备，返回自身（支持链式调用）。"""
        self._counts.to(device)
        self._topk_correct = self._topk_correct.to(device)
        self._topk_total = self._topk_total.to(device)
        return self

    def cuda(self, device: torch.device | None = None) -> ClassificationMetric:
        return self.to(
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )

    def cpu(self) -> ClassificationMetric:
        return self.to(torch.device("cpu"))

    # ------------------------------------------------------------------
    # 计数访问（property）
    # ------------------------------------------------------------------

    @property
    def tp(self) -> torch.Tensor:
        """每类正确预测数（对角线），shape=(num_classes,)。"""
        return self._counts.tp

    @property
    def fp(self) -> torch.Tensor:
        """每类误报数，shape=(num_classes,)。"""
        return self._counts.fp

    @property
    def fn(self) -> torch.Tensor:
        """每类漏检数，shape=(num_classes,)。"""
        return self._counts.fn

    @property
    def tn(self) -> torch.Tensor:
        """每类真阴数（one-vs-rest），shape=(num_classes,)。"""
        return self._counts.tn

    @property
    def total(self) -> int:
        """有效样本总数（已排除 ignore_index）。"""
        return self._counts.total

    @property
    def counts(self) -> ConfusionMatrixCounts:
        """暴露内部混淆矩阵状态容器。"""
        return self._counts

    def confusion_matrix(self) -> torch.Tensor:
        """返回混淆矩阵 (num_classes, num_classes) 的独立副本，供可视化等下游使用。"""
        return self._counts.confusion_matrix()

    # ------------------------------------------------------------------
    # 键名 / 主指标 / 克隆
    # ------------------------------------------------------------------

    @property
    def metric_keys(self) -> list[str]:
        """compute() 输出键名列表（带 prefix/postfix，无需调用 compute）。"""
        p, s = self.prefix, self.postfix
        return [
            f"{p}acc{s}", f"{p}balanced_acc{s}",
            f"{p}precision{s}", f"{p}recall{s}", f"{p}f1{s}", f"{p}kappa{s}",
        ]

    @property
    def core_metric_keys(self) -> tuple[str, ...]:
        """核心指标键名（不含 prefix/postfix），供训练日志选择性 log。"""
        return ("precision", "recall", "f1", "accuracy")

    def primary_value(self, results: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取主指标值（用于 early stopping / model selection 等标量比较场景）。"""
        if self.monitor is None:
            raise ValueError("未设置 monitor，无法提取主指标值")
        return results[f"{self.prefix}{self.monitor}{self.postfix}"]

    def clone(self, prefix: str | None = None, postfix: str | None = None) -> ClassificationMetric:
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        return new_metric

    # ------------------------------------------------------------------
    # 按需获取：逐类指标 / Top-k
    # ------------------------------------------------------------------

    def per_class_metrics(self) -> dict[str, torch.Tensor]:
        """
        逐类详细指标（按需获取，不包含在 compute() 输出中）。

        从混淆矩阵推导每类的 precision / recall / f1，
        键名格式 precision_{i} / recall_{i} / f1_{i}。

        Returns:
            逐类指标字典，值为 float32 标量张量
        """
        eps = 1e-10
        tp = self._counts.tp.float()
        gt_count = self._counts.gt_count.float()
        pred_count = self._counts.pred_count.float()
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        p, s = self.prefix, self.postfix
        results = {}
        for i in range(self.num_classes):
            results[f"{p}precision_{i}{s}"] = precision[i]
            results[f"{p}recall_{i}{s}"] = recall[i]
            results[f"{p}f1_{i}{s}"] = f1[i]
        return results

    def topk_acc(self) -> torch.Tensor | None:
        """
        Top-k 准确率（按需获取，不包含在 compute() 输出中）。

        仅当 top_k 非空且已累积有效统计时返回结果，否则返回 None。
        """
        if self.top_k is None or self._topk_total == 0:
            return None
        return self._topk_correct.float() / self._topk_total.float()

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report(
        self,
        results: dict,
        elapsed_time: float | None = None,
        speed: float | None = None,
        log: logging.Logger | None = None,
    ) -> str:
        """
        生成本指标的测试报告文本，并经 log 输出。

        Args:
            results: compute() 返回的指标字典，可额外包含 loss
            elapsed_time: 测试耗时（秒），None 时跳过性能段
            speed: 测试吞吐（samples/sec）
            log: 日志器（None 时用模块级 logger）
        """
        log = log or logger
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("CLASSIFICATION TEST REPORT".center(60))
        lines.append("=" * 60)

        # 聚合指标
        p, s = self.prefix, self.postfix
        lines.append("")
        lines.append("Classification Metrics:")
        if "loss" in results:
            lines.append(f"  • Loss            : {fmt_value(results.get('loss'), '.4f')}")
        lines.append(
            f"  • Accuracy        : {fmt_value(results.get(f'{p}acc{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Balanced Acc    : {fmt_value(results.get(f'{p}balanced_acc{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Precision ({self.average:>8}) : {fmt_value(results.get(f'{p}precision{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Recall ({self.average:>8})    : {fmt_value(results.get(f'{p}recall{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • F1 ({self.average:>8})        : {fmt_value(results.get(f'{p}f1{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Kappa           : {fmt_value(results.get(f'{p}kappa{s}'), '.4f')}"
        )

        # Top-k
        topk = self.topk_acc()
        if topk is not None:
            lines.append(
                f"  • Top-{self.top_k} Acc       : {fmt_value(topk.item(), '.2f', scale=100, suffix='%')}"
            )

        # 逐类指标
        per_class = self._counts.per_class_metrics()
        lines.append("")
        lines.append("Per-class Breakdown:")
        lines.append(f"  {'Class':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
        for pc in per_class:
            lines.append(
                f"  {pc['class']:>6}  {pc['precision']:>10.4f}  {pc['recall']:>10.4f}"
                f"  {pc['f1']:>10.4f}  {pc['tp'] + pc['fn']:>8}"
            )

        # 原始计数
        lines.append("")
        lines.append(f"Total Samples: {self.total}")

        if elapsed_time is not None or speed is not None:
            lines.append("")
            lines.append("Performance:")
            lines.append(f"  • Time         : {fmt_value(elapsed_time, '.2f', suffix='s')}")
            lines.append(f"  • Speed        : {fmt_value(speed, '.0f', suffix=' samples/sec')}")

        # 混淆矩阵摘要
        cm = self.confusion_matrix()
        if cm is not None and cm.numel() > 0:
            diag = int(torch.trace(cm))
            cm_total = int(cm.sum())
            lines.append("")
            lines.append("Confusion Matrix Summary:")
            lines.append(f"  • Diagonal (correct) : {diag}")
            lines.append(f"  • Off-diagonal (err) : {cm_total - diag}")
            lines.append(f"  • Total              : {cm_total}")

        lines.append("=" * 60)

        report_text = "\n".join(lines)
        log.info("\n%s", report_text)
        return report_text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"average={self.average}, top_k={self.top_k}, total={self.total})"
        )



class TorchSegmentationMetric:
    """
    分割任务指标计算层（纯 Python 类）。

    TorchClassificationMetric 的分割任务对应版本：持有 ConfusionMatrixCounts
    作为唯一累积状态（纯 Python 对象，手动设备管理），所有指标从其计数视图
    （tp / fp / fn / tn / gt_count / pred_count）推导。

    与 TorchClassificationMetric 的关键差异：
        1. 输入预处理：支持分割任务的空间维度 (B, C, H, W) / (B, H, W)，
           自动展平为像素级 1-D 张量后构建混淆矩阵；
        2. 分割特有指标：IoU（Jaccard）、Dice、Frequency-weighted IoU；
        3. 无 Top-k：排序信息对密集预测任务无意义。

    compute() 返回汇总指标：
        - pixel_acc     像素准确率（多分类下恒等于 micro P/R/F1）
        - mean_iou      平均交并比（mIoU），分割任务最核心指标
        - mean_dice     平均 Dice 系数
        - precision     精确率（聚合方式由 average 参数控制）
        - recall        召回率（聚合方式由 average 参数控制）
        - f1            F1 分数（= Dice，聚合方式由 average 参数控制）
        - freq_iou      频率加权 IoU
        - kappa         Cohen's Kappa，扣除随机一致性后的一致度

    按需获取：
        - per_class_metrics()  逐类 precision / recall / f1 / iou / dice

    注意：
        macro 平均对验证集中未出现的类别计为 0（与 sklearn 默认行为一致）。

    Example:
        >>> metric = TorchSegmentationMetric(num_classes=21, prefix="val/")
        >>> for preds, target in dataloader:
        ...     metric.update(preds, target)
        >>> results = metric.compute()
        >>> cm = metric.confusion_matrix()   # 混淆矩阵独立访问

    Args:
        num_classes: 类别总数（>= 2），含背景类
        average: 多类别聚合方式，"macro"/"micro"/"weighted"
        ignore_index: 忽略的标签索引（如边界区域、void 类）
        prefix: 指标键名前缀（如 'val/'）
        postfix: 指标键名后缀
        monitor: 主监控指标键名，用于早停
        **kwargs: 兼容扩展
    """

    # 随 epoch 变化、且量纲一致（均为 [0, 1] 比例）可同图对比的指标键。
    CURVE_KEYS = ("pixel_acc", "mean_iou", "mean_dice", "precision", "recall", "f1")

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted"] = "macro",
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        monitor: str | None = "mean_iou",
        **kwargs: Any,
    ):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"average 需为 'macro'/'micro'/'weighted' 之一，实际得到 '{average}'")
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self._counts = ConfusionMatrixCounts(num_classes, ignore_index=ignore_index)
        self.prefix = prefix or ""
        self.postfix = postfix or ""
        self.monitor = monitor
        if self.monitor is not None and self.monitor not in self.core_metric_keys:
            raise ValueError(f"monitor 必须为 {self.core_metric_keys}，实际得到 {self.monitor}")

    # ------------------------------------------------------------------
    # 状态维护：重置 / 累积
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置全部累积状态（混淆矩阵）。"""
        self._counts.reset()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。

        Args:
            preds: 预测结果。
                   - logits/probabilities (B, C, H, W) 自动沿 dim=1 取 argmax
                   - 类别索引 (B, H, W) 直接使用
            target: 真实标签，与 argmax 后的 preds 同 shape 的整数张量
                    支持 (B, H, W) 或 (B, 1, H, W)（自动 squeeze）
        """
        pred_idx, target_flat = self._preprocess_preds(preds, target)

        # 向量化批量统计，返回 ConfusionMatrixCounts 后合并
        counts = self._classify_batch(pred_idx, target_flat)
        self._counts += counts

    def _preprocess_preds(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预处理分割预测：logits → argmax、空间展平、ignore_index 过滤、越界校验。

        支持的输入格式：
            preds:  (B, C, H, W) logits/probs 或 (B, H, W) 类别索引
            target: (B, H, W) 或 (B, 1, H, W) 类别索引

        Returns:
            (pred_idx, target_flat): 均为 1-D long tensor，已过滤 ignore_index
        """
        p = preds
        g = target

        # target: (B, 1, H, W) → (B, H, W)
        if g.ndim >= 3 and g.shape[1] == 1:
            g = g.squeeze(1)

        # logits (B, C, H, W) → 类别索引 (B, H, W)
        if p.ndim == g.ndim + 1:
            if p.shape[1] < 2:
                raise ValueError(
                    f"logits 的通道维需 >= 2，实际得到 {p.shape[1]}；"
                    "单通道输出请先阈值化后传入类别索引"
                )
            p = p.argmax(dim=1)
        elif p.ndim != g.ndim:
            raise ValueError(
                f"preds 与 target 维度不匹配: {tuple(p.shape)} vs {tuple(g.shape)}；"
                "preds 需为 (B, C, H, W) 或 (B, H, W)，target 需为 (B, H, W)"
            )

        if p.shape != g.shape:
            raise ValueError(
                f"preds 与 target 形状不匹配: {tuple(p.shape)} vs {tuple(g.shape)}"
            )

        # 展平为 1-D
        p = p.reshape(-1).long()
        g = g.reshape(-1).long()

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = g != self.ignore_index
            p, g = p[valid], g[valid]

        # 越界标签校验
        for name, t in (("preds", p), ("target", g)):
            if t.numel() and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {self.num_classes - 1}]"
                )

        return p, g

    def _classify_batch(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> ConfusionMatrixCounts:
        """向量化批量统计，返回 ConfusionMatrixCounts。

        用 bincount 构建混淆矩阵，全程保持 Tensor 形式，避免逐次 .item() 同步。
        """
        C = self.num_classes
        device = self._counts.device

        # 防御性设备对齐
        preds = preds.to(device)
        target = target.to(device)

        # 行=真实类别，列=预测类别；bincount 批量统计
        indices = target * C + preds
        counts = torch.bincount(indices, minlength=C * C)
        matrix = counts.reshape(C, C).to(torch.int64)

        batch_cm = ConfusionMatrixCounts(C, ignore_index=self.ignore_index, device=device)
        batch_cm.matrix = matrix
        return batch_cm

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def _per_class_iou_dice(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算逐类 IoU 和 Dice（供 compute / per_class_metrics 复用）。

        Returns:
            (iou, dice): shape=(num_classes,) 的 float32 张量
        """
        eps = 1e-10
        tp = self._counts.tp.float()
        fp = self._counts.fp.float()
        fn = self._counts.fn.float()

        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        return iou, dice

    def compute(self) -> dict[str, torch.Tensor]:
        """
        从累积状态计算分割指标。

        汇总指标的聚合方式由构造时的 average 参数决定：
            - "macro"   ：逐类计算后等权平均
            - "micro"   ：全局 TP/FP/FN 累加后统一计算（多分类下等价于 pixel_acc）
            - "weighted"：逐类计算后按真实类别频率加权平均

        Returns:
            dict 包含：{p}pixel_acc{s}, {p}mean_iou{s}, {p}mean_dice{s},
                      {p}precision{s}, {p}recall{s}, {p}f1{s},
                      {p}freq_iou{s}, {p}kappa{s}
        """
        eps = 1e-10
        tp = self._counts.tp.float()
        gt_count = self._counts.gt_count.float()
        pred_count = self._counts.pred_count.float()
        total = self._counts.total

        # ---- 逐类指标 ----
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou, dice = self._per_class_iou_dice()

        # ---- 总体指标 ----
        pixel_acc = tp.sum() / (total + eps)
        mean_iou = iou.mean()  # mIoU 恒为 macro IoU
        mean_dice = dice.mean()

        # Frequency-weighted IoU
        freq = gt_count / (total + eps)
        freq_iou = (freq * iou).sum()

        # Cohen's Kappa: pe 为边缘分布下的随机一致性率
        pe = (gt_count * pred_count).sum() / ((total + eps) ** 2)
        if 1 - pe > eps:
            kappa = (pixel_acc - pe) / (1 - pe)
        else:
            kappa = torch.zeros((), dtype=torch.float32, device=tp.device)

        # ---- 按 average 聚合 ----
        agg_p, agg_r, agg_f = self._aggregate(precision, recall, f1, pixel_acc)

        p, s = self.prefix, self.postfix
        return {
            f"{p}pixel_acc{s}": pixel_acc,
            f"{p}mean_iou{s}": mean_iou,
            f"{p}mean_dice{s}": mean_dice,
            f"{p}precision{s}": agg_p,
            f"{p}recall{s}": agg_r,
            f"{p}f1{s}": agg_f,
            f"{p}freq_iou{s}": freq_iou,
            f"{p}kappa{s}": kappa,
        }

    def _aggregate(
        self,
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1: torch.Tensor,
        pixel_acc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按 average 方式聚合逐类指标。"""
        if self.average == "micro":
            return pixel_acc, pixel_acc, pixel_acc
        elif self.average == "weighted":
            freq = self._counts.gt_count.float() / (self._counts.total + 1e-10)
            return (freq * precision).sum(), (freq * recall).sum(), (freq * f1).sum()
        else:  # macro
            return precision.mean(), recall.mean(), f1.mean()

    # ------------------------------------------------------------------
    # DDP 聚合
    # ------------------------------------------------------------------

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：混淆矩阵同步，未初始化时 no-op。

        Note:
            含集合通信，所有 rank 必须同步调用且次数一致；重复调用会重复累加，
            应由调用方保证每轮只汇总一次。
        """
        self._counts.all_reduce()

    # ------------------------------------------------------------------
    # 设备管理（与 TorchClassificationMetric 行为对齐）
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "TorchSegmentationMetric":
        """将内部计数张量迁移到指定设备，返回自身（支持链式调用）。"""
        self._counts.to(device)
        return self

    def cuda(self, device: torch.device | None = None) -> "TorchSegmentationMetric":
        return self.to(
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )

    def cpu(self) -> "TorchSegmentationMetric":
        return self.to(torch.device("cpu"))

    # ------------------------------------------------------------------
    # 计数访问（property）
    # ------------------------------------------------------------------

    @property
    def tp(self) -> torch.Tensor:
        """每类正确预测数（对角线），shape=(num_classes,)。"""
        return self._counts.tp

    @property
    def fp(self) -> torch.Tensor:
        """每类误报数，shape=(num_classes,)。"""
        return self._counts.fp

    @property
    def fn(self) -> torch.Tensor:
        """每类漏检数，shape=(num_classes,)。"""
        return self._counts.fn

    @property
    def tn(self) -> torch.Tensor:
        """每类真阴数（one-vs-rest），shape=(num_classes,)。"""
        return self._counts.tn

    @property
    def total(self) -> int:
        """有效像素总数（已排除 ignore_index）。"""
        return self._counts.total

    @property
    def counts(self) -> ConfusionMatrixCounts:
        """暴露内部混淆矩阵状态容器。"""
        return self._counts

    def confusion_matrix(self) -> torch.Tensor:
        """返回混淆矩阵 (num_classes, num_classes) 的独立副本，供可视化等下游使用。"""
        return self._counts.confusion_matrix()

    # ------------------------------------------------------------------
    # 键名 / 主指标 / 克隆
    # ------------------------------------------------------------------

    @property
    def metric_keys(self) -> list[str]:
        """compute() 输出键名列表（带 prefix/postfix，无需调用 compute）。"""
        p, s = self.prefix, self.postfix
        return [
            f"{p}pixel_acc{s}", f"{p}mean_iou{s}", f"{p}mean_dice{s}",
            f"{p}precision{s}", f"{p}recall{s}", f"{p}f1{s}",
            f"{p}freq_iou{s}", f"{p}kappa{s}",
        ]

    @property
    def core_metric_keys(self) -> tuple[str, ...]:
        """核心指标键名（不含 prefix/postfix），供训练日志选择性 log。"""
        return ("pixel_acc", "mean_iou", "mean_dice", "precision", "recall", "f1")

    def primary_value(self, results: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取主指标值（用于 early stopping / model selection 等标量比较场景）。"""
        if self.monitor is None:
            raise ValueError("未设置 monitor，无法提取主指标值")
        return results[f"{self.prefix}{self.monitor}{self.postfix}"]

    def clone(self, prefix: str | None = None, postfix: str | None = None) -> "TorchSegmentationMetric":
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        return new_metric

    # ------------------------------------------------------------------
    # 按需获取：逐类指标
    # ------------------------------------------------------------------

    def per_class_metrics(self) -> dict[str, torch.Tensor]:
        """
        逐类详细指标（按需获取，不包含在 compute() 输出中）。

        从混淆矩阵推导每类的 precision / recall / f1 / iou / dice，
        键名格式 precision_{i} / recall_{i} / f1_{i} / iou_{i} / dice_{i}。

        Returns:
            逐类指标字典，值为 float32 标量张量
        """
        eps = 1e-10
        tp = self._counts.tp.float()
        gt_count = self._counts.gt_count.float()
        pred_count = self._counts.pred_count.float()
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou, dice = self._per_class_iou_dice()
        p, s = self.prefix, self.postfix
        results = {}
        for i in range(self.num_classes):
            results[f"{p}precision_{i}{s}"] = precision[i]
            results[f"{p}recall_{i}{s}"] = recall[i]
            results[f"{p}f1_{i}{s}"] = f1[i]
            results[f"{p}iou_{i}{s}"] = iou[i]
            results[f"{p}dice_{i}{s}"] = dice[i]
        return results

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report(
        self,
        results: dict,
        elapsed_time: float | None = None,
        speed: float | None = None,
        log: logging.Logger | None = None,
    ) -> str:
        """
        生成本指标的测试报告文本，并经 log 输出。

        Args:
            results: compute() 返回的指标字典，可额外包含 loss
            elapsed_time: 测试耗时（秒），None 时跳过性能段
            speed: 测试吞吐（samples/sec）
            log: 日志器（None 时用模块级 logger）
        """
        log = log or logger
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("SEGMENTATION TEST REPORT".center(60))
        lines.append("=" * 60)

        # 聚合指标
        p, s = self.prefix, self.postfix
        lines.append("")
        lines.append("Segmentation Metrics:")
        if "loss" in results:
            lines.append(f"  • Loss            : {fmt_value(results.get('loss'), '.4f')}")
        lines.append(
            f"  • Pixel Acc       : {fmt_value(results.get(f'{p}pixel_acc{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Mean IoU (mIoU) : {fmt_value(results.get(f'{p}mean_iou{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Mean Dice       : {fmt_value(results.get(f'{p}mean_dice{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Precision ({self.average:>8}) : {fmt_value(results.get(f'{p}precision{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Recall ({self.average:>8})    : {fmt_value(results.get(f'{p}recall{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • F1 ({self.average:>8})        : {fmt_value(results.get(f'{p}f1{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Freq-weighted IoU: {fmt_value(results.get(f'{p}freq_iou{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Kappa           : {fmt_value(results.get(f'{p}kappa{s}'), '.4f')}"
        )

        # 逐类指标
        per_class = self.per_class_metrics()
        lines.append("")
        lines.append("Per-class Breakdown:")
        lines.append(
            f"  {'Class':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}"
            f"  {'IoU':>10}  {'Dice':>10}  {'Support':>8}"
        )
        gt = self._counts.gt_count
        for i in range(self.num_classes):
            lines.append(
                f"  {i:>6}"
                f"  {per_class[f'{p}precision_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}recall_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}f1_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}iou_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}dice_{i}{s}'].item():>10.4f}"
                f"  {int(gt[i].item()):>8}"
            )

        # 原始计数
        lines.append("")
        lines.append(f"Total Pixels: {self.total}")

        if elapsed_time is not None or speed is not None:
            lines.append("")
            lines.append("Performance:")
            lines.append(f"  • Time         : {fmt_value(elapsed_time, '.2f', suffix='s')}")
            lines.append(f"  • Speed        : {fmt_value(speed, '.0f', suffix=' samples/sec')}")

        # 混淆矩阵摘要
        cm = self.confusion_matrix()
        if cm is not None and cm.numel() > 0:
            diag = int(torch.trace(cm))
            cm_total = int(cm.sum())
            lines.append("")
            lines.append("Confusion Matrix Summary:")
            lines.append(f"  • Diagonal (correct) : {diag}")
            lines.append(f"  • Off-diagonal (err) : {cm_total - diag}")
            lines.append(f"  • Total              : {cm_total}")

        lines.append("=" * 60)

        report_text = "\n".join(lines)
        log.info("\n%s", report_text)
        return report_text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"average={self.average}, total={self.total})"
        )


# # Alias for __init__.py import compatibility
# SegmentationMetric = TorchSegmentationMetric
# ClassificationMetric = TorchClassificationMetric



############ 自定义指标 ############


# ------------------------------------------------------------------
# 模块级工具函数（可独立调用，亦供 SeparatedKappaMetric 复用逻辑）
# ------------------------------------------------------------------


def _kappa_from_matrix(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """
    从混淆矩阵计算 Cohen's Kappa。

    边界约定与 SCD 官方实现一致：矩阵全零或期望一致率 pe 为 1 时返回 0。

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别
    """
    hist = matrix.double()
    total = hist.sum()
    if total == 0:
        return 0.0
    # Po（观察一致率）：实际一致的比例
    po = hist.diagonal().sum() / total
    # Pe（随机一致率）：随机一致的比例
    pe = (hist.sum(dim=1) * hist.sum(dim=0)).sum() / total ** 2
    if abs(1 - pe) < eps:
        return 0.0
    return ((po - pe) / (1 - pe)).item()



def separated_kappa(matrix: torch.Tensor, bg_index: int = 0) -> dict[str, float]:
    """
    分离 Kappa（Separated Kappa, SeK）系数（函数版本）。

    面向"多类前景 + 主导性背景"的场景（语义变化检测 SCD、灾损分级等）：
    传统 Kappa 会被巨量的"背景→背景"正确项（TN）虚高，SeK 将其从混淆
    矩阵中剔除后再计算 Kappa（衡量前景类语义分对了没），并乘以前景二值
    IoU 的指数惩罚项（衡量前景空间定位准不准）：

        SeK = kappa_n0 * exp(IoU_fg - 1)

    参考：SECOND 数据集官方指标 https://captain-whu.github.io/SCD/

    注意：
        - 前景仅 1 类时 kappa_n0 退化为 IoU 的变体，此时直接用 IoU/F1 即可；
        - 类别相对均衡的普通多分类任务不适用，应使用传统 Kappa。

    Example:
        >>> cm = ConfusionMatrixCounts(num_classes=5)
        >>> cm += batch_cm
        >>> results = separated_kappa(cm.matrix)
        >>> print(f"SeK: {results['sek']:.5f}")

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别，
            可直接传入 ConfusionMatrixCounts.matrix 或指标类的 confusion_matrix()
        bg_index: 背景/未变化类的索引，默认 0（SCD 官方约定）

    Returns:
        {
            'sek':      分离 Kappa 系数,
            'kappa_n0': 剔除背景 TN 后的 Kappa（前景语义分类一致度）,
            'iou_fg':   前景（变化区域）二值 IoU,
            'iou_bg':   背景（未变化区域）二值 IoU,
            'biou':     二值 mIoU = (iou_fg + iou_bg) / 2,
        }
    """
    hist = matrix.double()
    n = hist.shape[0]
    if hist.dim() != 2 or hist.shape[0] != hist.shape[1]:
        raise ValueError(f"matrix 需为方阵，实际 shape={tuple(matrix.shape)}")
    if not 0 <= bg_index < n:
        raise ValueError(f"bg_index 越界: {bg_index}，合法区间 [0, {n - 1}]")

    eps = 1e-10
    # 剔除"背景→背景"的巨量 TN，衡量前景类语义分类一致度
    hist_n0 = hist.clone()
    hist_n0[bg_index, bg_index] = 0
    kappa_n0 = _kappa_from_matrix(hist_n0)

    # 折叠为"背景/前景"二值矩阵，计算前景空间定位质量（行=真实，列=预测）
    tn = hist[bg_index, bg_index]                # 背景判对
    fn = hist[:, bg_index].sum() - tn            # 前景漏检：真前景 → 预测背景
    fp = hist[bg_index, :].sum() - tn            # 背景误检：真背景 → 预测前景
    tp = hist.sum() - tn - fp - fn               # 前景判为前景（含类间混淆）

    iou_fg = (tp / (tp + fp + fn + eps)).item()
    iou_bg = (tn / (tn + fp + fn + eps)).item()
    sek = kappa_n0 * math.exp(iou_fg - 1)

    return {
        'sek': sek,
        'kappa_n0': kappa_n0,
        'iou_fg': iou_fg,
        'iou_bg': iou_bg,
        'biou': (iou_fg + iou_bg) / 2,
    }


# ------------------------------------------------------------------
# SeparatedKappaMetric 类（可累积、可 DDP、与指标体系对齐）
# ------------------------------------------------------------------


class SeparatedKappaMetric:
    """
    分离 Kappa（Separated Kappa, SeK）指标类。

    separated_kappa() 函数版本需要一次性传入完整混淆矩阵；
    本类将其纳入 update/compute/reset 累积范式，复用 ConfusionMatrixCounts
    状态层，与 TorchClassificationMetric / TorchSegmentationMetric 接口对齐，
    可直接嵌入 LightningModule 的验证/测试流程。

    适用场景：
        "多类前景 + 主导性背景"的语义分割 / 变化检测任务（SCD、灾损分级等），
        传统 Kappa 被巨量背景 TN 虚高，SeK 剔除背景后评估前景语义分类质量，
        并以 exp(IoU_fg - 1) 惩罚前景空间定位偏差。

    compute() 返回指标：
        - sek           分离 Kappa = kappa_n0 * exp(IoU_fg - 1)
        - kappa_n0      剔除背景 TN 后的 Kappa（前景语义一致度）
        - iou_fg        前景二值 IoU（空间定位质量）
        - iou_bg        背景二值 IoU
        - biou          二值 mIoU = (iou_fg + iou_bg) / 2
        - kappa         传统全局 Kappa（供对照参考）

    Example:
        >>> metric = SeparatedKappaMetric(num_classes=7, bg_index=0, prefix="val/")
        >>> for preds, target in dataloader:
        ...     metric.update(preds, target)
        >>> results = metric.compute()
        >>> print(f"SeK: {results['val/sek']:.5f}")

    Args:
        num_classes: 类别总数（>= 2），含背景类
        bg_index: 背景/未变化类的索引，默认 0（SCD 官方约定）
        ignore_index: 忽略的标签索引（如边界区域、void 类）
        prefix: 指标键名前缀（如 'val/'）
        postfix: 指标键名后缀
        monitor: 主监控指标键名，用于早停
        **kwargs: 兼容扩展
    """

    # 随 epoch 变化可同图对比的指标键
    CURVE_KEYS = ("sek", "kappa_n0", "iou_fg", "iou_bg", "biou", "kappa")

    def __init__(
        self,
        num_classes: int,
        bg_index: int = 0,
        ignore_index: int | None = None,
        prefix: str | None = None,
        postfix: str | None = None,
        monitor: str | None = "sek",
        **kwargs: Any,
    ):
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        if not 0 <= bg_index < num_classes:
            raise ValueError(
                f"bg_index 越界: {bg_index}，合法区间 [0, {num_classes - 1}]"
            )
        self.num_classes = num_classes
        self.bg_index = bg_index
        self.ignore_index = ignore_index
        self._counts = ConfusionMatrixCounts(num_classes, ignore_index=ignore_index)
        self.prefix = prefix or ""
        self.postfix = postfix or ""
        self.monitor = monitor
        if self.monitor is not None and self.monitor not in self.core_metric_keys:
            raise ValueError(f"monitor 必须为 {self.core_metric_keys}，实际得到 {self.monitor}")

    # ------------------------------------------------------------------
    # 状态维护：重置 / 累积
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置全部累积状态（混淆矩阵）。"""
        self._counts.reset()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。

        自动适配输入维度：
            - 分割格式 (B, C, H, W) logits / (B, H, W) 类别索引
            - 分类格式 (N, C) logits / (N,) 类别索引
            - target 支持 (..., 1, ...) 通道维自动 squeeze

        Args:
            preds: 预测结果（logits/probs 或类别索引）
            target: 真实标签，整数张量
        """
        pred_idx, target_flat = self._preprocess_preds(preds, target)

        # 向量化批量统计，返回 ConfusionMatrixCounts 后合并
        counts = self._classify_batch(pred_idx, target_flat)
        self._counts += counts

    def _preprocess_preds(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预处理预测值：logits → argmax、形状校验、ignore_index 过滤、越界校验。

        同时支持分割 (B, C, H, W) 和分类 (N, C) 输入，统一展平为 1-D。

        Returns:
            (pred_idx, target_flat): 均为 1-D long tensor，已过滤 ignore_index
        """
        p = preds
        g = target

        # target: (..., 1, ...) → squeeze 通道维（仅对 ndim >= 3 且 shape[1]==1 生效）
        if g.ndim >= 3 and g.shape[1] == 1:
            g = g.squeeze(1)

        # logits (B, C, H, W) 或 (N, C) → 类别索引
        if p.ndim == g.ndim + 1:
            if p.shape[1] < 2:
                raise ValueError(
                    f"logits 的通道维需 >= 2，实际得到 {p.shape[1]}；"
                    "单通道输出请先阈值化后传入类别索引"
                )
            p = p.argmax(dim=1)
        elif p.ndim != g.ndim:
            raise ValueError(
                f"preds 与 target 维度不匹配: {tuple(p.shape)} vs {tuple(g.shape)}"
            )

        if p.shape != g.shape:
            raise ValueError(
                f"preds 与 target 形状不匹配: {tuple(p.shape)} vs {tuple(g.shape)}"
            )

        # 展平为 1-D
        p = p.reshape(-1).long()
        g = g.reshape(-1).long()

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = g != self.ignore_index
            p, g = p[valid], g[valid]

        # 越界标签校验
        for name, t in (("preds", p), ("target", g)):
            if t.numel() and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {self.num_classes - 1}]"
                )

        return p, g

    def _classify_batch(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> ConfusionMatrixCounts:
        """向量化批量统计，返回 ConfusionMatrixCounts。"""
        C = self.num_classes
        device = self._counts.device

        preds = preds.to(device)
        target = target.to(device)

        indices = target * C + preds
        counts = torch.bincount(indices, minlength=C * C)
        matrix = counts.reshape(C, C).to(torch.int64)

        batch_cm = ConfusionMatrixCounts(C, ignore_index=self.ignore_index, device=device)
        batch_cm.matrix = matrix
        return batch_cm

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def _compute_sek(self) -> dict[str, float]:
        """从累积的混淆矩阵计算 SeK 相关指标（内部复用）。"""
        matrix = self._counts.matrix
        hist = matrix.double()
        bg = self.bg_index
        eps = 1e-10

        # 剔除背景 TN 后计算 kappa_n0
        hist_n0 = hist.clone()
        hist_n0[bg, bg] = 0
        kappa_n0 = _kappa_from_matrix(hist_n0)

        # 折叠为前景/背景二值，计算空间定位质量
        tn = hist[bg, bg]                        # 背景判对
        fn = hist[:, bg].sum() - tn              # 前景漏检
        fp = hist[bg, :].sum() - tn              # 背景误检
        tp = hist.sum() - tn - fp - fn           # 前景判对（含类间混淆）

        iou_fg = float((tp / (tp + fp + fn + eps)).item())
        iou_bg = float((tn / (tn + fp + fn + eps)).item())
        sek = kappa_n0 * math.exp(iou_fg - 1)

        # 传统全局 Kappa（供对照）
        kappa = _kappa_from_matrix(hist)

        return {
            'sek': sek,
            'kappa_n0': kappa_n0,
            'iou_fg': iou_fg,
            'iou_bg': iou_bg,
            'biou': (iou_fg + iou_bg) / 2,
            'kappa': kappa,
        }

    def compute(self) -> dict[str, torch.Tensor]:
        """
        从累积状态计算 SeK 相关指标。

        Returns:
            dict 包含：{p}sek{s}, {p}kappa_n0{s}, {p}iou_fg{s},
                      {p}iou_bg{s}, {p}biou{s}, {p}kappa{s}
        """
        raw = self._compute_sek()
        p, s = self.prefix, self.postfix
        device = self._counts.device
        return {
            f"{p}sek{s}": torch.tensor(raw['sek'], dtype=torch.float32, device=device),
            f"{p}kappa_n0{s}": torch.tensor(raw['kappa_n0'], dtype=torch.float32, device=device),
            f"{p}iou_fg{s}": torch.tensor(raw['iou_fg'], dtype=torch.float32, device=device),
            f"{p}iou_bg{s}": torch.tensor(raw['iou_bg'], dtype=torch.float32, device=device),
            f"{p}biou{s}": torch.tensor(raw['biou'], dtype=torch.float32, device=device),
            f"{p}kappa{s}": torch.tensor(raw['kappa'], dtype=torch.float32, device=device),
        }

    # ------------------------------------------------------------------
    # DDP 聚合
    # ------------------------------------------------------------------

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：混淆矩阵同步，未初始化时 no-op。

        Note:
            含集合通信，所有 rank 必须同步调用且次数一致。
        """
        self._counts.all_reduce()

    # ------------------------------------------------------------------
    # 设备管理
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "SeparatedKappaMetric":
        """将内部计数张量迁移到指定设备，返回自身（支持链式调用）。"""
        self._counts.to(device)
        return self

    def cuda(self, device: torch.device | None = None) -> "SeparatedKappaMetric":
        return self.to(
            torch.device("cuda", torch.cuda.current_device()) if device is None else device
        )

    def cpu(self) -> "SeparatedKappaMetric":
        return self.to(torch.device("cpu"))

    # ------------------------------------------------------------------
    # 计数访问（property）
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """有效样本总数（已排除 ignore_index）。"""
        return self._counts.total

    @property
    def counts(self) -> ConfusionMatrixCounts:
        """暴露内部混淆矩阵状态容器。"""
        return self._counts

    def confusion_matrix(self) -> torch.Tensor:
        """返回混淆矩阵 (num_classes, num_classes) 的独立副本。"""
        return self._counts.confusion_matrix()

    # ------------------------------------------------------------------
    # 键名 / 主指标 / 克隆
    # ------------------------------------------------------------------

    @property
    def metric_keys(self) -> list[str]:
        """compute() 输出键名列表（带 prefix/postfix）。"""
        p, s = self.prefix, self.postfix
        return [
            f"{p}sek{s}", f"{p}kappa_n0{s}",
            f"{p}iou_fg{s}", f"{p}iou_bg{s}",
            f"{p}biou{s}", f"{p}kappa{s}",
        ]

    @property
    def core_metric_keys(self) -> tuple[str, ...]:
        """核心指标键名（不含 prefix/postfix）。"""
        return ("sek", "kappa_n0", "iou_fg", "iou_bg", "biou", "kappa")

    def primary_value(self, results: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取主指标值（用于 early stopping / model selection）。"""
        if self.monitor is None:
            raise ValueError("未设置 monitor，无法提取主指标值")
        return results[f"{self.prefix}{self.monitor}{self.postfix}"]

    def clone(self, prefix: str | None = None, postfix: str | None = None) -> "SeparatedKappaMetric":
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        return new_metric

    # ------------------------------------------------------------------
    # 按需获取：逐类指标（与 TorchSegmentationMetric 对齐）
    # ------------------------------------------------------------------

    def per_class_metrics(self) -> dict[str, torch.Tensor]:
        """
        逐类详细指标（按需获取，不包含在 compute() 输出中）。

        从混淆矩阵推导每类的 precision / recall / f1 / iou / dice，
        键名格式 precision_{i} / recall_{i} / f1_{i} / iou_{i} / dice_{i}。

        Returns:
            逐类指标字典，值为 float32 标量张量
        """
        eps = 1e-10
        tp = self._counts.tp.float()
        gt_count = self._counts.gt_count.float()
        pred_count = self._counts.pred_count.float()
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        fp = self._counts.fp.float()
        fn = self._counts.fn.float()
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        p, s = self.prefix, self.postfix
        results = {}
        for i in range(self.num_classes):
            results[f"{p}precision_{i}{s}"] = precision[i]
            results[f"{p}recall_{i}{s}"] = recall[i]
            results[f"{p}f1_{i}{s}"] = f1[i]
            results[f"{p}iou_{i}{s}"] = iou[i]
            results[f"{p}dice_{i}{s}"] = dice[i]
        return results

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report(
        self,
        results: dict,
        elapsed_time: float | None = None,
        speed: float | None = None,
        log: logging.Logger | None = None,
    ) -> str:
        """
        生成本指标的测试报告文本，并经 log 输出。

        Args:
            results: compute() 返回的指标字典，可额外包含 loss
            elapsed_time: 测试耗时（秒），None 时跳过性能段
            speed: 测试吞吐（samples/sec）
            log: 日志器（None 时用模块级 logger）
        """
        log = log or logger
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("SEPARATED KAPPA TEST REPORT".center(60))
        lines.append("=" * 60)

        p, s = self.prefix, self.postfix
        lines.append("")
        lines.append("Separated Kappa Metrics:")
        if "loss" in results:
            lines.append(f"  • Loss            : {fmt_value(results.get('loss'), '.4f')}")
        lines.append(
            f"  • SeK             : {fmt_value(results.get(f'{p}sek{s}'), '.5f')}"
        )
        lines.append(
            f"  • Kappa (no-bg)   : {fmt_value(results.get(f'{p}kappa_n0{s}'), '.4f')}"
        )
        lines.append(
            f"  • Kappa (global)  : {fmt_value(results.get(f'{p}kappa{s}'), '.4f')}"
        )
        lines.append(
            f"  • IoU (foreground): {fmt_value(results.get(f'{p}iou_fg{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • IoU (background): {fmt_value(results.get(f'{p}iou_bg{s}'), '.2f', scale=100, suffix='%')}"
        )
        lines.append(
            f"  • Binary mIoU     : {fmt_value(results.get(f'{p}biou{s}'), '.2f', scale=100, suffix='%')}"
        )

        # 逐类指标
        per_class = self.per_class_metrics()
        lines.append("")
        lines.append("Per-class Breakdown:")
        lines.append(
            f"  {'Class':>6}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}"
            f"  {'IoU':>10}  {'Dice':>10}  {'Support':>8}"
        )
        gt = self._counts.gt_count
        for i in range(self.num_classes):
            tag = " (bg)" if i == self.bg_index else ""
            lines.append(
                f"  {i:>6}{tag:<5}"
                f"  {per_class[f'{p}precision_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}recall_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}f1_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}iou_{i}{s}'].item():>10.4f}"
                f"  {per_class[f'{p}dice_{i}{s}'].item():>10.4f}"
                f"  {int(gt[i].item()):>8}"
            )

        lines.append("")
        lines.append(f"Total Samples: {self.total}")

        if elapsed_time is not None or speed is not None:
            lines.append("")
            lines.append("Performance:")
            lines.append(f"  • Time         : {fmt_value(elapsed_time, '.2f', suffix='s')}")
            lines.append(f"  • Speed        : {fmt_value(speed, '.0f', suffix=' samples/sec')}")

        # 混淆矩阵摘要
        cm = self.confusion_matrix()
        if cm is not None and cm.numel() > 0:
            diag = int(torch.trace(cm))
            cm_total = int(cm.sum())
            lines.append("")
            lines.append("Confusion Matrix Summary:")
            lines.append(f"  • Diagonal (correct) : {diag}")
            lines.append(f"  • Off-diagonal (err) : {cm_total - diag}")
            lines.append(f"  • Total              : {cm_total}")

        lines.append("=" * 60)

        report_text = "\n".join(lines)
        log.info("\n%s", report_text)
        return report_text

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"bg_index={self.bg_index}, total={self.total})"
        )

