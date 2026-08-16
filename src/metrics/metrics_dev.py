"""
通用分割/分类任务的评价指标模块。

架构设计：
    ConfusionMatrixView (Property View) 计算视图层
        纯计算容器：接收 matrix 张量，通过 property 暴露逐类 one-vs-rest 
        计数视图（tp/fp/fn/tn 向量），不持有任何状态，不参与 DDP 同步。
        提供 from_predictions classmethod 从预测批量构造增量视图，
        将 argmax/ignore_index 过滤/bincount 编码等输入处理逻辑收敛于此。
    
    ClassificationMetric / SegmentationMetric (State + Compute)
        利用 torchmetrics.Metric 的 add_state 管理 matrix 状态，实现 DDP 自动同步。
        update 阶段调用 ConfusionMatrixView.from_predictions 取得增量并累加，
        compute 阶段用 ConfusionMatrixView(self.matrix) 包装后读取派生计数计算指标。

    SeparatedKappaMetric (自定义指标开发参考)
        继承 torchmetrics 内置 MulticlassConfusionMatrix，复用其 confmat 状态和 update 逻辑，
        仅覆写 compute() 实现自定义指标推导。展示了"站在 torchmetrics 肩膀上"开发自定义指标的标准范式。

所有指标类均继承 torchmetrics.Metric（或其子类），行为约定与原生 Metric 对齐：
    metric = ClassificationMetric(num_classes=10, top_k=5)
    metric.update(logits, target)    # 单步累积
    results = metric.compute()       # 自动 DDP 同步 + 计算

混淆矩阵约定：
    shape 为 (num_classes, num_classes)，
    cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
"""
from typing import Any, Dict, Optional, Literal
import math
import warnings
import copy
import torch

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix


################


class ConfusionMatrixView:
    """
    混淆矩阵计算视图（无状态纯容器）。
    
    接收 matrix 张量，通过 property 暴露 tp/fp/fn/tn 等派生视图。
    不参与 DDP 同步，不继承 Metric，仅负责指标计算的语义封装。
    from_predictions classmethod 负责从预测批量构造增量视图，
    将 argmax/ignore_index 过滤/越界校验/bincount 编码等输入处理收敛于此。
    
    Example:
        >>> matrix = torch.tensor([[10, 2], [3, 5]], dtype=torch.int64)
        >>> cm = ConfusionMatrixView(matrix)
        >>> cm.tp, cm.fp, cm.fn
        >>> # 从预测批量构造增量视图
        >>> delta = ConfusionMatrixView.from_predictions(preds, target, num_classes=3)
    """

    def __init__(self, matrix: torch.Tensor):
        """
        Args:
            matrix: 混淆矩阵 (num_classes, num_classes)，int64 类型
        """
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix 需为方阵，实际 shape={tuple(matrix.shape)}")
        self._matrix = matrix

    @classmethod
    def from_predictions(
        cls,
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: Optional[int] = None,
    ) -> "ConfusionMatrixView":
        """
        从一个 batch 的预测构造增量混淆矩阵视图（纯函数语义，无副作用）。

        处理逻辑：
            - logits/probabilities (N, C, ...) 自动沿 dim=1 取 argmax
            - 类别索引 (N, ...) 直接使用
            - 过滤 ignore_index 标签
            - 越界索引显式报错（避免破坏 bincount 编码）
            - 行=真实类别，列=预测类别，bincount 批量统计

        Args:
            preds: 预测结果 (logits 或类别索引)
            target: 真实标签，与 argmax 后的 preds 同 shape 的整数张量
            num_classes: 类别总数
            ignore_index: 忽略的标签索引，None 表示不忽略

        Returns:
            单批次增量混淆矩阵对应的 ConfusionMatrixView 视图
        """
        if preds.dim() == target.dim() + 1:
            # 单通道 (N, 1, ...) 输出 argmax 恒为 0，会静默统计错误，必须拦截
            if preds.shape[1] < 2:
                raise ValueError(
                    f"logits 的通道维需 >= 2，实际得到 {preds.shape[1]}；"
                    "单通道输出请先阈值化后传入类别索引"
                )
            preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError(
                f"preds 与 target 形状不匹配: {tuple(preds.shape)} vs {tuple(target.shape)}"
            )

        preds = preds.reshape(-1).long()
        target = target.reshape(-1).long()

        if ignore_index is not None:
            valid = target != ignore_index
            preds, target = preds[valid], target[valid]

        for name, t in (("preds", preds), ("target", target)):
            if t.numel() and (t.min() < 0 or t.max() >= num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {num_classes - 1}]"
                )

        indices = target * num_classes + preds
        counts = torch.bincount(indices, minlength=num_classes ** 2)
        return cls(counts.reshape(num_classes, num_classes))

    @property
    def matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)。"""
        return self._matrix

    @property
    def num_classes(self) -> int:
        return self._matrix.shape[0]

    @property
    def total(self) -> int:
        """有效样本总数。"""
        return int(self._matrix.sum())

    @property
    def gt_count(self) -> torch.Tensor:
        """每类真实样本数（行和）。"""
        return self._matrix.sum(dim=1)

    @property
    def pred_count(self) -> torch.Tensor:
        """每类被预测样本数（列和）。"""
        return self._matrix.sum(dim=0)

    @property
    def tp(self) -> torch.Tensor:
        """每类正确预测数（对角线）。"""
        return self._matrix.diagonal()

    @property
    def fp(self) -> torch.Tensor:
        """每类误报数。"""
        return self.pred_count - self.tp

    @property
    def fn(self) -> torch.Tensor:
        """每类漏检数。"""
        return self.gt_count - self.tp

    @property
    def tn(self) -> torch.Tensor:
        """每类真阴数。"""
        return self.total - self.tp - self.fp - self.fn

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"total={self.total})"
        )

    def compute(self) -> torch.Tensor:
        """返回原始混淆矩阵张量（API 兼容，与 torchmetrics.ConfusionMatrix 对齐）。"""
        return self._matrix


class _CMAdapter:
    """
    混淆矩阵状态适配器（轻量桥接对象）。

    将 Metric 内部的 matrix 状态通过统一的 .cm.compute() 接口暴露给下游，
    与 general.py（torchmetrics ConfusionMatrix）和 metrics.py（ConfusionMatrixAccumulator）
    的 .cm.compute() 访问风格对齐。

    不持有任何独立状态，始终通过 metric 引用读取最新的 matrix（含 DDP 同步后的值）。
    不继承 nn.Module，不会被注册为 Metric 的子模块。
    """

    def __init__(self, metric: Metric):
        self._metric = metric

    def compute(self) -> torch.Tensor:
        """返回原始混淆矩阵 (num_classes, num_classes)，int64 张量。"""
        return self._metric.matrix


class ClassificationMetric(Metric):
    """
    分类任务指标计算层（torchmetrics Metric 子类）。

    通过 add_state 管理 matrix 状态，利用无状态 ConfusionMatrixView 视图计算指标。
    接口风格与 metrics.py / general.py 对齐。

    compute() 返回汇总指标：
        - acc             总体准确率
        - balanced_acc    平衡准确率（= macro Recall）
        - precision       精确率（聚合方式由 average 参数控制）
        - recall          召回率（聚合方式由 average 参数控制）
        - f1              F1 分数（聚合方式由 average 参数控制）
        - kappa           Cohen's Kappa

    按需获取：
        - topk_acc()           Top-k 准确率
        - per_class_metrics()  逐类 precision/recall/f1
        - cm.compute()         混淆矩阵（与 general.py / metrics.py 接口对齐）

    Args:
        num_classes: 类别总数（>= 2）
        average: 多类别聚合方式，"macro"/"micro"/"weighted"，默认 "macro"
        top_k: 额外统计 Top-k 准确率，None 表示不统计
        ignore_index: 忽略的标签索引，分类任务通常保持 None
        prefix: 指标键名前缀（如 'val/'）
        postfix: 指标键名后缀（如 '_mean'）
        **kwargs: torchmetrics Metric 标准参数
    """

    # update() 仅追加局部 matrix，不读取其他 rank 状态 → 安全关闭全量同步
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted"] = "macro",
        top_k: Optional[int] = None,
        ignore_index: Optional[int] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
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
        self.prefix = prefix or ''
        self.postfix = postfix or ''

        self.add_state(
            "matrix",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        if top_k is not None:
            self.add_state(
                "topk_correct",
                default=torch.zeros((), dtype=torch.int64),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "topk_total",
                default=torch.zeros((), dtype=torch.int64),
                dist_reduce_fx="sum",
            )
        self._topk_index_warned = False
        # 混淆矩阵状态适配器，暴露 .cm.compute() 供下游统一访问
        self.cm = _CMAdapter(self)

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)。"""
        return self.matrix

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。先累积 Top-k（需在 argmax 前拿到完整 logits），
        再通过 ConfusionMatrixView.from_predictions 取得增量并累加。
        """
        if self.top_k is not None:
            if preds.dim() == target.dim() + 1:
                self._update_topk(preds, target)
            elif not self._topk_index_warned:
                warnings.warn(
                    "top_k 已启用，但传入的 preds 是类别索引而非 logits，"
                    "Top-k 准确率将无法统计；请在 update 中传入 (N, C, ...) 形状的 logits"
                )
                self._topk_index_warned = True

        delta = ConfusionMatrixView.from_predictions(
            preds, target, self.num_classes, self.ignore_index
        )
        self.matrix += delta.matrix.to(self.matrix.device)

    def _update_topk(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        topk_idx = logits.topk(self.top_k, dim=1).indices
        hit = (topk_idx == target.unsqueeze(1).long()).any(dim=1)
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            hit = hit[valid]
        self.topk_correct += hit.sum().to(self.topk_correct.device)
        self.topk_total += hit.numel()

    def compute(self) -> Dict[str, torch.Tensor]:
        """从累积状态计算汇总分类指标。"""
        cm = ConfusionMatrixView(self.matrix)
        eps = 1e-10

        tp = cm.tp.double()
        gt_count = cm.gt_count.double()
        pred_count = cm.pred_count.double()
        total = cm.total

        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = tp.sum() / (total + eps)

        if self.average == "micro":
            agg_precision = acc
            agg_recall = acc
            agg_f1 = acc
        elif self.average == "weighted":
            freq = gt_count / (total + eps)
            agg_precision = (freq * precision).sum()
            agg_recall = (freq * recall).sum()
            agg_f1 = (freq * f1).sum()
        else:  # macro
            agg_precision = precision.mean()
            agg_recall = recall.mean()
            agg_f1 = f1.mean()

        pe = (gt_count * pred_count).sum() / ((total + eps) ** 2)
        if 1 - pe > eps:
            kappa = (acc - pe) / (1 - pe)
        else:
            kappa = torch.zeros((), dtype=torch.float64)

        return {
            f'{self.prefix}acc{self.postfix}': acc,
            f'{self.prefix}balanced_acc{self.postfix}': recall.mean(),
            f'{self.prefix}precision{self.postfix}': agg_precision,
            f'{self.prefix}recall{self.postfix}': agg_recall,
            f'{self.prefix}f1{self.postfix}': agg_f1,
            f'{self.prefix}kappa{self.postfix}': kappa,
        }

    def per_class_metrics(self) -> Dict[str, torch.Tensor]:
        """逐类详细指标（按需获取，不包含在 compute() 输出中）。"""
        cm = ConfusionMatrixView(self.matrix)
        eps = 1e-10
        tp = cm.tp.double()
        gt_count = cm.gt_count.double()
        pred_count = cm.pred_count.double()
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        p, s = self.prefix, self.postfix
        results = {}
        for i in range(self.num_classes):
            results[f'{p}precision_{i}{s}'] = precision[i]
            results[f'{p}recall_{i}{s}'] = recall[i]
            results[f'{p}f1_{i}{s}'] = f1[i]
        return results

    def topk_acc(self) -> Optional[torch.Tensor]:
        """Top-k 准确率（按需获取，不包含在 compute() 输出中）。"""
        if self.top_k is None or self.topk_total == 0:
            return None
        return self.topk_correct.double() / self.topk_total

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "ClassificationMetric":
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        new_metric._computed = None  # 清除 torchmetrics compute 缓存，确保新 prefix/postfix 生效
        return new_metric

    def metric_keys(self) -> list:
        """返回 compute() 输出中的键名列表（无需调用 compute）。"""
        p, s = self.prefix, self.postfix
        return [f'{p}acc{s}', f'{p}balanced_acc{s}',
                f'{p}precision{s}', f'{p}recall{s}', f'{p}f1{s}', f'{p}kappa{s}']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"top_k={self.top_k}, total={self.matrix.sum().item()})"
        )


class SegmentationMetric(Metric):
    """
    语义分割任务指标计算层（torchmetrics Metric 子类）。

    与 ClassificationMetric 同构：通过 add_state 管理 matrix 状态，
    利用无状态 ConfusionMatrixView 视图计算指标。

    compute() 返回汇总指标：
        - oa              Overall Accuracy
        - iou             平均交并比（聚合方式由 average 参数控制）
        - f1              平均 F1（聚合方式由 average 参数控制）

    按需获取：
        - per_class_metrics()  逐类 iou/precision/recall/f1
        - cm.compute()         混淆矩阵（与 general.py / metrics.py 接口对齐）

    Args:
        num_classes: 类别总数（含背景类，>= 2）
        average: 多类别聚合方式，"macro"/"micro"/"weighted"，默认 "macro"
        ignore_index: 忽略的标签索引，默认 255
        prefix: 指标键名前缀
        postfix: 指标键名后缀
        **kwargs: torchmetrics Metric 标准参数
    """

    # update() 仅追加局部 matrix，不读取其他 rank 状态 → 安全关闭全量同步
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted"] = "macro",
        ignore_index: Optional[int] = 255,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"average 需为 'macro'/'micro'/'weighted' 之一，实际得到 '{average}'")
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.prefix = prefix or ''
        self.postfix = postfix or ''

        self.add_state(
            "matrix",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        # 混淆矩阵状态适配器，暴露 .cm.compute() 供下游统一访问
        self.cm = _CMAdapter(self)

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)。"""
        return self.matrix

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """累积一个 batch，通过 ConfusionMatrixView.from_predictions 取得增量并累加。"""
        delta = ConfusionMatrixView.from_predictions(
            preds, target, self.num_classes, self.ignore_index
        )
        self.matrix += delta.matrix.to(self.matrix.device)

    def compute(self) -> Dict[str, torch.Tensor]:
        """从累积状态计算汇总分割指标。"""
        cm = ConfusionMatrixView(self.matrix)
        eps = 1e-10

        tp = cm.tp.double()
        gt_count = cm.gt_count.double()
        pred_count = cm.pred_count.double()
        total = cm.total

        oa = tp.sum() / (total + eps)
        recall = tp / (gt_count + eps)
        precision = tp / (pred_count + eps)
        iou = tp / (gt_count + pred_count - tp + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        if self.average == "micro":
            agg_iou = oa
            agg_f1 = oa
        elif self.average == "weighted":
            freq = gt_count / (total + eps)
            agg_iou = (freq * iou).sum()
            agg_f1 = (freq * f1).sum()
        else:  # macro
            agg_iou = iou.mean()
            agg_f1 = f1.mean()

        return {
            f'{self.prefix}oa{self.postfix}': oa,
            f'{self.prefix}iou{self.postfix}': agg_iou,
            f'{self.prefix}f1{self.postfix}': agg_f1,
        }

    def per_class_metrics(self) -> Dict[str, torch.Tensor]:
        """逐类详细指标（按需获取，不包含在 compute() 输出中）。"""
        cm = ConfusionMatrixView(self.matrix)
        eps = 1e-10
        tp = cm.tp.double()
        gt_count = cm.gt_count.double()
        pred_count = cm.pred_count.double()
        recall = tp / (gt_count + eps)
        precision = tp / (pred_count + eps)
        iou = tp / (gt_count + pred_count - tp + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        p, s = self.prefix, self.postfix
        results = {}
        for i in range(self.num_classes):
            results[f'{p}iou_{i}{s}'] = iou[i]
            results[f'{p}precision_{i}{s}'] = precision[i]
            results[f'{p}recall_{i}{s}'] = recall[i]
            results[f'{p}f1_{i}{s}'] = f1[i]
        return results

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "SegmentationMetric":
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        new_metric._computed = None  # 清除 torchmetrics compute 缓存，确保新 prefix/postfix 生效
        return new_metric

    def metric_keys(self) -> list:
        """返回 compute() 输出中的键名列表（无需调用 compute）。"""
        p, s = self.prefix, self.postfix
        return [f'{p}oa{s}', f'{p}iou{s}', f'{p}f1{s}']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.matrix.sum().item()})"
        )


############ 自定义指标：继承 torchmetrics 内置类的开发参考 ############


class SeparatedKappaMetric(MulticlassConfusionMatrix):
    """
    分离 Kappa（Separated Kappa, SeK）系数（torchmetrics Metric 子类）。

    继承 torchmetrics.classification.MulticlassConfusionMatrix：
        - 复用父类的 confmat 状态（N×N int64 混淆矩阵，DDP 下 sum 归约）
        - 复用父类的 update()（argmax / ignore_index / bincount 输入处理）
        - 覆写 compute() 实现 SeK 自定义推导逻辑

    这是"站在 torchmetrics 肩膀上开发自定义指标"的标准范式：
    当自定义指标的状态容器与某个 torchmetrics 内置指标相同时，
    直接继承该内置类，只覆写 compute()，无需重复实现状态管理和输入处理。

    面向场景：
        "多类前景 + 主导性背景"的语义变化检测（SCD）、灾损分级等。
        传统 Kappa 会被巨量的"背景→背景"正确项虚高，SeK 将其剔除后
        计算前景 Kappa，并乘以前景二值 IoU 的指数惩罚项：

            SeK = kappa_n0 * exp(IoU_fg - 1)

    compute() 返回：
        - sek        分离 Kappa 系数
        - kappa_n0   剔除背景 TN 后的 Kappa（前景语义分类一致度）
        - iou_fg     前景（变化区域）二值 IoU
        - iou_bg     背景（未变化区域）二值 IoU
        - biou       二值 mIoU = (iou_fg + iou_bg) / 2

    参考：SECOND 数据集官方指标 https://captain-whu.github.io/SCD/

    Example:
        >>> metric = SeparatedKappaMetric(num_classes=5, bg_index=0, prefix='val/')
        >>> metric.update(preds, target)   # (N, 5, H, W) logits 或 (N, H, W) 索引
        >>> results = metric.compute()
        >>> print(f"SeK: {results['val/sek']:.5f}")

    Args:
        num_classes: 类别总数（含背景类，>= 2）
        bg_index: 背景/未变化类的索引，默认 0（SCD 官方约定）
        ignore_index: 忽略的标签索引，默认 255
        prefix: 指标键名前缀
        postfix: 指标键名后缀
        **kwargs: torchmetrics Metric 标准参数
    """

    def __init__(
        self,
        num_classes: int,
        bg_index: int = 0,
        ignore_index: Optional[int] = 255,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if not 0 <= bg_index < num_classes:
            raise ValueError(
                f"bg_index 越界: {bg_index}，合法区间 [0, {num_classes - 1}]"
            )
        self.bg_index = bg_index
        self.prefix = prefix or ''
        self.postfix = postfix or ''

    # ------------------------------------------------------------------
    # 计算工具
    # ------------------------------------------------------------------

    @staticmethod
    def _kappa_from_matrix(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        从混淆矩阵计算 Cohen's Kappa（张量版本，支持梯度图/设备一致）。

        边界约定与 SCD 官方实现一致：矩阵全零或 pe≈1 时返回 0。
        """
        hist = matrix.double()
        total = hist.sum()
        if total == 0:
            return torch.zeros((), dtype=torch.float64, device=matrix.device)
        po = hist.diagonal().sum() / total
        pe = (hist.sum(dim=1) * hist.sum(dim=0)).sum() / total ** 2
        if 1 - pe > eps:
            return (po - pe) / (1 - pe)
        return torch.zeros((), dtype=torch.float64, device=matrix.device)

    # ------------------------------------------------------------------
    # torchmetrics 接口
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        从累积混淆矩阵计算 SeK 指标。

        继承自父类的 self.confmat 在 compute() 时已由 torchmetrics
        完成 DDP 同步（all_reduce sum），此处直接读取即可。
        """
        matrix = self.confmat
        n = matrix.shape[0]
        bg = self.bg_index
        eps = 1e-10

        # ---- Step 1: 剔除"背景→背景"TN，计算前景语义 Kappa ----
        hist_n0 = matrix.double().clone()
        hist_n0[bg, bg] = 0
        kappa_n0 = self._kappa_from_matrix(hist_n0)

        # ---- Step 2: 折叠为二值 (bg/fg)，计算空间定位 IoU ----
        hist = matrix.double()
        tn = hist[bg, bg]                         # 背景判对
        fn = hist[:, bg].sum() - tn               # 前景漏检
        fp = hist[bg, :].sum() - tn               # 背景误检
        tp = hist.sum() - tn - fp - fn            # 前景判对（含类间混淆）

        iou_fg = tp / (tp + fp + fn + eps)
        iou_bg = tn / (tn + fp + fn + eps)

        # ---- Step 3: SeK = kappa_n0 * exp(IoU_fg - 1) ----
        sek = kappa_n0 * torch.exp(iou_fg - 1)

        p, s = self.prefix, self.postfix
        return {
            f'{p}sek{s}':      sek,
            f'{p}kappa_n0{s}': kappa_n0,
            f'{p}iou_fg{s}':   iou_fg,
            f'{p}iou_bg{s}':   iou_bg,
            f'{p}biou{s}':     (iou_fg + iou_bg) / 2,
        }

    def primary_value(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取主指标值（用于 early stopping / model selection 等标量比较场景）。"""
        return results[f'{self.prefix}sek{self.postfix}']

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "SeparatedKappaMetric":
        """深拷贝指标实例，可选覆盖 prefix / postfix。"""
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        new_metric._computed = None  # 清除 torchmetrics compute 缓存，确保新 prefix/postfix 生效
        return new_metric

    def metric_keys(self) -> list:
        """返回 compute() 输出中的键名列表（无需调用 compute）。"""
        p, s = self.prefix, self.postfix
        return [f'{p}sek{s}', f'{p}kappa_n0{s}',
                f'{p}iou_fg{s}', f'{p}iou_bg{s}', f'{p}biou{s}']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"bg_index={self.bg_index}, total={int(self.confmat.sum())})"
        )
