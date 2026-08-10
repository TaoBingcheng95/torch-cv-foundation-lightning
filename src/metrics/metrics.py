"""
通用分割/分类任务的评价指标模块。

分层设计（与 torchmetrics 深度集成）：
    ConfusionMatrix(Metric)    状态层
        只负责混淆矩阵的累积（update）与同步（add_state + dist_reduce_fx），
        以 property 暴露逐类 one-vs-rest 计数视图（tp/fp/fn/tn 向量），
        不推导任何指标。
    指标计算层（ClassificationMetric / SegmentationMetric）
        持有 ConfusionMatrix 子指标，从计数视图推导各自语义正确的指标；
        额外状态（如 topk 计数）通过 add_state 注册，自动参与 DDP 同步。

所有类均继承 torchmetrics.Metric，行为约定与原生 Metric/MetricCollection 对齐：
    metric = ClassificationMetric(num_classes=10, top_k=5)
    metric.update(logits, target)    # 单步累积（参数名与 torchmetrics 一致）
    results = metric.compute()       # 自动 DDP 同步 + 计算，返回 Dict[str, Tensor]
    MetricCollection({"cls": metric}) 可直接包装（含 clone/prefix 支持）。
    __init__ 透传 Metric 标准 kwargs（compute_on_cpu/sync_on_compute 等）。

混淆矩阵约定：
    shape 为 (num_classes, num_classes)，
    cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
"""
from typing import Any, Callable, Dict, Optional
import math
import torch
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors


class ConfusionMatrix(Metric):
    """
    混淆矩阵状态容器（torchmetrics Metric 子类）。

    通过 add_state 注册 matrix，自动处理：
        - DDP all_reduce（dist_reduce_fx="sum"）
        - device 迁移（跟随第一个输入 batch 的 device）
        - reset（继承自 Metric）

    二分类是 num_classes=2 的特例，无需单独的标量计数器。

    Example:
        >>> cm = ConfusionMatrix(num_classes=10)
        >>> for preds, target in dataloader:
        ...     cm.update(preds, target)   # logits 或类别索引均可
        >>> cm.tp, cm.fp, cm.fn, cm.tn     # 逐类 one-vs-rest 计数向量

    Args:
        num_classes: 类别总数（>= 2）
        ignore_index: 忽略的标签索引（如分割任务中 255 表示未标注区域），
            None 表示不忽略。分类任务通常无需设置。
    """

    # update 只做纯累积、不读取已有状态，batch 值可独立计算（torchmetrics 惯例，
    # 必须声明为类属性：Metric.__setattr__ 禁止实例级修改该常量）
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 使用 add_state 注册状态：自动 DDP 求和 + 设备跟随
        self.add_state(
            "matrix",
            default=torch.zeros((num_classes, num_classes), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    # ------------------------------------------------------------------
    # 状态维护：累积 / 重置
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch 的预测结果。

        Args:
            preds: 预测结果。
                   - logits/probabilities (N, C, ...) 自动沿 dim=1 取 argmax
                   - 类别索引 (N, ...) 直接使用
            target: 真实标签，与 argmax 后的 preds 同 shape 的整数张量
        """
        # preds 比 target 多一维 → 视为 logits，取 argmax
        if preds.dim() == target.dim() + 1:
            preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError(
                f"preds 与 target 形状不匹配: {tuple(preds.shape)} vs {tuple(target.shape)}"
            )

        # 统一搬运到矩阵所在设备（由 add_state 自动管理，通常与训练设备一致）
        preds = preds.reshape(-1).long().to(self.matrix.device)
        target = target.reshape(-1).long().to(self.matrix.device)

        # 过滤 ignore_index
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            preds, target = preds[valid], target[valid]

        # 越界标签会破坏 bincount 的索引编码，必须显式报错而非静默丢弃
        for name, t in (("preds", preds), ("target", target)):
            if t.numel() and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"{name} 存在越界类别索引: 范围 [{t.min()}, {t.max()}]，"
                    f"合法区间 [0, {self.num_classes - 1}]"
                )

        # 行=真实类别，列=预测类别；bincount 批量统计
        indices = target * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> torch.Tensor:
        """返回当前累积的混淆矩阵（DDP 同步后的值）。"""
        return self.matrix

    # ------------------------------------------------------------------
    # 派生视图：逐类 one-vs-rest 计数（均为 shape=(num_classes,) 的 int64 向量）
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """有效样本总数（已排除 ignore_index）。"""
        return int(self.matrix.sum())

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.total})"
        )


class ClassificationMetric(Metric):
    """
    分类任务指标计算层（torchmetrics Metric 子类）。

    持有 ConfusionMatrix 子指标作为唯一累积状态，所有指标从其计数视图推导；
    例外是 Top-k 准确率——它依赖 logits 排序信息，无法从混淆矩阵还原，
    因此通过 add_state 单独维护命中/总数两个标量计数。

    支持指标（compute 返回键名，命名与 torchmetrics MetricCollection 对齐）：
        - acc             总体准确率（单标签多分类下恒等于 micro P/R/F1）
        - balanced_acc    平衡准确率（= macro Recall），类别不均衡时更可靠
        - precision / recall / f1   宏平均（各类等权，对齐 torchmetrics average="macro" 默认行为）
        - weighted_f1     按类别真实频率加权的 F1
        - kappa           Cohen's Kappa，扣除随机一致性后的一致度
        - top{k}_acc      Top-k 准确率（仅当 top_k 非空且 update 传入 logits 时）
        - precision_i / recall_i / f1_i               逐类指标

    注意：
        macro 平均对验证集中未出现的类别计为 0（与 sklearn 默认行为一致）。

    Example:
        >>> metric = ClassificationMetric(num_classes=10, top_k=5)
        >>> for logits, target in dataloader:
        ...     metric.update(logits, target)
        >>> results = metric.compute()   # Dict[str, 0维 Tensor]
        >>> print(f"acc={results['acc']:.4f}, top5={results['top5_acc']:.4f}")

        命名约定：precision/recall/f1 为宏平均（各类等权），与
        torchmetrics.MetricCollection(average="macro") 的键名一致。
        balanced_acc 等价于 macro recall，额外提供语义更明确的别名。

    Args:
        num_classes: 类别总数（>= 2）
        top_k: 额外统计 Top-k 准确率，None 表示不统计
        ignore_index: 忽略的标签索引，分类任务通常保持 None
    """

    # update 只做纯累积、不读取已有状态，batch 值可独立计算（torchmetrics 惯例，
    # 必须声明为类属性：Metric.__setattr__ 禁止实例级修改该常量）
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        top_k: Optional[int] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if top_k is not None and not 1 <= top_k <= num_classes:
            raise ValueError(
                f"top_k 需在 [1, num_classes={num_classes}] 内，实际得到 {top_k}"
            )
        # 嵌套 ConfusionMatrix 子指标，其 add_state 状态会自动参与父指标的 DDP 同步
        self.cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)
        self.top_k = top_k
        # Top-k 计数也通过 add_state 注册，自动 DDP 同步
        if top_k is not None:
            self.add_state(
                "topk_correct",
                default=torch.tensor(0, dtype=torch.int64),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "topk_total",
                default=torch.tensor(0, dtype=torch.int64),
                dist_reduce_fx="sum",
            )

    @property
    def num_classes(self) -> int:
        return self.cm.num_classes

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.matrix

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        """DDP 状态同步：外层状态 + 嵌套 ConfusionMatrix 子指标。

        compute() 由框架 _wrap_compute 包装，自动在计算前经 sync_context 调用本方法。
        但框架的 sync()/unsync() 只缓存与恢复外层自身的 add_state 状态，
        嵌套子指标的混淆矩阵不在其中，因此这里让子指标走自己的 sync() 机制
        （内部会把本地状态缓存进它自己的 _cache），compute() 结束前再手动
        unsync 恢复子指标本地状态，避免全局同步值污染后续累积。
        """
        # 外层状态（topk_correct/topk_total，若启用 top_k）由 super() 同步，
        # 框架 unsync() 会经 _copy_state_dict 缓存自动恢复，无需额外处理
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)
        # 嵌套子指标：框架的 sync()/unsync() 不感知嵌套 Metric 的状态，
        # 需显式调用子指标自己的 sync()（内部会把本地 matrix 缓存进它的 _cache）；
        # 非分布式环境下 sync() 自动判定后直接返回，无需外层包判断
        self.cm.sync(dist_sync_fn=dist_sync_fn, process_group=process_group)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。先累积 Top-k（需在 argmax 前拿到完整 logits），
        再交给嵌套的 ConfusionMatrix 子指标。
        """
        # 先累积 Top-k（需在 argmax 前拿到完整 logits）
        if self.top_k is not None and preds.dim() == target.dim() + 1:
            self._update_topk(preds, target)
        # 手动调用嵌套 CM 的 update（torchmetrics 不会自动转发）
        self.cm.update(preds, target)

    def _update_topk(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        # (N, C, ...) 沿类别维取 top-k 索引 → (N, k, ...)
        topk_idx = logits.topk(self.top_k, dim=1).indices
        hit = (topk_idx == target.unsqueeze(1).long()).any(dim=1)  # (N, ...)
        if self.cm.ignore_index is not None:
            valid = target != self.cm.ignore_index
            hit = hit[valid]
        self.topk_correct += int(hit.sum())
        self.topk_total += int(hit.numel())

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        从累积状态计算全部分类指标。

        注：本方法由框架 _wrap_compute 包装，DDP 下会先经 sync_context 调用
        _sync_dist 完成状态同步（外层 topk 状态 + 嵌套 CM 的混淆矩阵）。
        框架的 unsync 只恢复外层自身缓存，嵌套子指标需在返回前手动 unsync，
        否则 compute 后继续累积会累加到全局同步值上。

        Returns:
            汇总指标 + 逐类指标（precision_i/recall_i/f1_i，
            键名格式与 trainer 日志分组正则 `^(.+)_(\\d+)$` 兼容），
            值均为 0 维 Tensor（与 torchmetrics 原生指标的 compute 返回约定一致）
        """
        try:
            return self._compute_from_matrix()
        finally:
            # 嵌套子指标的本地状态恢复（非分布式环境下 _is_synced 恒为 False，直接跳过）
            if self.cm._is_synced:
                self.cm.unsync()

    def _compute_from_matrix(self) -> Dict[str, torch.Tensor]:
        """从当前累积状态推导全部分类指标（纯计算，无同步/缓存副作用）。"""
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        # 在 CPU 上计算以兼容 MPS 等不支持 float64 的设备
        tp = self.cm.tp.cpu().float()
        gt_count = self.cm.gt_count.cpu().float()
        pred_count = self.cm.pred_count.cpu().float()
        total = self.cm.total

        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        acc = tp.sum() / (total + eps)
        # 按真实频率加权的 F1
        freq = gt_count / (total + eps)
        weighted_f1 = (freq * f1).sum()
        # Cohen's Kappa：pe 为边缘分布下的期望一致率
        pe = (gt_count * pred_count).sum() / (total ** 2 + eps)
        kappa = (acc - pe) / (1 - pe + eps)

        results = {
            'acc': acc,
            'balanced_acc': recall.mean(),
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'weighted_f1': weighted_f1,
            'kappa': kappa,
        }
        if self.top_k is not None and self.topk_total > 0:
            results[f'top{self.top_k}_acc'] = self.topk_correct / self.topk_total

        # # 逐类详细指标
        # for i in range(self.num_classes):
        #     results[f'precision_{i}'] = precision[i]
        #     results[f'recall_{i}'] = recall[i]
        #     results[f'f1_{i}'] = f1[i]

        return results

    def reset(self) -> None:
        """重置全部累积状态（含嵌套 ConfusionMatrix 子指标）。"""
        super().reset()
        self.cm.reset()

    def forward(self, *args, **kwargs):
        """一步完成累积 + 当前 batch 计算。

        重写以修复 torchmetrics 1.9.0 在嵌套子指标场景下的状态恢复 bug：
        原 _forward_full_state_update 仅保存/恢复外层 _defaults，
        嵌套 CM 的状态在 reset 后未被清零，导致二次 update 时子指标被重复累加。

        注意：
            - add_state 默认 persistent=False，state_dict() 不包含非 buffer 状态，
              因此使用 _copy_state_dict() 进行保存与恢复。
            - batch 值用 _compute_from_matrix() 而非 compute()：后者经 _wrap_compute
              包装，会把 batch 值写入 _computed 缓存（DDP 下还会触发逐 batch 同步），
              导致随后真正的 compute() 返回陈旧缓存。
        """
        # 全局累积
        self.update(*args, **kwargs)

        # 保存完整状态（含嵌套子指标）
        outer_cache = self._copy_state_dict()
        inner_cache = self.cm._copy_state_dict()

        # 清零 + 仅当前 batch 计算
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self._compute_from_matrix()

        # 恢复完整状态
        self.load_state_dict(outer_cache, strict=False)
        self.cm.load_state_dict(inner_cache, strict=False)

        return batch_val

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"top_k={self.top_k}, total={self.cm.total})"
        )


class SegmentationMetric(Metric):
    """
    语义分割任务指标计算层（torchmetrics Metric 子类）。

    与 ClassificationMetric 同构：持有 ConfusionMatrix 子指标作为唯一累积状态，
    所有指标从其计数视图推导。每个像素视为一个独立样本，
    输入 preds/target 的 shape 通常为 (N, C, H, W) logits 或 (N, H, W) 类别索引。

    支持指标（compute 返回键名）：
        - oa              Overall Accuracy，总体像素精度
        - mpa             Mean Pixel Accuracy，平均类别像素精度（= macro Recall）
        - miou            Mean IoU，平均交并比（分割主监控指标）
        - fwiou           Frequency Weighted IoU，按类别频率加权的 IoU
        - mf1             Mean F1（= mean Dice）
        - iou_i / precision_i / recall_i / f1_i       逐类指标

    Example:
        >>> metric = SegmentationMetric(num_classes=21, ignore_index=255)
        >>> # preds: (N, 21, H, W) logits 或 (N, H, W) 类别索引
        >>> metric.update(preds, target)
        >>> results = metric.compute()   # Dict[str, 0维 Tensor]
        >>> print(f"mIoU: {results['miou']:.4f}")

    Args:
        num_classes: 类别总数（含背景类，>= 2）
        ignore_index: 忽略的标签索引，默认 255（VOC 等数据集的 void 区域约定）
    """

    # update 只做纯累积、不读取已有状态，batch 值可独立计算（torchmetrics 惯例，
    # 必须声明为类属性：Metric.__setattr__ 禁止实例级修改该常量）
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: Optional[int] = 255, **kwargs: Any):
        super().__init__(**kwargs)
        # 嵌套 ConfusionMatrix 子指标
        self.cm = ConfusionMatrix(num_classes, ignore_index=ignore_index)

    @property
    def num_classes(self) -> int:
        return self.cm.num_classes

    @property
    def ignore_index(self) -> Optional[int]:
        return self.cm.ignore_index

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.matrix

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        """DDP 状态同步：嵌套 ConfusionMatrix 子指标。

        与 ClassificationMetric 同理：框架的 sync()/unsync() 只覆盖外层自身状态，
        嵌套子指标需显式调用它自己的 sync()（内部缓存本地状态），
        compute() 结束前再手动 unsync 恢复。
        """
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)
        self.cm.sync(dist_sync_fn=dist_sync_fn, process_group=process_group)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """累积一个 batch，输入约定与 ConfusionMatrix.update 一致。"""
        self.cm.update(preds, target)

    def reset(self) -> None:
        """重置全部累积状态（含嵌套 ConfusionMatrix 子指标）。"""
        super().reset()
        self.cm.reset()

    def forward(self, *args, **kwargs):
        """一步完成累积 + 当前 batch 计算。

        重写以修复 torchmetrics 1.9.0 在嵌套子指标场景下的状态恢复 bug。
        add_state 默认 persistent=False，需使用 _copy_state_dict() 保存/恢复；
        batch 值用 _compute_from_matrix() 而非 compute()，避免 _wrap_compute
        的 _computed 缓存污染与逐 batch DDP 同步。
        """
        self.update(*args, **kwargs)

        outer_cache = self._copy_state_dict()
        inner_cache = self.cm._copy_state_dict()

        self.reset()
        self.update(*args, **kwargs)
        batch_val = self._compute_from_matrix()

        self.load_state_dict(outer_cache, strict=False)
        self.cm.load_state_dict(inner_cache, strict=False)

        return batch_val

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        从累积状态计算全部分割指标。

        注：本方法由框架 _wrap_compute 包装，DDP 下会先经 sync_context 调用
        _sync_dist 完成嵌套 CM 混淆矩阵的同步；嵌套子指标需在返回前手动
        unsync，否则 compute 后继续累积会累加到全局同步值上。

        Returns:
            汇总指标（oa/mpa/miou/fwiou/mf1）+ 逐类指标
            （iou_i/precision_i/recall_i/f1_i，键名格式与 trainer
            日志分组正则 `^(.+)_(\\d+)$` 兼容），值均为 0 维 Tensor
        """
        try:
            return self._compute_from_matrix()
        finally:
            # 嵌套子指标的本地状态恢复（非分布式环境下 _is_synced 恒为 False，直接跳过）
            if self.cm._is_synced:
                self.cm.unsync()

    def _compute_from_matrix(self) -> Dict[str, torch.Tensor]:
        """从当前混淆矩阵状态推导全部分割指标（纯计算，无同步/缓存副作用）。"""
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        # 在 CPU 上计算以兼容 MPS 等不支持 float64 的设备
        tp = self.cm.tp.cpu().float()
        gt_count = self.cm.gt_count.cpu().float()
        pred_count = self.cm.pred_count.cpu().float()
        total = self.cm.total

        # ---- Overall Accuracy ----
        oa = tp.sum() / (total + eps)

        # ---- 逐类 Recall（Pixel Accuracy per class）与 Precision ----
        recall = tp / (gt_count + eps)
        precision = tp / (pred_count + eps)
        mpa = recall.mean()

        # ---- 逐类 IoU: TP / (GT + Pred - TP) ----
        iou = tp / (gt_count + pred_count - tp + eps)
        miou = iou.mean()

        # ---- Frequency Weighted IoU ----
        freq = gt_count / (total + eps)
        fwiou = (freq * iou).sum()

        # ---- 逐类 F1（= Dice）----
        f1 = 2 * precision * recall / (precision + recall + eps)
        mf1 = f1.mean()

        results = {
            'oa': oa,
            'mpa': mpa,
            'miou': miou,
            'fwiou': fwiou,
            'mf1': mf1,
        }

        # 逐类详细指标
        for i in range(self.num_classes):
            results[f'iou_{i}'] = iou[i]
            results[f'precision_{i}'] = precision[i]
            results[f'recall_{i}'] = recall[i]
            results[f'f1_{i}'] = f1[i]

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.cm.total})"
        )


# ------------------------------------------------------------------
# 工具函数：独立于 Metric 生命周期的纯计算函数
# ------------------------------------------------------------------


def _kappa_from_matrix(matrix: torch.Tensor, eps: float = 1e-10) -> float:
    """
    从混淆矩阵计算 Cohen's Kappa。

    边界约定与 SCD 官方实现一致：矩阵全零或期望一致率 pe 为 1 时返回 0。

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别
    """
    hist = matrix.cpu().float()
    total = hist.sum()
    if total == 0:
        return 0.0
    po = hist.diagonal().sum() / total
    pe = (hist.sum(dim=1) * hist.sum(dim=0)).sum() / total ** 2
    if abs(1 - pe) < eps:
        return 0.0
    return ((po - pe) / (1 - pe)).item()


def separated_kappa(matrix: torch.Tensor, bg_index: int = 0) -> Dict[str, float]:
    """
    分离 Kappa（Separated Kappa, SeK）系数。

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
        >>> cm = ConfusionMatrix(num_classes=5)
        >>> cm.update(preds, targets)
        >>> results = separated_kappa(cm.matrix)
        >>> print(f"SeK: {results['sek']:.5f}")

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别，
            可直接传入 ConfusionMatrix.matrix 或 SegmentationMetric.confusion_matrix
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
    hist = matrix.cpu().float()
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
