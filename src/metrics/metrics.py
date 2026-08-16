"""
基于原生Pytorch的通用分割/分类任务的评价指标模块。

定位：
    仅依赖 torch、不依赖 torchmetrics，适用于：
        1. 自定义指标模块开发的学习与参考（基于混淆矩阵的指标推导范式）；
        2. 无 torchmetrics 依赖环境下的后备选择。
    其余两个模块的定位：
        general.py     基于 torchmetrics 封装的常用指标集合（开箱即用）；
        metrics_dev.py 开发中的、集成 torchmetrics 特性的指标类实现。

分层设计：
    ConfusionMatrixAccumulator（状态层）
        只负责混淆矩阵的累积（update）、重置（reset）、合并（__add__ / all_reduce），
        并以 property 暴露逐类 one-vs-rest 计数视图（tp/fp/fn/tn 向量），
        不推导任何指标。
    指标计算层（后续的 ClassificationMetric / SegmentationMetric）
        持有 ConfusionMatrixAccumulator，从计数视图推导各自语义正确的指标。

混淆矩阵约定：
    shape 为 (num_classes, num_classes)，
    cm[i, j] 表示真实类别为 i、被预测为 j 的样本数量。
"""
from typing import Optional, Dict, Literal
import math
import copy
import warnings
import torch
import torch.distributed as dist

try:
    import torchmetrics
    _HAS_TORCHMETRICS = True
except ImportError:
    torchmetrics = None
    _HAS_TORCHMETRICS = False



class ConfusionMatrixAccumulator:
    """
    混淆矩阵状态容器（参数接收层）。

    只做三件事：累积、重置、合并，指标推导交给上层计算类。
    二分类是 num_classes=2 的特例，无需单独的标量计数器。

    Example:
        >>> cm = ConfusionMatrixAccumulator(num_classes=10)
        >>> for preds, target in dataloader:
        ...     cm.update(preds, target)   # logits 或类别索引均可
        >>> cm.tp, cm.fp, cm.fn, cm.tn     # 逐类 one-vs-rest 计数向量

    Args:
        num_classes: 类别总数（>= 2）
        ignore_index: 忽略的标签索引（如分割任务中 255 表示未标注区域），
            None 表示不忽略。分类任务通常无需设置。
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        if num_classes < 2:
            raise ValueError(f"num_classes 至少为 2，实际得到 {num_classes}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 计数使用 int64：整数统计精确无舍入，且实际样本量远不会溢出
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # ------------------------------------------------------------------
    # 状态维护：累积 / 重置 / 合并
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清零混淆矩阵，开始新一轮统计。"""
        self.matrix.zero_()

    def to(self, device: torch.device) -> "ConfusionMatrixAccumulator":
        """
        将混淆矩阵迁移到指定设备并返回自身（支持链式调用）。

        默认构造在 CPU；GPU 评估或 NCCL DDP 场景下可预先挪到训练设备，
        避免 update 时逐 batch 的 D2H 拷贝同步。
        """
        self.matrix = self.matrix.to(device)
        return self

    def cpu(self) -> "ConfusionMatrixAccumulator":
        """迁移到 CPU 的快捷方法，对齐 torchmetrics/nn.Module 接口。"""
        return self.to('cpu')

    def cuda(self, device: Optional[torch.device] = None) -> "ConfusionMatrixAccumulator":
        """迁移到 GPU 的快捷方法，默认当前 CUDA 设备。"""
        return self.to(device or torch.device('cuda', torch.cuda.current_device()))

    @property
    def device(self) -> torch.device:
        """矩阵当前所在设备。"""
        return self.matrix.device

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
            # 单通道 (N, 1, ...) 输出（如 sigmoid 二值分割）argmax 恒为 0，
            # 会静默统计错误，必须显式拦截
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

        # 统一搬运到矩阵所在设备（默认 CPU，可经 to() 迁移到训练设备）
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

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        update + compute 一步到位，对齐 torchmetrics.Metric 的调用约定。

        Returns:
            更新后的混淆矩阵 (num_classes, num_classes)
        """
        self.update(preds, target)
        return self.matrix

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(preds, target)

    def _check_compatible(self, other: "ConfusionMatrixAccumulator") -> None:
        if self.num_classes != other.num_classes:
            raise ValueError(
                f"num_classes 不一致，无法合并: {self.num_classes} vs {other.num_classes}"
            )
        # 不同 ignore_index 下被过滤的样本集合不同，统计口径不一致，禁止静默合并
        if self.ignore_index != other.ignore_index:
            raise ValueError(
                f"ignore_index 不一致，无法合并: {self.ignore_index} vs {other.ignore_index}"
            )

    def __add__(self, other: "ConfusionMatrixAccumulator") -> "ConfusionMatrixAccumulator":
        """合并两个混淆矩阵（如多进程各自统计后离线汇总）。"""
        if not isinstance(other, ConfusionMatrixAccumulator):
            return NotImplemented
        self._check_compatible(other)
        merged = ConfusionMatrixAccumulator(self.num_classes, ignore_index=self.ignore_index)
        merged.matrix = self.matrix + other.matrix
        return merged

    def __iadd__(self, other: "ConfusionMatrixAccumulator") -> "ConfusionMatrixAccumulator":
        """就地合并。"""
        if not isinstance(other, ConfusionMatrixAccumulator):
            return NotImplemented
        self._check_compatible(other)
        self.matrix += other.matrix
        return self

    def all_reduce(self) -> None:
        """
        DDP 多进程汇总：对各 rank 的矩阵求和并同步到所有进程。
        未初始化分布式环境时为 no-op，可无条件调用。
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        mat = self.matrix
        # 通信张量设备需匹配后端：NCCL 只支持 GPU 张量，其余后端（如 gloo）只支持 CPU
        if dist.get_backend() == dist.Backend.NCCL and not mat.is_cuda:
            mat = mat.cuda(torch.cuda.current_device())
        elif dist.get_backend() != dist.Backend.NCCL and mat.is_cuda:
            mat = mat.cpu()
        dist.all_reduce(mat, op=dist.ReduceOp.SUM)
        if mat is not self.matrix:
            self.matrix.copy_(mat.to(self.matrix.device))

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

    @classmethod
    def from_predictions(
        cls,
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: Optional[int] = None,
    ) -> "ConfusionMatrixAccumulator":
        """
        从预测批量一次性构造已填充的混淆矩阵（非增量语义）。

        等价于先建空矩阵再 update，但语义更直观：
        返回的实例已包含统计结果，适合离线/测试场景。

        Args:
            preds: 预测结果（logits 或类别索引）
            target: 真实标签
            num_classes: 类别总数
            ignore_index: 忽略的标签索引，None 表示不忽略

        Returns:
            已包含该 batch 统计结果的 ConfusionMatrixAccumulator
        """
        instance = cls(num_classes, ignore_index=ignore_index)
        instance.update(preds, target)
        return instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.total})"
        )



class ClassificationMetric:
    """
    分类任务指标计算层。

    持有 ConfusionMatrixAccumulator 作为唯一累积状态，所有指标从其计数视图推导；
    例外是 Top-k 准确率——它依赖 logits 排序信息，无法从混淆矩阵还原，
    因此单独维护命中/总数两个标量计数。

    compute() 返回汇总指标：
        - acc             总体准确率（单标签多分类下恒等于 micro P/R/F1）
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
        >>> print(f"acc={results['acc'].item():.4f}, top5={results['top5_acc'].item():.4f}")

    Args:
        num_classes: 类别总数（>= 2）
        average: 多类别聚合方式，"macro"（各类等权）/ "micro"（全局累加）/ "weighted"（按真实频率加权），默认 "macro"
        top_k: 额外统计 Top-k 准确率，None 表示不统计
        ignore_index: 忽略的标签索引，分类任务通常保持 None
    """

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted"] = "macro",
        top_k: Optional[int] = None,
        ignore_index: Optional[int] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None
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
        self.prefix = prefix or ''
        self.postfix = postfix or ''
        self.cm = ConfusionMatrixAccumulator(num_classes, ignore_index=ignore_index)
        # Top-k 计数器用 int64 张量在与矩阵同设备上累积，
        # 避免 int()/item() 造成逐 batch 的 GPU→CPU 同步
        self._topk_correct = torch.zeros((), dtype=torch.int64)
        self._topk_total = torch.zeros((), dtype=torch.int64)
        # top_k 启用但传入类别索引时只警告一次的标志
        self._topk_index_warned = False

    @property
    def confusion_matrix(self) -> torch.Tensor:
        """原始混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.matrix

    def reset(self) -> None:
        """重置全部累积状态（混淆矩阵 + Top-k 计数）。"""
        self.cm.reset()
        self._topk_correct.zero_()
        self._topk_total.zero_()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "ClassificationMetric":
        """
        深拷贝指标实例，用于从模板派生不同阶段的指标对象。

        原始实例作为模板保留全部状态（混淆矩阵、Top-k 计数等），
        clone 产生独立副本，可选覆盖 prefix / postfix 以适配不同日志阶段。

        Args:
            prefix: 覆盖前缀（如 'val/'、'test/'），None 表示沿用模板值
            postfix: 覆盖后缀，None 表示沿用模板值

        Returns:
            独立的新实例，累积状态与模板完全隔离
        """
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        return new_metric

    def to(self, device: torch.device) -> "ClassificationMetric":
        """将内部状态迁移到指定设备（默认构造在 CPU），返回自身。"""
        self.cm.to(device)
        self._topk_correct = self._topk_correct.to(device)
        self._topk_total = self._topk_total.to(device)
        return self

    def cpu(self) -> "ClassificationMetric":
        """迁移到 CPU 的快捷方法，对齐 torchmetrics/nn.Module 接口。"""
        return self.to('cpu')

    def cuda(self, device: Optional[torch.device] = None) -> "ClassificationMetric":
        """迁移到 GPU 的快捷方法，默认当前 CUDA 设备。"""
        return self.to(device or torch.device('cuda', torch.cuda.current_device()))

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        累积一个 batch。输入约定与 ConfusionMatrixAccumulator.update 一致；
        若启用了 top_k，仅当 preds 为 logits 形式时才能累积 Top-k 统计。
        """
        # 先交给混淆矩阵累积（其内部含 logits 通道数/形状/越界校验，
        # 且不会改动传入的 preds），再基于原始 logits 累积 Top-k
        self.cm.update(preds, target)
        if self.top_k is not None:
            if preds.dim() == target.dim() + 1:
                self._update_topk(preds, target)
            elif not self._topk_index_warned:
                # 索引形式的 preds 已丢失 logits 排序信息，无法累积 Top-k；只警告一次
                warnings.warn(
                    "top_k 已启用，但传入的 preds 是类别索引而非 logits，"
                    "Top-k 准确率将无法统计；请在 update 中传入 (N, C, ...) 形状的 logits"
                )
                self._topk_index_warned = True

    def _update_topk(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        # (N, C, ...) 沿类别维取 top-k 索引 → (N, k, ...)
        topk_idx = logits.topk(self.top_k, dim=1).indices
        hit = (topk_idx == target.unsqueeze(1).long()).any(dim=1)  # (N, ...)
        if self.cm.ignore_index is not None:
            valid = target != self.cm.ignore_index
            hit = hit[valid]
        # 张量累加 + 设备对齐，避免 int()/.item() 强制 GPU→CPU 同步
        self._topk_correct += hit.sum().to(self._topk_correct.device)
        self._topk_total += hit.numel()

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """update + compute 一步到位，对齐 torchmetrics.Metric 的调用约定。"""
        self.update(preds, target)
        return self.compute()

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(preds, target)

    def all_reduce(self) -> None:
        """DDP 汇总：混淆矩阵 + Top-k 计数一并同步，未初始化时 no-op。"""
        self.cm.all_reduce()
        if not (dist.is_available() and dist.is_initialized()):
            return
        if self.top_k is not None:
            t = torch.stack([self._topk_correct, self._topk_total])
            # 通信张量设备需匹配后端：NCCL 只支持 GPU 张量
            if dist.get_backend() == dist.Backend.NCCL and not t.is_cuda:
                t = t.cuda(torch.cuda.current_device())
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self._topk_correct.copy_(t[0].to(self._topk_correct.device))
            self._topk_total.copy_(t[1].to(self._topk_total.device))

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        从累积状态计算分类指标。

        汇总指标的聚合方式由构造时的 average 参数决定：
            - "macro"   ：逐类计算后等权平均
            - "micro"   ：全局 TP/FP/FN 累加后统一计算（多分类下等价于 acc）
            - "weighted"：逐类计算后按真实类别频率加权平均

        Returns:
            汇总指标 + 逐类指标（precision_i/recall_i/f1_i，
            键名格式与 trainer 日志分组正则 `^(.+)_(\\d+)$` 兼容）。
            值为 double 标量张量，与 torchmetrics 一致，取值用 .item()
        """
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
        total = self.cm.total

        # ---- 逐类指标 ----
        precision = tp / (pred_count + eps)
        recall = tp / (gt_count + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        acc = tp.sum() / (total + eps)

        # ---- 按 average 聚合 ----
        if self.average == "micro":
            # micro: 全局 TP/FP/FN 累加，多分类下 precision = recall = acc
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

        # Cohen's Kappa: pe 为边缘分布下的随机一致性率
        # 退化情形（空矩阵 / pe≈1）显式返回 0，避免 eps 兜底产生无意义数值
        pe = (gt_count * pred_count).sum() / ((total + eps) ** 2)
        if 1 - pe > eps:
            kappa = (acc - pe) / (1 - pe)
        else:
            # 保持 Dict[str, Tensor] 契约，退化分支同样返回 float64 张量
            kappa = torch.zeros((), dtype=torch.float64)

        return {
            f'{self.prefix}acc{self.postfix}': acc,
            f'{self.prefix}balanced_acc{self.postfix}': recall.mean(),  # 恒为 macro recall，不受 average 影响
            f'{self.prefix}precision{self.postfix}': agg_precision,
            f'{self.prefix}recall{self.postfix}': agg_recall,
            f'{self.prefix}f1{self.postfix}': agg_f1,
            f'{self.prefix}kappa{self.postfix}': kappa,
        }

    def per_class_metrics(self) -> Dict[str, torch.Tensor]:
        """
        逐类详细指标（按需获取，不包含在 compute() 输出中）。

        从混淆矩阵推导每类的 precision / recall / f1，
        键名格式 precision_{i} / recall_{i} / f1_{i}。

        Returns:
            逐类指标字典，值为 double 标量张量
        """
        eps = 1e-10
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
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
        """
        Top-k 准确率（按需获取，不包含在 compute() 输出中）。

        仅当 top_k 非空且已累积有效统计时返回结果，否则返回 None。

        Returns:
            double 标量张量，或 None
        """
        if self.top_k is None or self._topk_total == 0:
            return None
        return self._topk_correct.double() / self._topk_total

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的键名列表（无需调用 compute）。

        仅含汇总指标，逐类指标和 top_k 需通过
        per_class_metrics() / topk_acc() 单独获取。

        Returns:
            键名列表，如 ['val/acc', 'val/f1', 'val/kappa']
        """
        p, s = self.prefix, self.postfix
        return [f'{p}acc{s}', f'{p}balanced_acc{s}',
                f'{p}precision{s}', f'{p}recall{s}', f'{p}f1{s}', f'{p}kappa{s}']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"top_k={self.top_k}, total={self.cm.total})"
        )



class SegmentationMetric:
    """
    语义分割任务指标计算层。

    与 ClassificationMetric 同构：持有 ConfusionMatrixAccumulator 作为唯一累积状态，
    所有指标从其计数视图推导。每个像素视为一个独立样本，
    输入 preds/target 的 shape 通常为 (N, C, H, W) logits 或 (N, H, W) 类别索引。

    compute() 返回汇总指标：
        - oa              Overall Accuracy，总体像素精度（恒等于 micro 平均）
        - iou             平均交并比（聚合方式由 average 参数控制）
        - f1              平均 F1 / Dice（聚合方式由 average 参数控制）

    按需获取：
        - per_class_metrics()  逐类 iou/precision/recall/f1

    Example:
        >>> metric = SegmentationMetric(num_classes=21, ignore_index=255)
        >>> # preds: (N, 21, H, W) logits 或 (N, H, W) 类别索引
        >>> metric.update(preds, target)
        >>> results = metric.compute()
        >>> print(f"mIoU: {results['iou'].item():.4f}")

    Args:
        num_classes: 类别总数（含背景类，>= 2）
        average: 多类别聚合方式，"macro"（各类等权）/ "micro"（全局累加）/ "weighted"（按真实频率加权），默认 "macro"
        ignore_index: 忽略的标签索引，默认 255（VOC 等数据集的 void 区域约定）
    """

    def __init__(self, num_classes: int, 
                 average: Literal["macro", "micro", "weighted"] = "macro", 
                 ignore_index: Optional[int] = 255,
                 prefix: Optional[str] = None,
                 postfix: Optional[str] = None):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"average 需为 'macro'/'micro'/'weighted' 之一，实际得到 '{average}'")
        self.average = average
        self.prefix = prefix or ''
        self.postfix = postfix or ''
        self.cm = ConfusionMatrixAccumulator(num_classes, ignore_index=ignore_index)

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

    def reset(self) -> None:
        """重置混淆矩阵，开始新一轮评估。"""
        self.cm.reset()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "SegmentationMetric":
        """
        深拷贝指标实例，用于从模板派生不同阶段的指标对象。

        原始实例作为模板保留全部状态（混淆矩阵），
        clone 产生独立副本，可选覆盖 prefix / postfix 以适配不同日志阶段。

        Args:
            prefix: 覆盖前缀（如 'val/'、'test/'），None 表示沿用模板值
            postfix: 覆盖后缀，None 表示沿用模板值

        Returns:
            独立的新实例，累积状态与模板完全隔离
        """
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        if postfix is not None:
            new_metric.postfix = postfix
        return new_metric

    def to(self, device: torch.device) -> "SegmentationMetric":
        """将内部状态迁移到指定设备（默认构造在 CPU），返回自身。"""
        self.cm.to(device)
        return self

    def cpu(self) -> "SegmentationMetric":
        """迁移到 CPU 的快捷方法，对齐 torchmetrics/nn.Module 接口。"""
        return self.to('cpu')

    def cuda(self, device: Optional[torch.device] = None) -> "SegmentationMetric":
        """迁移到 GPU 的快捷方法，默认当前 CUDA 设备。"""
        return self.to(device or torch.device('cuda', torch.cuda.current_device()))

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """累积一个 batch，输入约定与 ConfusionMatrixAccumulator.update 一致。"""
        self.cm.update(preds, target)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """update + compute 一步到位，对齐 torchmetrics.Metric 的调用约定。"""
        self.update(preds, target)
        return self.compute()

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(preds, target)

    def all_reduce(self) -> None:
        """DDP 多进程汇总，未初始化时 no-op。"""
        self.cm.all_reduce()

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        从累积状态计算分割指标。

        汇总指标的聚合方式由构造时的 average 参数决定：
            - "macro"   ：逐类计算后等权平均
            - "micro"   ：全局 TP/FP/FN 累加后统一计算
            - "weighted"：逐类计算后按真实类别频率加权平均

        Returns:
            汇总指标（oa/iou/f1）+ 逐类指标
            （iou_i/precision_i/recall_i/f1_i，键名格式与 trainer
            日志分组正则 `^(.+)_(\\d+)$` 兼容）。
            值为 double 标量张量，与 torchmetrics 一致，取值用 .item()
        """
        eps = 1e-10  # 防除零：未出现的类别对应指标记为 0
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
        total = self.cm.total

        # ---- Overall Accuracy ----
        oa = tp.sum() / (total + eps)

        # ---- 逐类指标 ----
        recall = tp / (gt_count + eps)
        precision = tp / (pred_count + eps)
        iou = tp / (gt_count + pred_count - tp + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # ---- 按 average 聚合 ----
        if self.average == "micro":
            agg_iou = oa  # micro IoU 在多分类下等价于 overall accuracy
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
        """
        逐类详细指标（按需获取，不包含在 compute() 输出中）。

        从混淆矩阵推导每类的 iou / precision / recall / f1，
        键名格式 iou_{i} / precision_{i} / recall_{i} / f1_{i}。

        Returns:
            逐类指标字典，值为 double 标量张量
        """
        eps = 1e-10
        tp = self.cm.tp.double()
        gt_count = self.cm.gt_count.double()
        pred_count = self.cm.pred_count.double()
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

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的键名列表（无需调用 compute）。

        仅含汇总指标，逐类指标需通过 per_class_metrics() 单独获取。

        Returns:
            键名列表，如 ['val/oa', 'val/iou', 'val/f1']
        """
        p, s = self.prefix, self.postfix
        return [f'{p}oa{s}', f'{p}iou{s}', f'{p}f1{s}']

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, total={self.cm.total})"
        )


############ 自定义指标 ############


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
        >>> cm = ConfusionMatrixAccumulator(num_classes=5)
        >>> cm.update(preds, target)
        >>> results = separated_kappa(cm.matrix)
        >>> print(f"SeK: {results['sek']:.5f}")

    Args:
        matrix: 混淆矩阵 (num_classes, num_classes)，行=真实类别，列=预测类别，
            可直接传入 ConfusionMatrixAccumulator.matrix 或 SegmentationMetric.confusion_matrix
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


############ torchmetrics 适配层（实现模板，暂未接入训练流程） ############

if _HAS_TORCHMETRICS:

    class TorchMetricsClassificationWrapper(torchmetrics.Metric):
        """
        ClassificationMetric 的 torchmetrics.Metric 适配层（实现模板）。

        设计思路：
            torchmetrics 通过 add_state 托管状态张量（负责 reset / to / all_reduce），
            内层 ClassificationMetric 仅作为计算引擎，在 update / compute 前
            通过 _sync() 将托管状态同步到内层，确保两者指向同一张量。

        状态注册：
            - cm_matrix      : (num_classes, num_classes) int64 混淆矩阵，DDP 下 sum 归约
            - topk_correct   : int64 标量，Top-k 命中计数
            - topk_total     : int64 标量，Top-k 总样本计数

        Example:
            >>> template = ClassificationMetric(num_classes=10, prefix='val/')
            >>> wrapper = TorchMetricsClassificationWrapper(template)
            >>> wrapper.update(logits, target)
            >>> results = wrapper.compute()

        Note:
            现阶段仅作为 Metric 包装的实现模板，暂未接入训练流程。
        """

        def __init__(self, template: ClassificationMetric, **kwargs):
            super().__init__(**kwargs)
            self._inner = template
            n = template.num_classes
            self.add_state("cm_matrix",
                           default=torch.zeros(n, n, dtype=torch.int64),
                           dist_reduce_fx="sum")
            self.add_state("topk_correct",
                           default=torch.zeros((), dtype=torch.int64),
                           dist_reduce_fx="sum")
            self.add_state("topk_total",
                           default=torch.zeros((), dtype=torch.int64),
                           dist_reduce_fx="sum")

        def _sync(self):
            """将 torchmetrics 托管状态同步到内层 metric（重建张量引用）。"""
            self._inner.cm.matrix = self.cm_matrix
            self._inner._topk_correct = self.topk_correct
            self._inner._topk_total = self.topk_total

        def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
            self._sync()
            self._inner.update(preds, target)
            # 内层 update 原地修改 matrix（+=），self.cm_matrix 自动同步

        def compute(self) -> Dict[str, torch.Tensor]:
            self._sync()
            return self._inner.compute()

        def reset(self) -> None:
            super().reset()  # torchmetrics 原地 zero_() 或替换张量
            self._sync()     # 重建引用，确保内层指向托管张量

        def to(self, *args, **kwargs):
            super().to(*args, **kwargs)
            # super().to() 可能替换张量（引用断裂），_sync 重建连接
            self._sync()
            return self

        def per_class_metrics(self) -> Dict[str, torch.Tensor]:
            """透传内层 per_class_metrics()，调用前自动同步状态。"""
            self._sync()
            return self._inner.per_class_metrics()

        def topk_acc(self) -> Optional[torch.Tensor]:
            """透传内层 topk_acc()，调用前自动同步状态。"""
            self._sync()
            return self._inner.topk_acc()

        def metric_keys(self) -> list:
            return self._inner.metric_keys()

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}(num_classes={self._inner.num_classes}, "
                f"top_k={self._inner.top_k}, total={int(self.cm_matrix.sum())})"
            )

    class TorchMetricsSegmentationWrapper(torchmetrics.Metric):
        """
        SegmentationMetric 的 torchmetrics.Metric 适配层（实现模板）。

        与 TorchMetricsClassificationWrapper 同构，但无 Top-k 状态。

        状态注册：
            - cm_matrix : (num_classes, num_classes) int64 混淆矩阵，DDP 下 sum 归约

        Example:
            >>> template = SegmentationMetric(num_classes=21, ignore_index=255)
            >>> wrapper = TorchMetricsSegmentationWrapper(template)
            >>> wrapper.update(logits, target)
            >>> results = wrapper.compute()

        Note:
            现阶段仅作为 Metric 包装的实现模板，暂未接入训练流程。
        """

        def __init__(self, template: SegmentationMetric, **kwargs):
            super().__init__(**kwargs)
            self._inner = template
            n = template.num_classes
            self.add_state("cm_matrix",
                           default=torch.zeros(n, n, dtype=torch.int64),
                           dist_reduce_fx="sum")

        def _sync(self):
            """将 torchmetrics 托管状态同步到内层 metric（重建张量引用）。"""
            self._inner.cm.matrix = self.cm_matrix

        def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
            self._sync()
            self._inner.update(preds, target)

        def compute(self) -> Dict[str, torch.Tensor]:
            self._sync()
            return self._inner.compute()

        def reset(self) -> None:
            super().reset()
            self._sync()

        def to(self, *args, **kwargs):
            super().to(*args, **kwargs)
            self._sync()
            return self

        def per_class_metrics(self) -> Dict[str, torch.Tensor]:
            """透传内层 per_class_metrics()，调用前自动同步状态。"""
            self._sync()
            return self._inner.per_class_metrics()

        def metric_keys(self) -> list:
            return self._inner.metric_keys()

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}(num_classes={self._inner.num_classes}, "
                f"ignore_index={self._inner.ignore_index}, "
                f"total={int(self.cm_matrix.sum())})"
            )
