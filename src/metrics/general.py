"""
基于 torchmetrics 封装的常用评价指标集合。

定位：
    将 torchmetrics 内置指标组装成开箱即用的 MetricCollection，
    覆盖二分类/多分类/二值分割/多类分割四类场景，
    适合快速搭建常用指标、无需自定义计算逻辑时使用。
    其余两个模块的定位：
        metrics.py     基于原生 PyTorch 的实现，自定义指标开发/无 torchmetrics 时的选择；
        metrics_dev.py 开发中的、集成 torchmetrics 特性的指标类实现。

使用建议：
    cm 不作为 MetricCollection 成员（避免与标量指标混在一起 log_dict），
    而是作为独立属性 self.cm 管理，update/reset/clone/to 显式委托。
    推荐定义方式：

        self.val_metric = MulticlassClassificationMetric(
            num_classes=num_classes,
            prefix="val/",
        )

        def validation_step(self, batch, batch_idx):
            preds, target = batch
            self.log_dict(self.val_metric(preds, target))
        cm = self.val_metric.cm.compute()
"""

from typing import Literal, Optional
import copy

from torch import Tensor

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryCohenKappa,
    BinaryConfusionMatrix,
    BinaryJaccardIndex,
)



class BinaryClassificationMetric(MetricCollection):
    """
    二分类指标集合，继承 torchmetrics.MetricCollection。

    Args:
        threshold: 正类概率阈值，默认 0.5。
        ignore_index: 忽略的类别值，默认为None
        prefix: 指标前缀。
        postfix: 指标后缀。
    
    Example:
        metric = BinaryClassificationMetric(
            threshold=0.5,
            prefix="val/",
        )
    
        preds: Tensor, shape = (batch_size,), 通常是正类概率或 logits
        target: Tensor, shape = (batch_size,) 取值 0 或 1
    
        metric.update(preds, target)
        results = metric.compute()
        cm = metric.cm.compute()  # 混淆矩阵通过 self.cm 独立访问
    """
    def __init__(self,
                 threshold: float = 0.5,
                 ignore_index: Optional[int] = None,
                 prefix: Optional[str] = None,
                 postfix: Optional[str] = None):
            
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in range [0.0, 1.0]")
        metrics = {
            "acc": BinaryAccuracy(threshold=threshold,ignore_index=ignore_index),
            "f1": BinaryF1Score(threshold=threshold,ignore_index=ignore_index),
            "precision": BinaryPrecision(threshold=threshold,ignore_index=ignore_index),
            "recall": BinaryRecall(threshold=threshold,ignore_index=ignore_index),
            "kappa": BinaryCohenKappa(threshold=threshold,ignore_index=ignore_index),
        }
        super().__init__(metrics, prefix=prefix, postfix=postfix)
        # 混淆矩阵独立管理，不参与 compute() / log_dict
        # 使用 object.__setattr__ 绕过 nn.Module.__setattr__ 的自动子模块注册，
        # 否则 MetricCollection 会将 cm 纳入 compute() / keys() / clone()
        object.__setattr__(self, 'cm', BinaryConfusionMatrix(threshold=threshold, ignore_index=ignore_index))

    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        更新指标。

        Args:
            preds: 预测值，形状为 (batch_size,) 通常是正类概率或 logits。
            target: 标签，形状为 (batch_size,) 取值 0 或 1。
        """
        super().update(preds, target)
        self.cm.update(preds, target)

    def reset(self) -> None:
        super().reset()
        self.cm.reset()

    def clone(self, *args, **kwargs):
        """深拷贝指标集合，独立拷贝混淆矩阵（cm 非 MetricCollection 注册成员）。"""
        new_collection = super().clone(*args, **kwargs)
        object.__setattr__(new_collection, 'cm', copy.deepcopy(self.cm))
        return new_collection

    def to(self, *args, **kwargs):
        """移动指标到指定设备，同步移动混淆矩阵。"""
        super().to(*args, **kwargs)
        object.__setattr__(self, 'cm', self.cm.to(*args, **kwargs))
        return self

    @property
    def confusion_matrix(self) -> Tensor:
        """返回混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.compute()

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的标量指标键名列表（不含混淆矩阵）。

        键名已包含构造时指定的 prefix / postfix，无需调用 compute()，
        适合在初始化阶段预构建 CSV 表头、TensorBoard tag 集合等。

        Returns:
            键名列表，如 ['val/acc', 'val/f1', 'val/precision', ...]
        """
        return list(self.keys())



class MulticlassClassificationMetric(MetricCollection):
    """
    多分类指标集合，继承 torchmetrics.MetricCollection。

    Args:
        num_classes: 类别数，必须大于 1。
        average: 平均方式，支持 "macro"、"micro"，可选支持 "weighted"。
        prefix: 指标前缀。
        postfix: 指标后缀。

    Example:
        metric = MulticlassClassificationMetric(
            num_classes=5,
            average="macro",
            prefix="val/",
        )

        preds: Tensor, shape = (batch_size, num_classes) 或 (batch_size,)
        target: Tensor, shape = (batch_size,)

        metric.update(preds, target)
        results = metric.compute()
        cm = metric.cm.compute()  # 混淆矩阵通过 self.cm 独立访问
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        average: Literal["macro", "micro", "weighted"] = "macro",
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ):
        if num_classes < 2:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        if average not in {"macro", "micro", "weighted"}:
            raise ValueError("average must be one of 'macro', 'micro' or 'weighted'")

        metrics = {
            "acc": MulticlassAccuracy(num_classes=num_classes, average=average, ignore_index=ignore_index),
            "f1": MulticlassF1Score(num_classes=num_classes, average=average, ignore_index=ignore_index),
            "precision": MulticlassPrecision(num_classes=num_classes, average=average, ignore_index=ignore_index),
            "recall": MulticlassRecall(num_classes=num_classes, average=average, ignore_index=ignore_index),
            "kappa": MulticlassCohenKappa(num_classes=num_classes, ignore_index=ignore_index),
        }
        super().__init__(metrics, prefix=prefix, postfix=postfix)
        # 混淆矩阵独立管理，不参与 compute() / log_dict
        # 使用 object.__setattr__ 绕过 nn.Module.__setattr__ 的自动子模块注册，
        # 否则 MetricCollection 会将 cm 纳入 compute() / keys() / clone()
        object.__setattr__(self, 'cm', MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index))
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        更新指标。

        Args:
            preds: 预测值，形状为 (batch_size, num_classes) 或 (batch_size,)。
            target: 标签，形状为 (batch_size,)。
        """
        super().update(preds, target)
        self.cm.update(preds, target)

    def reset(self) -> None:
        super().reset()
        self.cm.reset()

    def clone(self, *args, **kwargs):
        """深拷贝指标集合，独立拷贝混淆矩阵（cm 非 MetricCollection 注册成员）。"""
        new_collection = super().clone(*args, **kwargs)
        object.__setattr__(new_collection, 'cm', copy.deepcopy(self.cm))
        return new_collection

    def to(self, *args, **kwargs):
        """移动指标到指定设备，同步移动混淆矩阵。"""
        super().to(*args, **kwargs)
        object.__setattr__(self, 'cm', self.cm.to(*args, **kwargs))
        return self

    @property
    def confusion_matrix(self) -> Tensor:
        """返回混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.compute()

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的标量指标键名列表（不含混淆矩阵）。

        键名已包含构造时指定的 prefix / postfix，无需调用 compute()，
        适合在初始化阶段预构建 CSV 表头、TensorBoard tag 集合等。

        Returns:
            键名列表，如 ['val/acc', 'val/f1', 'val/precision', ...]
        """
        return list(self.keys())



class BinarySegmentationMetric(MetricCollection):
    """
    二值分割指标集合。

    Example:
        metric = BinarySegmentationMetric(
            threshold=0.5,
            prefix="val/",
        )

        preds = torch.rand(4, 1, 256, 256)
        target = torch.randint(0, 2, (4, 256, 256))

        metric.update(preds, target)
        results = metric.compute()
        cm = metric.cm.compute()  # 混淆矩阵通过 self.cm 独立访问
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in range [0.0, 1.0]")
        self.task = "binary"

        metrics = {
                "pixel_acc": BinaryAccuracy(ignore_index=ignore_index, threshold=threshold,),
                "iou": BinaryJaccardIndex(ignore_index=ignore_index, threshold=threshold),
                "dice": BinaryF1Score(ignore_index=ignore_index, threshold=threshold),
                "precision": BinaryPrecision(ignore_index=ignore_index,threshold=threshold,),
                "recall": BinaryRecall(ignore_index=ignore_index, threshold=threshold,),
                "kappa": BinaryCohenKappa(ignore_index=ignore_index,threshold=threshold),
            }
        super().__init__(metrics, prefix=prefix, postfix=postfix)
        # 混淆矩阵独立管理，不参与 compute() / log_dict
        # 使用 object.__setattr__ 绕过 nn.Module.__setattr__ 的自动子模块注册
        object.__setattr__(self, 'cm', BinaryConfusionMatrix(ignore_index=ignore_index, threshold=threshold))


    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        将分割输入展平后交给底层分类指标更新状态。
        """
        preds, target = self._flatten_binary(preds, target)
        super().update(preds, target)
        self.cm.update(preds, target)

    def reset(self) -> None:
        super().reset()
        self.cm.reset()

    def clone(self, *args, **kwargs):
        """深拷贝指标集合，独立拷贝混淆矩阵（cm 非 MetricCollection 注册成员）。"""
        new_collection = super().clone(*args, **kwargs)
        object.__setattr__(new_collection, 'cm', copy.deepcopy(self.cm))
        return new_collection

    def to(self, *args, **kwargs):
        """移动指标到指定设备，同步移动混淆矩阵。"""
        super().to(*args, **kwargs)
        object.__setattr__(self, 'cm', self.cm.to(*args, **kwargs))
        return self

    @property
    def confusion_matrix(self) -> Tensor:
        """返回混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.compute()

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的标量指标键名列表（不含混淆矩阵）。

        键名已包含构造时指定的 prefix / postfix，无需调用 compute()，
        适合在初始化阶段预构建 CSV 表头、TensorBoard tag 集合等。

        Returns:
            键名列表，如 ['val/pixel_acc', 'val/iou', 'val/dice', ...]
        """
        return list(self.keys())



    @staticmethod
    def _flatten_binary(preds: Tensor, target: Tensor):
        """
        二值分割输入展平。

        支持:
            preds: (B, H, W), (B, 1, H, W), (B, 2, H, W)
            target: (B, H, W), (B, 1, H, W)
        """
        # target: (B, 1, H, W) -> (B, H, W)
        if target.ndim >= 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        # preds: (B, 1, H, W) -> (B, H, W)
        if preds.ndim == target.ndim + 1 and preds.ndim >= 3 and preds.shape[1] == 1:
            preds = preds.squeeze(1)

        # preds: (B, 2, H, W) -> 默认取 channel 1 作为前景
        elif preds.ndim == target.ndim + 1 and preds.ndim >= 3 and preds.shape[1] == 2:
            preds = preds[:, 1]

        if preds.ndim != target.ndim:
            raise ValueError(
                "Binary segmentation expects preds and target with the same spatial shape. "
                f"Got preds.shape={tuple(preds.shape)}, target.shape={tuple(target.shape)}. "
                "Expected preds: (B, H, W), (B, 1, H, W) or (B, 2, H, W); "
                "target: (B, H, W) or (B, 1, H, W)."
            )

        preds = preds.reshape(-1)
        target = target.reshape(-1)

        return preds, target



class MulticlassSegmentationMetric(MetricCollection):
    """
    多分类语义分割指标集合。

    Example:
        metric = MulticlassSegmentationMetric(
            num_classes=5,
            average="macro",
            prefix="val/",
        )

        preds = torch.randn(4, 5, 256, 256)
        target = torch.randint(0, 5, (4, 256, 256))

        metric.update(preds, target)
        results = metric.compute()
        cm = metric.cm.compute()  # 混淆矩阵通过 self.cm 独立访问
    """

    def __init__(
        self,
        num_classes: int,
        average: Literal["macro", "micro", "weighted", "none"] = "macro",
        ignore_index: Optional[int] = None,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ):
        if num_classes < 2:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        self.task = "multiclass"

        metrics = {
                "pixel_acc": MulticlassAccuracy(num_classes=num_classes, average=average, ignore_index=ignore_index),
                "iou": MulticlassJaccardIndex(num_classes=num_classes, average=average, ignore_index=ignore_index),
                "dice": MulticlassF1Score(num_classes=num_classes, average=average, ignore_index=ignore_index),
                "precision": MulticlassPrecision(num_classes=num_classes, average=average, ignore_index=ignore_index),
                "recall": MulticlassRecall(num_classes=num_classes, average=average, ignore_index=ignore_index),
                # CohenKappa 无 average 参数（全局一致性指标，不逐类平均）
                "kappa": MulticlassCohenKappa(num_classes=num_classes, ignore_index=ignore_index),
            }
        super().__init__(metrics, prefix=prefix, postfix=postfix)
        # 混淆矩阵独立管理，不参与 compute() / log_dict
        # 使用 object.__setattr__ 绕过 nn.Module.__setattr__ 的自动子模块注册
        object.__setattr__(self, 'cm', MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index))


    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        将分割输入展平后交给底层分类指标更新状态。
        """
        preds, target = self._flatten_multiclass(preds, target)
        super().update(preds, target)
        self.cm.update(preds, target)

    def reset(self) -> None:
        super().reset()
        self.cm.reset()

    def clone(self, *args, **kwargs):
        """深拷贝指标集合，独立拷贝混淆矩阵（cm 非 MetricCollection 注册成员）。"""
        new_collection = super().clone(*args, **kwargs)
        object.__setattr__(new_collection, 'cm', copy.deepcopy(self.cm))
        return new_collection

    def to(self, *args, **kwargs):
        """移动指标到指定设备，同步移动混淆矩阵。"""
        super().to(*args, **kwargs)
        object.__setattr__(self, 'cm', self.cm.to(*args, **kwargs))
        return self

    @property
    def confusion_matrix(self) -> Tensor:
        """返回混淆矩阵 (num_classes, num_classes)，供可视化等下游使用。"""
        return self.cm.compute()

    def metric_keys(self) -> list:
        """
        返回 compute() 输出中的标量指标键名列表（不含混淆矩阵）。

        键名已包含构造时指定的 prefix / postfix，无需调用 compute()，
        适合在初始化阶段预构建 CSV 表头、TensorBoard tag 集合等。

        Returns:
            键名列表，如 ['val/pixel_acc', 'val/iou', 'val/dice', ...]
        """
        return list(self.keys())


    @staticmethod
    def _flatten_multiclass(preds: Tensor, target: Tensor):
        """
        多分类语义分割输入展平。

        支持:
            preds: (B, C, H, W), logits/probs
            preds: (B, H, W), class indices
            target: (B, H, W), class indices
            target: (B, 1, H, W), class indices
        """
        # target: (B, 1, H, W) -> (B, H, W)
        if target.ndim >= 3 and target.shape[1] == 1:
            target = target.squeeze(1)

        # preds: (B, C, H, W), target: (B, H, W)
        if preds.ndim == target.ndim + 1:
            num_classes = preds.shape[1]

            # (B, C, H, W) -> (B, H, W, C) -> (B * H * W, C)
            preds = preds.movedim(1, -1).reshape(-1, num_classes)

            # (B, H, W) -> (B * H * W,)
            target = target.reshape(-1)

            return preds, target

        # preds: (B, H, W), target: (B, H, W)，都是类别索引
        if preds.ndim == target.ndim:
            preds = preds.reshape(-1)
            target = target.reshape(-1)

            return preds, target

        raise ValueError(
            "Multiclass segmentation expects preds as (B, C, H, W) or (B, H, W), "
            "and target as (B, H, W) or (B, 1, H, W). "
            f"Got preds.shape={tuple(preds.shape)}, target.shape={tuple(target.shape)}."
        )
