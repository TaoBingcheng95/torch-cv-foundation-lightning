"""
"""

# 默认指标类：开发中的 torchmetrics 特性版本
from .metrics_dev import (
    ClassificationMetric,
    SegmentationMetric,
)

# 原生 PyTorch 实现：状态容器 + 自定义指标函数
from .metrics_pytorch import TorchClassificationMetric, TorchSegmentationMetric

# 基于 torchmetrics 封装的常用指标集合
from .general import (
    BinaryClassificationMetric,
    MulticlassClassificationMetric,
    BinarySegmentationMetric,
    MulticlassSegmentationMetric,
)

__all__ = [
    # metrics_pytorch.py（原生 PyTorch）
    "TorchClassificationMetric",
    "TorchSegmentationMetric",
    # metrics_dev.py
    "SegmentationMetric",
    "ClassificationMetric",
    # general
    'BinaryClassificationMetric',
    'MulticlassClassificationMetric',
    'BinarySegmentationMetric',
    'MulticlassSegmentationMetric',
]
