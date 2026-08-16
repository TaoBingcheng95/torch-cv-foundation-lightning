"""
评价指标包，包含三个定位不同的模块：

- metrics (metrics.py):
    基于原生 PyTorch 的通用分割/分类评价指标，不依赖 torchmetrics，
    自定义指标模块开发/无 torchmetrics 时的选择。
    与 metrics_dev 同名，需通过子模块路径导入：from metrics.metrics import ...
- general (general.py):
    基于 torchmetrics 封装的常用指标集合（MetricCollection），开箱即用。
- metrics_dev (metrics_dev.py):
    开发中的、集成 torchmetrics 特性的指标类实现。

顶层导出约定：
    ClassificationMetric / SegmentationMetric / ConfusionMatrixView / SeparatedKappaMetric
    默认来自 metrics_dev（trainer 当前依赖其行为）；
    ConfusionMatrixAccumulator 来自 metrics.py（原生 PyTorch 状态容器）；
    general.py 的集合类在顶层直接导出。
"""

# 默认指标类：开发中的 torchmetrics 特性版本（trainer 当前依赖）
from .metrics_dev import (
    ClassificationMetric,
    ConfusionMatrixView,
    SegmentationMetric,
    SeparatedKappaMetric,
)

# 原生 PyTorch 实现：状态容器 + 自定义指标函数
from .metrics import ConfusionMatrixAccumulator, separated_kappa

# 基于 torchmetrics 封装的常用指标集合
from .general import (
    BinaryClassificationMetric,
    MulticlassClassificationMetric,
    BinarySegmentationMetric,
    MulticlassSegmentationMetric,
)

__all__ = [
    # metrics_dev（默认）
    'ClassificationMetric',
    'ConfusionMatrixView',
    'SegmentationMetric',
    'SeparatedKappaMetric',
    # metrics.py（原生 PyTorch）
    'ConfusionMatrixAccumulator',
    'separated_kappa',
    # general
    'BinaryClassificationMetric',
    'MulticlassClassificationMetric',
    'BinarySegmentationMetric',
    'MulticlassSegmentationMetric',
]
