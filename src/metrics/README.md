# Metrics 评价体系

面向分类 / 语义分割任务的 PyTorch 评价指标体系，兼顾**业务直接使用**和**自定义指标开发**两个目标。

## 模块总览

```
metrics/
├── __init__.py        # 顶层导出入口
├── general.py         # 基于 torchmetrics 封装的常用指标集合（MetricCollection），开箱即用
├── metrics.py         # 基于原生 PyTorch 的实现，自定义指标开发的学习参考 / 无 torchmetrics 后备
└── metrics_dev.py     # 开发中的、集成 torchmetrics 特性的指标类实现（trainer 当前依赖）
```

三个模块定位不同、互为补充：

| 模块 | 依赖 | 定位 | 典型用途 |
|------|------|------|----------|
| `general.py` | torchmetrics | 组装常用指标为 `MetricCollection`，开箱即用 | 快速搭建 val/test 指标，`self.log_dict(metric(preds, target))` |
| `metrics_dev.py` | torchmetrics | 继承 `torchmetrics.Metric` 的指标类实现，支持 DDP 自动同步 | trainer 默认指标；自定义指标开发参考 |
| `metrics.py` | 仅 torch | 原生 PyTorch 实现，手动管理状态和 DDP 同步 | 无 torchmetrics 环境；学习混淆矩阵指标推导范式 |

## 顶层导出

`from metrics import ...` 默认导入的是 `metrics_dev` 中的类（trainer 当前依赖其行为）：

```python
from metrics import (
    # metrics_dev（默认）—— torchmetrics Metric 子类
    ClassificationMetric,       # 分类指标
    SegmentationMetric,         # 分割指标
    ConfusionMatrixView,        # 无状态混淆矩阵计算视图
    SeparatedKappaMetric,       # 自定义指标开发参考（SeK）

    # metrics.py —— 原生 PyTorch 实现
    ConfusionMatrixAccumulator, # 混淆矩阵状态容器
    separated_kappa,            # SeK 函数（原生 PyTorch 版）

    # general.py —— MetricCollection 集合
    BinaryClassificationMetric,
    MulticlassClassificationMetric,
    BinarySegmentationMetric,
    MulticlassSegmentationMetric,
)
```

> **同名说明**：`metrics.py` 和 `metrics_dev.py` 均定义了 `ClassificationMetric` / `SegmentationMetric`，顶层导出的是 `metrics_dev` 版本。如需使用原生 PyTorch 版本，需通过子模块路径导入：`from metrics.metrics import ClassificationMetric`。

---

## 混淆矩阵约定

所有模块统一约定：

- 混淆矩阵 shape 为 `(num_classes, num_classes)`
- `cm[i, j]` 表示**真实类别为 i、被预测为 j** 的样本数量
- 行 = 真实类别，列 = 预测类别

---

## 一、general.py — 开箱即用的指标集合

继承 `torchmetrics.MetricCollection`，将 torchmetrics 内置指标组装成四类场景的集合，适合快速搭建常用指标。

### 类一览

| 类 | 场景 | 包含指标 |
|----|------|----------|
| `BinaryClassificationMetric` | 二分类 | acc, f1, precision, recall, kappa + cm |
| `MulticlassClassificationMetric` | 多分类 | acc, f1, precision, recall, kappa + cm |
| `BinarySegmentationMetric` | 二值分割 | pixel_acc, iou, dice, precision, recall, kappa + cm |
| `MulticlassSegmentationMetric` | 多类分割 | pixel_acc, iou, dice, precision, recall, kappa + cm |

### 设计要点

- **混淆矩阵独立管理**：`cm` 不作为 `MetricCollection` 成员（避免与标量指标混在一起 `log_dict`），而是作为独立属性 `self.cm` 管理，`update` / `reset` / `clone` / `to` 显式委托。
- **`metric_keys()`**：返回 `compute()` 输出中的标量指标键名列表（不含混淆矩阵），键名已包含构造时指定的 `prefix` / `postfix`，适合预构建 CSV 表头、TensorBoard tag 集合等。

### 使用示例

```python
from metrics import MulticlassClassificationMetric

# 构造（prefix 会自动加到所有指标键名前）
val_metric = MulticlassClassificationMetric(
    num_classes=10,
    average="macro",
    prefix="val/",
)

# 在 validation_step 中使用
def validation_step(self, batch, batch_idx):
    preds, target = batch
    self.log_dict(val_metric(preds, target))   # 标量指标直接 log
    cm = val_metric.cm.compute()               # 混淆矩阵独立访问

# 完整生命周期
val_metric.update(preds, target)
results = val_metric.compute()     # {'val/acc': ..., 'val/f1': ..., ...}
cm = val_metric.cm.compute()       # (num_classes, num_classes) int64 张量
val_metric.reset()
```

### 分割指标的输入约定

分割指标自动展平空间维，支持多种输入格式：

```python
# 二值分割
metric = BinarySegmentationMetric(threshold=0.5, prefix="val/")
preds = torch.rand(4, 1, 256, 256)     # 概率图
# 或 preds = torch.randint(0, 2, (4, 256, 256))  # 类别索引
target = torch.randint(0, 2, (4, 256, 256))
metric.update(preds, target)

# 多类分割
metric = MulticlassSegmentationMetric(num_classes=21, prefix="val/")
preds = torch.randn(4, 21, 256, 256)   # logits (B, C, H, W)
# 或 preds = torch.randint(0, 21, (4, 256, 256))  # 类别索引 (B, H, W)
target = torch.randint(0, 21, (4, 256, 256))
metric.update(preds, target)
```

---

## 二、metrics_dev.py — torchmetrics Metric 子类

继承 `torchmetrics.Metric`，利用 `add_state` 管理状态，DDP 下自动同步。这是 trainer 当前使用的默认指标。

### 架构分层

```
ConfusionMatrixView (无状态计算视图)
    ↓ 被包装
ClassificationMetric / SegmentationMetric (State + Compute)
    ↑ 继承
    torchmetrics.Metric（add_state 管理 matrix，DDP 自动同步）

SeparatedKappaMetric (自定义指标开发参考)
    ↑ 继承
    torchmetrics.MulticlassConfusionMatrix（复用父类状态和 update，仅覆写 compute）
```

- **`ConfusionMatrixView`**：纯计算容器，接收 matrix 张量，通过 property 暴露 `tp` / `fp` / `fn` / `tn` 等派生视图。`from_predictions` classmethod 负责从预测批量构造增量视图（含 argmax / ignore_index / 越界校验 / bincount 编码）。
- **`ClassificationMetric` / `SegmentationMetric`**：通过 `add_state` 管理 `matrix` 状态，`update` 阶段用 `ConfusionMatrixView.from_predictions` 取得增量并累加，`compute` 阶段用 `ConfusionMatrixView(self.matrix)` 包装后计算指标。
- **`_CMAdapter`**：轻量桥接对象，将 Metric 内部的 matrix 状态通过统一的 `.cm.compute()` 接口暴露给下游，与 `general.py` / `metrics.py` 的访问风格对齐。

### ClassificationMetric

```python
from metrics import ClassificationMetric

metric = ClassificationMetric(
    num_classes=10,
    average="macro",       # "macro" / "micro" / "weighted"
    top_k=5,               # 可选，额外统计 Top-k 准确率
    prefix="val/",
)

# 单步累积
metric.update(logits, target)

# 汇总指标
results = metric.compute()
# {
#     'val/acc': ...,
#     'val/balanced_acc': ...,    # 恒等于 macro Recall
#     'val/precision': ...,
#     'val/recall': ...,
#     'val/f1': ...,
#     'val/kappa': ...,           # Cohen's Kappa
# }

# 按需获取
per_class = metric.per_class_metrics()   # {'val/precision_0': ..., 'val/recall_0': ..., 'val/f1_0': ..., ...}
top5 = metric.topk_acc()                 # double 标量张量
cm = metric.cm.compute()                 # (10, 10) int64 混淆矩阵
keys = metric.metric_keys()              # ['val/acc', 'val/balanced_acc', ...]

# 深拷贝（从模板派生不同阶段的指标对象）
test_metric = metric.clone(prefix="test/")
```

### SegmentationMetric

```python
from metrics import SegmentationMetric

metric = SegmentationMetric(
    num_classes=21,
    average="macro",
    ignore_index=255,      # VOC 等数据集的 void 区域
    prefix="val/",
)

metric.update(logits, target)   # logits: (N, 21, H, W)
results = metric.compute()
# {'val/oa': ..., 'val/iou': ..., 'val/f1': ...}

per_class = metric.per_class_metrics()
# {'val/iou_0': ..., 'val/precision_0': ..., 'val/recall_0': ..., 'val/f1_0': ..., ...}
```

### SeparatedKappaMetric — 自定义指标开发参考

继承 `torchmetrics.MulticlassConfusionMatrix`，复用父类的 `confmat` 状态和 `update()` 逻辑，仅覆写 `compute()` 实现自定义推导。这是**"站在 torchmetrics 肩膀上开发自定义指标"的标准范式**。

```python
from metrics import SeparatedKappaMetric

metric = SeparatedKappaMetric(
    num_classes=5,
    bg_index=0,          # 背景类索引
    prefix="val/",
)
metric.update(preds, target)
results = metric.compute()
# {
#     'val/sek': ...,        # SeK = kappa_n0 * exp(IoU_fg - 1)
#     'val/kappa_n0': ...,   # 剔除背景 TN 后的 Kappa
#     'val/iou_fg': ...,     # 前景二值 IoU
#     'val/iou_bg': ...,     # 背景二值 IoU
#     'val/biou': ...,       # 二值 mIoU
# }
primary = metric.primary_value(results)   # 提取主指标，用于 early stopping / model selection
```

---

## 三、metrics.py — 原生 PyTorch 实现

仅依赖 `torch`，不依赖 `torchmetrics`。适用于自定义指标开发的学习参考，或无 `torchmetrics` 依赖环境下的后备选择。

### 分层设计

```
ConfusionMatrixAccumulator（状态层）
    只负责混淆矩阵的累积（update）、重置（reset）、合并（__add__ / all_reduce），
    以 property 暴露 tp/fp/fn/tn 向量，不推导任何指标。

ClassificationMetric / SegmentationMetric（计算层）
    持有 ConfusionMatrixAccumulator，从计数视图推导各自语义正确的指标。
```

### ConfusionMatrixAccumulator

```python
from metrics import ConfusionMatrixAccumulator

cm = ConfusionMatrixAccumulator(num_classes=10, ignore_index=255)

# 累积
cm.update(preds, target)       # logits 或类别索引均可

# 派生视图
cm.tp, cm.fp, cm.fn, cm.tn    # 逐类 one-vs-rest 计数向量
cm.gt_count                     # 每类真实样本数（行和）
cm.pred_count                   # 每类预测样本数（列和）
cm.total                        # 有效样本总数

# 合并（多进程各自统计后离线汇总）
merged = cm1 + cm2
cm1 += cm2

# DDP 汇总（未初始化时为 no-op）
cm.all_reduce()
```

### separated_kappa 函数

```python
from metrics import ConfusionMatrixAccumulator, separated_kappa

cm = ConfusionMatrixAccumulator(num_classes=5)
cm.update(preds, target)
results = separated_kappa(cm.matrix, bg_index=0)
# {'sek': ..., 'kappa_n0': ..., 'iou_fg': ..., 'iou_bg': ..., 'biou': ...}
```

### torchmetrics 适配层

`metrics.py` 底部还提供了 `TorchMetricsClassificationWrapper` 和 `TorchMetricsSegmentationWrapper`，演示如何将原生 PyTorch 指标包装为 `torchmetrics.Metric` 子类（通过 `add_state` 托管状态、`_sync()` 桥接内外层）。现阶段作为实现模板，暂未接入训练流程。

---

## 公共接口约定

所有指标类均遵循统一的接口风格：

| 方法 | 说明 |
|------|------|
| `update(preds, target)` | 累积一个 batch 的预测结果 |
| `compute()` | 从累积状态计算指标，返回 `Dict[str, Tensor]` |
| `reset()` | 重置全部累积状态 |
| `clone(prefix=, postfix=)` | 深拷贝实例，可选覆盖前缀/后缀 |
| `to(device)` / `cpu()` / `cuda()` | 设备迁移 |
| `metric_keys()` | 返回 `compute()` 输出的键名列表（无需调用 compute） |
| `per_class_metrics()` | 逐类详细指标（按需获取，不包含在 compute 输出中） |
| `__call__(preds, target)` | `update` + `compute` 一步到位 |

---

## 如何选择

| 场景 | 推荐模块 | 示例 |
|------|----------|------|
| 快速搭建 val/test 指标，配合 `self.log_dict()` | `general.py` | `MulticlassClassificationMetric(num_classes=10, prefix="val/")` |
| trainer 内使用，需要 DDP 自动同步 | `metrics_dev.py`（默认导出） | `ClassificationMetric(num_classes=10)` |
| 需要混淆矩阵做可视化等下游任务 | 任意模块的 `.cm.compute()` 或 `.confusion_matrix` | `cm = metric.cm.compute()` |
| 无 torchmetrics 环境 / 学习指标推导 | `metrics.py` | `from metrics.metrics import ClassificationMetric` |
| 开发自定义指标 | `metrics_dev.py` 的 `SeparatedKappaMetric` 作为参考 | 继承 torchmetrics 内置类，覆写 `compute()` |
| 离线评估 / 多进程汇总 | `metrics.py` 的 `ConfusionMatrixAccumulator` | `cm1 + cm2` 或 `cm.all_reduce()` |

---

## 自定义指标开发指南

### 方式一：继承 torchmetrics 内置类（推荐）

当自定义指标的状态容器与某个 torchmetrics 内置指标相同时（如都基于混淆矩阵），直接继承该内置类，只覆写 `compute()`：

```python
from torchmetrics.classification import MulticlassConfusionMatrix

class MyCustomMetric(MulticlassConfusionMatrix):
    def __init__(self, num_classes, prefix=None, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.prefix = prefix or ''

    def compute(self):
        matrix = self.confmat   # 父类管理的混淆矩阵，DDP 已自动同步
        # ... 自定义推导逻辑 ...
        return {f'{self.prefix}my_metric': result}

    def clone(self, prefix=None, postfix=None):
        new_metric = copy.deepcopy(self)
        if prefix is not None:
            new_metric.prefix = prefix
        new_metric._computed = None
        return new_metric
```

参考 `SeparatedKappaMetric` 的完整实现。

### 方式二：基于 ConfusionMatrixView 组合

当需要更灵活的计算视图组合时，利用 `ConfusionMatrixView` 的无状态 property 视图：

```python
from metrics.metrics_dev import ConfusionMatrixView

# 从 batch 获取增量矩阵
delta = ConfusionMatrixView.from_predictions(preds, target, num_classes=10)
print(delta.tp, delta.iou if hasattr(delta, 'iou') else ...)

# 包装累积后的矩阵做自定义计算
cm = ConfusionMatrixView(accumulated_matrix)
# 利用 cm.tp / cm.fp / cm.fn / cm.tn / cm.gt_count / cm.pred_count 推导
```

### 方式三：基于原生 PyTorch（无 torchmetrics 依赖）

参考 `metrics.py` 中 `ConfusionMatrixAccumulator` 的状态管理模式，手动管理矩阵累积和 DDP 同步。
