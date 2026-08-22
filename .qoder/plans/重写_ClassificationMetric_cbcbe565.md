
# 重写 ClassificationMetric（仿 template.py 设计）

## 背景

`metrics_pytorch_v2.py` 中的 `ClassificationMetric`（第 565-925 行）存在大量从 `DetectionMetric` 复制粘贴后未清理的致命错误，无法正常运行。需要仿照 `template.py` 中 `DetectionMetric` 的设计模式，基于同文件中已有的 `ConfusionMatrixCounts`（多分类混淆矩阵状态容器）重新实现。

## 设计对照

| 维度 | template.py `DetectionMetric` | 新 `ClassificationMetric` |
|------|------|------|
| 基类 | 纯 Python 类 | 纯 Python 类 |
| 状态容器 | `ConfusionCounts`（5 标量） | `ConfusionMatrixCounts`（C×C 矩阵） |
| 类常量 | `CLASS_NAMES`, `CURVE_KEYS` | `CURVE_KEYS`（无固定 CLASS_NAMES，由 num_classes 决定） |
| update 模式 | `_classify_batch → +=` | `_classify_batch → +=`（但返回 `ConfusionMatrixCounts`） |
| compute 输出 | 4 个比例指标 | acc / balanced_acc / precision / recall / f1 / kappa |
| 额外统计 | — | Top-k 准确率（需 logits，单独 buffer 累积） |

## 修改内容

### 1. 修复文件头 imports（第 1-42 行区域）

- 第 1 行前插入 `from __future__ import annotations`（启用 PEP 604 `X | Y` 语法）
- 第 31 行 `from typing import Optional, Dict, Literal` → `from typing import Any, Literal`
  - 移除未使用的 `Optional`、`Dict`
  - 添加 `Any`（`__init__` 的 `**kwargs` 需要）

### 2. 重写 `ClassificationMetric` 类（替换第 565-925 行）

新类结构如下：

```
ClassificationMetric
├── 类常量
│   └── CURVE_KEYS = ("precision", "recall", "f1")
├── __init__(num_classes, average, top_k, ignore_index, prefix, postfix, monitor, **kwargs)
│   └── self._counts = ConfusionMatrixCounts(num_classes, ignore_index=ignore_index)
│   └── self._topk_correct / self._topk_total（int64 0-dim tensor，仅 top_k 启用时累积）
├── reset()
├── update(preds, target)
│   ├── 预处理：logits → argmax（含通道数/形状/越界校验）
│   ├── counts = self._classify_batch(preds, target)
│   ├── self._counts += counts
│   └── if top_k: self._update_topk(logits, target)
├── _preprocess_preds(preds, target) → (pred_idx, target_flat)
│   ├── logits (N,C,...) → argmax(dim=1)；通道 < 2 报错
│   ├── 形状校验
│   ├── ignore_index 过滤
│   └── 越界标签校验
├── _classify_batch(preds, target) → ConfusionMatrixCounts
│   └── 创建临时 ConfusionMatrixCounts，bincount 填充矩阵
├── _update_topk(logits, target)
│   └── topk(dim=1) → 命中统计累加到 buffer
├── compute() → dict[str, Tensor]
│   ├── per-class: precision / recall / f1（从 cm.tp / pred_count / gt_count 推导）
│   ├── acc = diag_sum / total
│   ├── balanced_acc = macro recall
│   ├── kappa（Cohen's Kappa，含退化处理）
│   └── 按 average 聚合 precision / recall / f1
├── _aggregate(per_class, freq) → Tensor
│   └── micro / macro / weighted 三分支
├── all_reduce()
│   └── cm.all_reduce() + topk buffer all_reduce
├── 设备管理：to / cuda / cpu（返回 self）
├── 计数 property：confusion_matrix / tp / fp / fn / tn / total / counts
├── 键名/主指标：metric_keys / core_metric_keys / primary_value
├── clone(prefix, postfix) → 深拷贝
├── per_class_metrics() → dict[str, Tensor]（按需获取，不含在 compute 输出中）
├── topk_acc() → Tensor | None（按需获取）
├── report(results, elapsed_time, speed, log) → str
│   └── 分类报告：聚合指标 → 逐类 P/R/F1 表 → 原始计数 → 混淆矩阵摘要
└── __repr__()
```

#### 关键设计决策

1. **`_classify_batch` 返回 `ConfusionMatrixCounts`**：创建临时实例，用 `bincount` 填充矩阵，通过 `__iadd__` 合并到 `self._counts`。与 template 的 `_classify_batch → ConfusionCounts → +=` 模式完全对齐。

2. **`confusion_matrix()` 为方法**（非 property）：与 template 的 `DetectionMetric.confusion_matrix()` 一致，trainer 中调用方式统一。

3. **`compute()` 只返回 6 个聚合指标**：原始计数通过 `self._counts.summary()` 获取，逐类指标通过 `per_class_metrics()` 获取，保持 compute 输出精简。

4. **`average` 参数完整实现**：
   - `macro`：逐类等权平均
   - `micro`：全局 TP/FP/FN 累加后计算（多分类下 P=R=acc）
   - `weighted`：按真实类别频率加权

5. **Top-k 准确率**：单独维护 `_topk_correct` / `_topk_total` 两个 int64 buffer，需 logits 输入，通过 `topk_acc()` 按需获取（不含在 compute 输出中）。

6. **Cohen's Kappa**：从混淆矩阵边缘分布计算随机一致性率 pe，含退化情形处理（pe≈1 时返回 0）。

7. **report 文案为分类场景**：不再出现 "DETECTION" / "Frame Counts" 等检测术语，改为 "CLASSIFICATION TEST REPORT" / "Sample Counts" / 逐类指标表。

### 3. 不变部分

- `ConfusionCounts`（第 47-283 行）：二值检测计数器，template 依赖，不修改
- `ConfusionMatrixCounts`（第 286-561 行）：多分类混淆矩阵状态容器，不修改
- `utils.py`：不修改

## 验证

- 运行 `python -c "from src.metrics.metrics_pytorch_v2 import ClassificationMetric"` 确认无导入错误
- 运行现有测试 `pytest tests/ -x` 确认无回归
