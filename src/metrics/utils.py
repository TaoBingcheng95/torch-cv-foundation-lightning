from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist


# ======================================================================
# 共享工具：报告格式化
# ======================================================================


def fmt_value(
    val, pattern: str = ".4f", default: str = "N/A", scale: float = 1.0, suffix: str = "") -> str:
    """
    安全格式化数值（None / 非法值返回 default），供各指标类的 report() 复用。

    从 train/visualizer.py 迁入：报告排版属于指标口径的展示（哪些键、什么量纲、
    是否乘 100 变百分数都由指标定义决定），放在 metrics 下才能被指标类直接使用；
    train 依赖 metrics 是既有方向，visualizer 反过来从这里导入即可。

    Args:
        val: 数值
        pattern: 格式化模式
        default: 默认值（当 val 为 None 或格式化失败时）
        scale: 缩放因子
        suffix: 后缀字符串

    Returns:
        格式化后的字符串
    """
    if val is None:
        return default
    try:
        scaled = val * scale
        if isinstance(scaled, float) and math.isnan(scaled):
            return default
        if isinstance(scaled, torch.Tensor) and torch.isnan(scaled).any():
            return default
        return f"{scaled:{pattern}}{suffix}"
    except (TypeError, ValueError):
        return default


# ======================================================================
# 共享工具：分布式聚合
# ======================================================================


def all_reduce_scalars(values: Sequence[float]) -> list[float]:
    """
    把一组标量在所有 rank 间求和（SUM），供各指标类的 all_reduce() 复用

    指标类的累积状态都是可加量（计数、距离和），因此 SUM 后各 rank 持有完全一致的
    全局统计。所有标量打包进单个 float64 tensor，一次集合通信完成；float64 可精确
    表示 2^53 以内的整数，计数不会因浮点化而失真。

    Args:
        values: 待聚合的标量序列，各 rank 必须以相同长度、相同顺序传入

    Returns:
        求和后的 float 列表；单进程或未初始化进程组时原样返回（no-op），
        因此调用方可无条件调用。

    Note:
        含集合通信，所有 rank 必须同步调用且调用次数一致。
    """
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
        return [float(v) for v in values]
    # NCCL 要求 CUDA tensor，gloo 用 CPU tensor
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    packed = torch.tensor([float(v) for v in values], dtype=torch.float64, device=device)
    dist.all_reduce(packed)  # 默认 SUM
    return packed.tolist()
