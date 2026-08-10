import os
import sys
# import numpy as np

import torch



def get_smart_num_workers():
    """获取推荐 num_workers，感知容器限制并适配平台。"""
    # 获取可用 CPU 核心数（容器友好）
    if hasattr(os, 'sched_getaffinity'):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1  # Windows/macOS 直接使用

    if sys.platform != 'linux':
        # spawn 平台：保守取值，最多 4 个 worker
        return max(1, min(available // 2, 4))
    else:
        # Linux (fork)：保留一个核心给主进程，无硬性上限（但可设 cap）
        # 如果你希望限制上限，可以用 min(available, 16) 等，这里建议不设限
        return max(1, available - 1)


def auto_pin_memory(device, num_workers=0):
    """智能判断是否使用 pin_memory，考虑平台和 num_workers。"""
    if not torch.cuda.is_available():
        return False

    dev = torch.device(device) if device is not None else torch.device('cuda')
    if dev.type != 'cuda':
        return False

    # Windows 上，若 num_workers > 0，pin_memory 易导致死锁或错误，强制禁用
    if sys.platform == 'win32' and num_workers > 0:
        return False

    # macOS 上，spawn 机制同样有风险，保守禁用（除非 num_workers=0）
    if sys.platform == 'darwin' and num_workers > 0:
        return False

    # Linux 且 num_workers >= 0 时均可启用（但 num_workers=0 时其实无加速效果，但仍可开）
    return True
