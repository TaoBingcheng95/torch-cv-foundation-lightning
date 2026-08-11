"""
构建优化器与调度器（配置驱动的工厂模式）

提供两个通用入口：
  - :func:`build_optimizer`  : 根据配置字典构建优化器，支持分层学习率
  - :func:`build_scheduler`  : 根据配置字典构建学习率调度器

支持的优化器类型::

    adam / adamw / sgd / rmsprop / adagrad

支持的调度器类型::

    steplr / multisteplr / exponentiallr / reducelronplateau /
    cosineannealinglr / onecyclelr / warmup_cosine / none

使用示例::

    optimizer = build_optimizer(model, {"type": "adamw", "lr": 1e-4, "weight_decay": 1e-4})
    scheduler = build_scheduler(optimizer, {"type": "cosineannealinglr", "T_max": 30})

    # 分层学习率 (backbone × 1.0, head × 3.0)
    optimizer = build_optimizer(model, {"type": "adamw", "lr": 1e-4, "head_lr_scale": 3.0})

    # Warmup + Cosine 组合调度
    scheduler = build_scheduler(optimizer, {"type": "warmup_cosine",
                                            "total_epochs": 30, "warmup_epochs": 5})
"""

import math
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR


# ======================================================================
#  工厂注册表
# ======================================================================

# 优化器工厂：类型 -> 类映射
OPTIMIZER_FACTORY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
}

# 调度器工厂
SCHEDULER_FACTORY = {
    "steplr": lr_scheduler.StepLR,
    "multisteplr": lr_scheduler.MultiStepLR,
    "exponentiallr": lr_scheduler.ExponentialLR,
    "reducelr": lr_scheduler.ReduceLROnPlateau,
    "plateau": lr_scheduler.ReduceLROnPlateau,
    "cosineannealinglr": lr_scheduler.CosineAnnealingLR,
    "onecyclelr": lr_scheduler.OneCycleLR,
    # "warmup_cosine" 为自定义组合调度，在 build_scheduler 中单独处理
}


# ======================================================================
#  优化器构建
# ======================================================================

def _build_param_groups(model: nn.Module, 
                        lr: float, 
                        weight_decay: float,
                        head_lr_scale: float = 1.0,
                        backbone_keywords: list = None,):
    """
    构建参数组，可选分层学习率。

    lr 作为全局基础学习率，各分组在此基础上乘以倍率派生:
    - backbone: lr × 1.0 (较小，微调预训练特征)
    - head:     lr × head_lr_scale (较大，快速学习新任务)
    - fallback: lr × 1.0 (无 backbone 区分或 head_lr_scale=1 时退化为单组)

    参数名中包含 ``backbone`` 或 ``features`` 的归为 backbone 组，
    其余归为 head 组。自动过滤 ``requires_grad=False`` 的参数。
    """
    if backbone_keywords is None:
        backbone_keywords = ["backbone", "features"]
    # 不启用分层学习率时，直接返回单组可训练参数
    if head_lr_scale == 1.0:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr, "weight_decay": weight_decay}]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in backbone_keywords):
            backbone_params.append(param)
        else:
            head_params.append(param)

    # 如果模型没有明确区分 backbone，退化为单组参数
    if not backbone_params:
        return [{"params": head_params, "lr": lr, "weight_decay": weight_decay}]

    return [
        {"params": backbone_params, "lr": lr,                 "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr * head_lr_scale, "weight_decay": weight_decay},
    ]


def build_optimizer(*,
                    model: nn.Module,
                    cfg: Optional[Dict[str, Any]] = None) -> optim.Optimizer:
    """
    根据配置字典构建优化器。

    Args:
        model: PyTorch 模型
        cfg: 优化器配置字典，缺省项使用默认值。常用字段::

            type          : 优化器类型 (默认 'adam')，见 OPTIMIZER_FACTORY
            lr            : 基础学习率 (默认 1e-4)
            weight_decay  : 权重衰减 (默认: adamw 为 1e-4，其余为 0)
            head_lr_scale : head 相对 lr 的倍率，>1 时启用分层学习率 (默认 1.0)
            betas / eps   : 仅对 Adam/AdamW 生效
            momentum / nesterov : 仅对 SGD 生效
            momentum / alpha    : 仅对 RMSprop 生效

    Returns:
        optim.Optimizer: 构建好的优化器实例

    Raises:
        ValueError: 优化器类型不在 OPTIMIZER_FACTORY 中
    """
    cfg = dict(cfg) if cfg else {}
    opt_type = cfg.get("type", "adam").lower()
    if opt_type not in OPTIMIZER_FACTORY:
        raise ValueError(
            f"Unsupported optimizer: '{opt_type}'. "
            f"Available: {list(OPTIMIZER_FACTORY.keys())}")

    lr = cfg.get("lr", 1e-4)
    # adamw 默认启用权重衰减，其余优化器默认关闭
    weight_decay = cfg.get("weight_decay", 1e-4 if opt_type == "adamw" else 0.0)
    head_lr_scale = cfg.get("head_lr_scale", 1.0)

    # 提取优化器专用参数（避免传入无关参数报错）
    opt_kwargs = {}
    if opt_type in ("adam", "adamw"):
        opt_kwargs["betas"] = tuple(cfg.get("betas", (0.9, 0.999)))
        opt_kwargs["eps"] = cfg.get("eps", 1e-8)
    elif opt_type == "sgd":
        opt_kwargs["momentum"] = cfg.get("momentum", 0.9)
        opt_kwargs["nesterov"] = cfg.get("nesterov", True)
    elif opt_type == "rmsprop":
        opt_kwargs["momentum"] = cfg.get("momentum", 0.0)
        opt_kwargs["alpha"] = cfg.get("alpha", 0.99)
        opt_kwargs["eps"] = cfg.get("eps", 1e-8)
    # adagrad 无额外专用参数

    param_groups = _build_param_groups(model, lr, weight_decay, head_lr_scale, 
                                       backbone_keywords=cfg.get("backbone_keywords"),)
    opt_class = OPTIMIZER_FACTORY[opt_type]
    return opt_class(param_groups, **opt_kwargs)


# ======================================================================
#  调度器构建
# ======================================================================

def _warmup_cosine_lambda(epoch, eta_min_ratio, warmup_epochs=5, total_epochs=30):
    """线性 Warmup + Cosine 衰减的统一 LR 乘子函数

    各边界情况处理：
      - warmup_epochs <= 0 : 跳过 warmup，直接 cosine 衰减
      - warmup_epochs >= total_epochs : 仅执行线性 warmup，到达目标 lr 后保持
      - 0 < warmup_epochs < total_epochs : 正常 warmup + cosine 衰减

    Args:
        epoch: 当前 epoch 索引 (0-based，由 LambdaLR 传入)
        eta_min_ratio: 最小学习率相对比例 (eta_min / base_lr)
        warmup_epochs: Warmup 轮数
        total_epochs: 总训练轮数
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # 线性 Warmup: start_factor=0.1 → end_factor=1.0
        return 0.1 + 0.9 * epoch / warmup_epochs

    # Cosine 衰减阶段
    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 0:
        return 1.0

    progress = (epoch - warmup_epochs) / decay_epochs
    return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_warmup_cosine(optimizer, total_epochs=30, warmup_epochs=5, eta_min=1e-6):
    """可被 Hydra instantiate 直接调用的 warmup+cosine 调度器"""
    lambdas = []
    for pg in optimizer.param_groups:
        base_lr = pg['lr']
        eta_min_ratio = min(eta_min / base_lr, 1.0) if base_lr > 0 else 0.0
        lambdas.append(partial(
            _warmup_cosine_lambda,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            eta_min_ratio=eta_min_ratio,
        ))
    return LambdaLR(optimizer, lr_lambda=lambdas)


def build_scheduler(optimizer: optim.Optimizer,
                    cfg: Optional[Dict[str, Any]] = None,
                    total_epochs: int = 30,
                    steps_per_epoch: Optional[int] = None) -> Optional[lr_scheduler._LRScheduler]:
    """
    根据配置字典构建学习率调度器。

    Args:
        optimizer: 已构建的优化器
        cfg: 调度器配置字典。``type`` 为 None/'none'/'' 时返回 None
            （即使用固定学习率）。各类型专用字段::

            steplr            : step_size (10), gamma (0.1)
            multisteplr       : milestones ([20, 25]), gamma (0.1)
            exponentiallr     : gamma (0.95)
            reducelr / plateau : mode ('min'), factor (0.5), patience (5)
            cosineannealinglr : T_max (total_epochs), eta_min (1e-6)
            onecyclelr        : max_lr (基础 lr × 10)，另需 steps_per_epoch
            warmup_cosine     : total_epochs, warmup_epochs (5), eta_min (1e-6)
        total_epochs: 总训练轮数，作为 T_max / total_epochs 的缺省值
        steps_per_epoch: 每轮迭代步数，仅 OneCycleLR 需要
            （通常传 ``len(train_loader)``）

    Returns:
        调度器实例；cfg 未指定有效类型时返回 None

    Raises:
        ValueError: 调度器类型不受支持，或 OneCycleLR 缺少 steps_per_epoch
    """
    cfg = dict(cfg) if cfg else {}
    sched_type = str(cfg.get("type", "") or "").lower()
    if sched_type in ("", "none", "null"):
        return None

    # ---- 自定义组合调度: warmup + cosine ----
    # 使用 LambdaLR 统一实现 warmup + cosine decay 组合调度，
    # 替代 SequentialLR 以避免 PyTorch 内部传递 epoch 参数触发的废弃警告。
    if sched_type == "warmup_cosine":
        return build_warmup_cosine(
            optimizer,
            total_epochs=cfg.get("total_epochs", total_epochs),
            warmup_epochs=cfg.get("warmup_epochs", 5),
            eta_min=cfg.get("eta_min", 1e-6),)

    if sched_type not in SCHEDULER_FACTORY:
        available = list(SCHEDULER_FACTORY.keys()) + ["warmup_cosine", "none"]
        raise ValueError(
            f"Unsupported scheduler: '{sched_type}'. Available: {available}")

    # 提取调度器专用参数
    if sched_type == "steplr":
        sched_kwargs = {"step_size": cfg.get("step_size", 10),
                        "gamma": cfg.get("gamma", 0.1)}
    elif sched_type == "multisteplr":
        sched_kwargs = {"milestones": cfg.get("milestones", [20, 25]),
                        "gamma": cfg.get("gamma", 0.1)}
    elif sched_type == "exponentiallr":
        sched_kwargs = {"gamma": cfg.get("gamma", 0.95)}
    elif sched_type == "reducelr" or sched_type == "plateau":
        import warnings
        warnings.warn(
            "ReduceLROnPlateau requires scheduler.step(metric) at each epoch. "
            "Ensure your training loop passes the monitored metric."
        )
        sched_kwargs = {"mode": cfg.get("mode", "min"),
                        "factor": cfg.get("factor", 0.5),
                        "patience": cfg.get("patience", 5)}
    elif sched_type == "cosineannealinglr":
        sched_kwargs = {"T_max": cfg.get("T_max", total_epochs),
                        "eta_min": cfg.get("eta_min", 1e-6)}
    elif sched_type == "onecyclelr":
        if steps_per_epoch is None:
            raise ValueError(
                "OneCycleLR 需要 steps_per_epoch 参数（通常为 len(train_loader)）")
        base_lr = max(pg["lr"] for pg in optimizer.param_groups)
        sched_kwargs = {"max_lr": cfg.get("max_lr", base_lr * 10),
                        "epochs": cfg.get("epochs", total_epochs),
                        "steps_per_epoch": steps_per_epoch}

    sched_class = SCHEDULER_FACTORY[sched_type]
    return sched_class(optimizer, **sched_kwargs)


# ======================================================================
#  梯度裁剪
# ======================================================================

def clip_grad_norm(model: nn.Module, max_norm: float = 1.0, norm_type: float = 2.0) -> float:
    """
    对模型的可学习参数进行梯度裁剪，防止梯度爆炸。
    
    Args:
        model: PyTorch 模型
        max_norm: 允许的最大梯度范数 (你的配置中为 1.0，非常合理)
        norm_type: 范数类型，2.0 表示 L2 范数 (最常用)
        
    Returns:
        total_norm: 裁剪前的总梯度范数 (可用于日志记录，监控梯度健康度)
    """
    # 显式过滤出需要梯度且当前 step 确实产生了梯度的参数
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
        
    # 调用 PyTorch 内置的 in-place 裁剪函数
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters, 
        max_norm=max_norm, 
        norm_type=norm_type
    )
    
    # 返回 total_norm (即使你当前不在 Trainer 中打印它，保留返回值也是好习惯)
    return float(total_norm)
