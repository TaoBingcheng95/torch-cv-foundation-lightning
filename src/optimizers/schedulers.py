"""自定义学习率调度器，兼容 Hydra instantiate + _partial_ 模式"""
import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler(LambdaLR):
    """Linear Warmup + Cosine Annealing 组合调度器。

    可直接作为 torch.optim.lr_scheduler 的替代项使用，
    支持 Hydra ``_partial_: true`` 延迟绑定 optimizer。

    Args:
        optimizer: 已构建的优化器
        total_epochs: 总训练轮数
        warmup_epochs: 线性 Warmup 轮数 (默认 5)
        eta_min: 最小学习率绝对值 (默认 1e-6)
        last_epoch: 起始 epoch 索引 (默认 -1)
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int = 30,
        warmup_epochs: int = 5,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        lambdas = []
        for pg in optimizer.param_groups:
            base_lr = pg["lr"]
            eta_min_ratio = min(eta_min / base_lr, 1.0) if base_lr > 0 else 0.0
            lambdas.append(
                partial(
                    self._warmup_cosine_lambda,
                    warmup_epochs=warmup_epochs,
                    total_epochs=total_epochs,
                    eta_min_ratio=eta_min_ratio,
                )
            )
        super().__init__(optimizer, lr_lambda=lambdas, last_epoch=last_epoch)

    @staticmethod
    def _warmup_cosine_lambda(epoch, eta_min_ratio, warmup_epochs, total_epochs):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        decay_epochs = total_epochs - warmup_epochs
        if decay_epochs <= 0:
            return 1.0
        progress = (epoch - warmup_epochs) / decay_epochs
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )