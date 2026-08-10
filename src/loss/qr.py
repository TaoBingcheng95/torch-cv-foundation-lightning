# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for learing on the prior."""

import torch
import torch.nn.functional as F
from torch.nn.modules import Module


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    先验平滑要求：
        target 应为严格大于 0 的逐像素类别先验分布（沿 dim=1 归一化）。
        若先验含 0（如 one-hot 硬标签），应先做均匀平滑：
            target = (1 - alpha) * target + alpha / num_classes
        内部的 clamp 仅是防 log(0) 的安全底线（clamp 后不重新归一化），
        不能替代平滑：未平滑的 one-hot 会使损失退化为带常数惩罚的硬交叉熵。

    .. versionadded:: 0.2
    """

    def __init__(self, from_logits: bool = True, eps: float = 1e-7):
        """
        Args:
            from_logits: 为 True 时 preds 视为 logits，内部沿 dim=1 做 softmax
                （与项目内 *WithLogitsLoss 的约定一致）；为 False 时视为
                已归一化的概率（torchgeo 原始行为）。
            eps: 数值下限，target 会被 clamp 到 [eps, 1 - eps] 防止 log(0)。
        """
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the QR (forwards) loss on prior.

        Args:
            preds: logits（from_logits=True）或预测概率（from_logits=False），
                expected shape B x C x H x W.
            target: prior probabilities, expected shape B x C x H x W.

        Returns:
            qr loss
        """
        q = preds.softmax(dim=1) if self.from_logits else preds
        target = target.clamp(self.eps, 1 - self.eps)

        q_bar = q.mean(dim=(0, 2, 3))
        # clamp 防止某类预测质量为 0（如 fp16 下溢）时 log 产生 nan
        qbar_log_S = (q_bar * torch.log(q_bar.clamp_min(self.eps))).sum()

        q_log_p = torch.einsum('bcxy,bcxy->bxy', q, torch.log(target)).mean()

        loss = qbar_log_S - q_log_p
        return loss


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    先验平滑要求：
        target 应为严格大于 0 的逐像素类别先验分布（沿 dim=1 归一化）。
        若先验含 0（如 one-hot 硬标签），应先做均匀平滑：
            target = (1 - alpha) * target + alpha / num_classes
        内部的 clamp 仅是防 log(0) 的安全底线（clamp 后不重新归一化），
        不能替代平滑。

    .. versionadded:: 0.2
    """

    def __init__(self, from_logits: bool = True, eps: float = 1e-7):
        """
        Args:
            from_logits: 为 True 时 preds 视为 logits，内部沿 dim=1 做 softmax
                （与项目内 *WithLogitsLoss 的约定一致）；为 False 时视为
                已归一化的概率（torchgeo 原始行为）。
            eps: 数值下限，target 会被 clamp 到 [eps, 1 - eps]，
                log 前的中间量也以此为下限防止 log(0)。
        """
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the RQ (backwards) loss on prior.

        Args:
            preds: logits（from_logits=True）或预测概率（from_logits=False），
                expected shape B x C x H x W
            target: prior probabilities, expected shape B x C x H x W

        Returns:
            rq loss
        """
        q = preds.softmax(dim=1) if self.from_logits else preds
        target = target.clamp(self.eps, 1 - self.eps)

        # manually normalize due to https://github.com/pytorch/pytorch/issues/70100
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(1e-12)
        r = F.normalize(z * target, p=1, dim=1)

        # r/q 取 log 前 clamp，防止零概率项产生 0 * (-inf) = nan
        loss = torch.einsum(
            'bcxy,bcxy->bxy',
            r,
            torch.log(r.clamp_min(self.eps)) - torch.log(q.clamp_min(self.eps)),
        ).mean()

        return loss
