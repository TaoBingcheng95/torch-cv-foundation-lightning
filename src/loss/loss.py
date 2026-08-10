
from typing import List, Optional
import torch
import torch.nn as nn
import torch.linalg as LA
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss



class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits.
    """

    def __init__(
        self,
        reduction: str ="mean",
        weight: Optional[float]=None,
        pos_weight: Optional[float]=None):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be one of ('none', 'mean', 'sum'), got {reduction!r}"
            )
        self.reduction = reduction
        if weight is not None:
            self.register_buffer(
                'weight_tensor',
                torch.tensor(weight, dtype=torch.float32)
            )
        else:
            self.weight_tensor = None
        # pos_weight 也注册为 buffer，避免每次 forward 重复创建张量
        if pos_weight is not None:
            self.register_buffer(
                'pos_weight_tensor',
                torch.tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight_tensor = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=self.weight_tensor,
            reduction="none",
            pos_weight=self.pos_weight_tensor,
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
                        


class CEWithLogitsLoss(nn.Module):
    """
    Cross Entropy Loss (接受原始 logits，内部执行 log_softmax + NLL).
    适用于多分类互斥任务（如语义分割、图像分类）。
    """
    def __init__(self, weight: Optional[List[float]]=None,
                 reduction: str = "mean",
                 ignore_index: int = -100, # 忽略的标签值（VOC 边界为 255）
                 label_smoothing: float = 0.0):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )


    def forward(self, output, target):
        loss = self.cross_entropy(output, target)
        return loss



class CEDiceLoss(nn.Module):
    """
    CE + Dice 组合损失，语义分割常用搭配：
    - CE：逐像素分类，梯度稳定、收敛快；
    - Dice：直接优化区域重叠度（与 mIoU 更对齐），缓解前景/背景类别不均衡。
    total = ce_weight * CE + dice_weight * Dice
    """
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weight: Optional[List[float]] = None,
        ignore_index: int = 255,
        smooth: float = 0.0,
    ):
        """
        Args:
            ce_weight: CE 项的权重
            dice_weight: Dice 项的权重
            class_weight: 各类别权重（仅作用于 CE 项）
            ignore_index: 忽略的标签值（VOC 边界为 255）
            smooth: Dice 的平滑系数
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = CEWithLogitsLoss(weight=class_weight,
                                   ignore_index=ignore_index)
        self.dice = DiceLoss(mode="multiclass",
                             from_logits=True,
                             ignore_index=ignore_index,
                             smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - logits: (N, C, H, W) 原始输出（未过 softmax）
            - target: (N, H, W) 类别索引标签
        """
        return (self.ce_weight * self.ce(logits, target)
                + self.dice_weight * self.dice(logits, target))



class DiceLoss(_Loss):
    BINARY_MODE: str = "binary"
    MULTICLASS_MODE: str = "multiclass"
    MULTILABEL_MODE: str = "multilabel"
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {self.BINARY_MODE, self.MULTILABEL_MODE, self.MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != self.BINARY_MODE, (
                "Masking classes is not supported with mode=binary"
            )
            classes = torch.tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        # Tversky 系数：alpha 惩罚 FP，beta 惩罚 FN；alpha=beta=0.5 即 Dice
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == self.MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == self.BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == self.MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == self.MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return self.soft_tversky_score(
            output, target, self.alpha, self.beta, smooth, eps, dims
        )

    @staticmethod
    def soft_tversky_score(
        output: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None,) -> torch.Tensor:
        """Tversky loss

        References:
            https://arxiv.org/pdf/2302.05666
            https://arxiv.org/pdf/2303.16296

        """
        assert output.size() == target.size()

        if dims is not None:
            output_sum = torch.sum(output, dim=dims)
            target_sum = torch.sum(target, dim=dims)
            difference = LA.vector_norm(output - target, ord=1, dim=dims)
        else:
            output_sum = torch.sum(output)
            target_sum = torch.sum(target)
            difference = LA.vector_norm(output - target, ord=1)

        intersection = (output_sum + target_sum - difference) / 2  # TP
        fp = output_sum - intersection
        fn = target_sum - intersection

        tversky_score = (intersection + smooth) / (
            intersection + alpha * fp + beta * fn + smooth
        ).clamp_min(eps)
        return tversky_score

    @staticmethod
    def soft_dice_score(
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None) -> torch.Tensor:
        assert output.size() == target.size()
        # 静态方法内无 self，通过类名调用同类静态方法；
        # Dice 即 alpha=beta=0.5 的 Tversky 特例
        dice_score = DiceLoss.soft_tversky_score(output, target, 0.5, 0.5, smooth, eps, dims)
        return dice_score



class TverskyLoss(DiceLoss):
    """Tversky loss = 1 - Tversky score，Dice 的广义形式。

    通过 alpha/beta 解耦 FP 与 FN 的惩罚：
    - alpha=beta=0.5 时退化为 DiceLoss；
    - beta > alpha 时加大对漏检 (FN) 的惩罚，适合小目标 / 类别不均衡的分割任务。

    其余参数 (mode / from_logits / ignore_index / smooth / classes / log_loss / eps)
    语义与 DiceLoss 完全一致，直接继承其 forward 与 mask 处理逻辑，无代码重复。
    """

    def __init__(
        self,
        mode: str,
        alpha: float = 0.3,
        beta: float = 0.7,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
            alpha=alpha,
            beta=beta,
        )



class IoULoss(DiceLoss):
    """IoU (Jaccard) loss = 1 - IoU，分割里与 mIoU 评测指标最对齐的区域重叠损失。

    数学上 IoU 是 Tversky 在 alpha=beta=1 时的特例：
        IoU = TP / (TP + FP + FN) = TP / union
    因此这里直接继承 DiceLoss（即 Tversky 引擎），将 alpha/beta 固定为 1.0，
    复用其 mode / from_logits / ignore_index / smooth / classes / log_loss / eps
    全套接口与 mask 处理逻辑，支持 binary / multiclass / multilabel 三种模式。

    Shape:
        - y_pred: (N, C, H, W) 原始 logits（from_logits=True 时内部过 softmax/sigmoid）
        - y_true: (N, H, W) 类别索引（multiclass）或 (N, C, H, W)（multilabel/binary）
    """

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        # IoU 定义为 alpha=beta=1，不暴露为参数以避免被改成“非 IoU”
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_index=ignore_index,
            eps=eps,
            alpha=1.0,
            beta=1.0,
        )



class FocalLoss(nn.Module):
    """Focal Loss，支持二分类与多分类。

    通过 (1 - p_t)^gamma 降低易样本权重、聚焦难样本，缓解类别不均衡。
      二分类:   p_t = sigmoid(logit) 对应真实标签的概率
      多分类:   p_t = softmax(logit) 在真实类别上的概率
    focal = alpha * (1 - p_t)^gamma * ce_per_element

    注: ignore_index 仅在 multiclass 模式下生效；alpha 为全局标量调制，
    若需 per-class 权重请配合上游重采样或使用 CEWithLogitsLoss(weight=...)。
    """
    BINARY_MODE: str = "binary"
    MULTICLASS_MODE: str = "multiclass"

    def __init__(
        self,
        mode: str = "binary",
        alpha: float = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        if mode not in (self.BINARY_MODE, self.MULTICLASS_MODE):
            raise ValueError(
                f"mode must be 'binary' or 'multiclass', got {mode!r}"
            )
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}"
            )
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == self.BINARY_MODE:
            # 拍平为 1D，兼容 (N,1) 与 (N,) 标签形状
            inputs = inputs.reshape(-1)
            t = targets.reshape(-1)
            # 逐元素 BCE（with_logits 内部用 log-sum-exp，数值稳定）
            bce = F.binary_cross_entropy_with_logits(inputs, t.float(), reduction="none")
            p_t = torch.exp(-bce)  # target=1 时 p_t=p，target=0 时 p_t=1-p
            focal = self.alpha * (1 - p_t).pow(self.gamma) * bce
            valid = None
        else:  # MULTICLASS_MODE: inputs (N, C), targets (N,)
            t = targets.long()
            # ignore_index 位置先用 0 替换避免 gather 越界，之后用 mask 清零
            if self.ignore_index is not None:
                valid = t != self.ignore_index
                t_safe = torch.where(valid, t, torch.zeros_like(t))
            else:
                valid = None
                t_safe = t
            log_probs = F.log_softmax(inputs, dim=1)
            log_pt = log_probs.gather(1, t_safe.unsqueeze(1)).squeeze(1)  # (N,)
            p_t = log_pt.exp()
            focal = self.alpha * (1 - p_t).pow(self.gamma) * (-log_pt)
            if valid is not None:
                focal = focal * valid.to(focal.dtype)

        if self.reduction == "none":
            return focal
        if self.reduction == "sum":
            return focal.sum()
        # mean: multiclass 下仅对有效 (非 ignored) 位置取平均
        if valid is not None:
            return focal.sum() / valid.to(focal.dtype).sum().clamp_min(1.0)
        return focal.mean()
