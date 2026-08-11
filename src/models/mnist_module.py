from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import Metric, MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class MNISTLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    NUM_CLASSES = 10
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: Optional[torch.nn.Module]=None,
        metrics: Optional[Metric] = None,
        compile: bool= False,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["metrics", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        if metrics is not None:
            # 自定义指标与原生指标行为约定已对齐（compute 返回 dict、支持 clone/reset），
            # 用 MetricCollection 包装后与默认路径完全同构，共享 clone(prefix) 语义
            _common = MetricCollection(metrics.clone())
        else:
            # average="macro" 对类别不均衡更直观；micro 在多分类下恒等于 acc，无意义。
            metric_kwargs = dict(num_classes=self.NUM_CLASSES, average="macro")
            # 默认路径：MetricCollection 包装原生 torchmetrics 指标
            # 每个子指标 compute() 返回 tensor，log_dict 自动记录
            _common = MetricCollection({
                "acc":       MulticlassAccuracy(**metric_kwargs),
                "f1":        MulticlassF1Score(**metric_kwargs),
                "precision": MulticlassPrecision(**metric_kwargs),
                "recall":    MulticlassRecall(**metric_kwargs),
            })
        self.val_metrics = _common.clone(prefix="val/")
        self.test_metrics = _common.clone(prefix="test/")

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of logits (raw model output, 供 metrics 计算 top-k 等).
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metrics.update(logits, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # 自定义指标 compute() 返回 dict，Lightning 的 log_dict(MetricCollection) 只接受
        # 逐子指标 tensor 返回值，因此统一推迟到 epoch 末记录 compute() 的 dict 结果

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # MetricCollection.compute() 返回带 prefix 的 dict（val/acc、val/f1...）
        val_results = self.val_metrics.compute()
        self.log_dict(val_results, prog_bar=False)
        acc = val_results.get("val/acc", 0.0)
        # 记录的是纯数值，Lightning 不会自动 reset 集合，需手动重置避免跨 epoch 累加
        self.val_metrics.reset()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metrics.update(logits, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # 与 val 同理，推迟到 epoch 末记录 compute() 的 dict 结果

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        test_results = self.test_metrics.compute()
        self.log_dict(test_results, prog_bar=False)
        self.test_metrics.reset()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Sanitize ``hyper_parameters`` so the checkpoint is loadable with ``torch.load(weights_only=True)``.

        The in-memory ``self.hparams`` is left untouched (``configure_optimizers`` still needs
        ``optimizer``/``scheduler`` at runtime). Only the copy written to disk is cleaned:
        ``net`` is already stored in ``state_dict``, and ``optimizer``/``scheduler`` are
        ``functools.partial`` objects that get re-created from the Hydra config on every run.
        Dropping them avoids the non-serializable globals that force ``weights_only=False``.
        """
        super().on_save_checkpoint(checkpoint)
        hp = checkpoint.get("hyper_parameters")
        if hp is not None:
            for key in ("net", "optimizer", "scheduler", "criterion"):
                hp.pop(key, None)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)



if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
