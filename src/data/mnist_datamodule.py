from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, random_split # ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from lightning import LightningDataModule


from src.data.utils import get_smart_num_workers, auto_pin_memory



class MNISTDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    # MNIST 的均值和标准差
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    DEFAULT_SIZE = (28, 28)
    RESIZE_SIZE = (32, 32)
    DEFAULT_TRAIN_LENGTH = 60000
    DEFAULT_TEST_LENGTH = 10000
    def __init__(
        self,
        data_dir: str = "data/",
        # 验证集划分：float ∈ (0,1) 按比例从训练集切分，int ≥ 1 按绝对数量切分；
        # 官方 test 集保持不动，保证指标与外部基准可比
        train_val_split: Union[int, float] = 0.1, # 5_000
        batch_size: int = 64,
        num_workers: int = -1,
        pin_memory: bool = False,
        use_normalize: bool = True,
        resize:bool = True,
        seed:int = 42
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_split: The validation and test split. Defaults to `0.1`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)            

        #【重要】固定随机种子，保证每次运行划分一致
        self.generator=torch.Generator()
        self.generator.manual_seed(self.hparams.seed)

        # data transformations
        if resize:
            # LeNet-5 经典输入是 32x32，MNIST 原是 28x28
            transforms_list = [
                transforms.Resize(self.RESIZE_SIZE)]
        else:
            transforms_list = []
        transforms_list.append(transforms.ToTensor())
        if use_normalize:
            transforms_list.append(transforms.Normalize(mean=self.MNIST_MEAN, 
                                                       std=self.MNIST_STD))
        self.transforms = transforms.Compose(transforms_list)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    @property
    def classes(self) -> list[str]:
        return self.data_test.classes  # type: ignore
    
    @property
    def class_to_idx(self) -> dict[str, int]:
        return self.data_test.class_to_idx

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {value: key for key, value in self.class_to_idx.items()}

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.train_val_test_split, 
            #     generator=self.generator)

            total_count = len(trainset)
            # float 按比例、int 按绝对数量
            val_count = int(total_count * self.hparams.train_val_split) if isinstance(self.hparams.train_val_split, float) else self.hparams.train_val_split
            train_count = total_count - val_count

            self.data_test = testset
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=[train_count, val_count],
                generator=self.generator,
            )

        # Resolve effective num_workers / pin_memory (uses smart helpers when
        # sentinels are set). Done here so the trainer (and its device) is
        # already attached.
        self._resolve_loader_settings()

    def _resolve_loader_settings(self) -> None:
        """Resolve effective ``num_workers`` / ``pin_memory``.

        Sentinel values trigger the smart helpers in
        :mod:`src.data.components.utils`:

        * ``num_workers < 0``    -> :func:`get_smart_num_workers`
        * ``pin_memory is None`` -> :func:`auto_pin_memory` (uses the resolved
          ``num_workers`` and the trainer's root device)

        Explicit user values are passed through unchanged. Results are cached on
        the instance so all dataloaders stay consistent.
        """
        # num_workers
        if self.hparams.num_workers < 0:
            nw = get_smart_num_workers()
        else:
            nw = int(self.hparams.num_workers)

        # Best-effort device lookup; trainer may be absent in standalone use.
        device = None
        if self.trainer is not None:
            try:
                device = self.trainer.strategy.root_device
            except Exception:
                device = None

        # pin_memory (depends on resolved num_workers)
        if self.hparams.pin_memory is None:
            pm = auto_pin_memory(device, num_workers=nw)
        else:
            pm = bool(self.hparams.pin_memory)

        self._num_workers = nw
        self._pin_memory = pm
        # self.hparams.num_workers = nw
        # self.hparams.pin_memory = pm

    def _ensure_loader_settings(self) -> None:
        """Lazily resolve loader settings if ``setup()`` was not called first."""
        if self._num_workers is None:
            self._resolve_loader_settings()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # 跨 epoch 复用 worker，避免 spawn 平台（macOS/Windows）每轮重建开销；
        # num_workers=0 时必须为 False，否则 DataLoader 报错
        persistent_workers=self._num_workers > 0
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=True,
            persistent_workers = persistent_workers
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        persistent_workers=self._num_workers > 0
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
            persistent_workers = persistent_workers
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        persistent_workers=self._num_workers > 0
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            shuffle=False,
            persistent_workers = persistent_workers
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def plot_sample(self, loader: Optional[DataLoader] = None):
        """
        可视化一个 batch 中的数据
        """
        import matplotlib.pyplot as plt

        if loader is None:
            loader = self.test_dataloader()
        
        images, labels = next(iter(loader))
        
        # 创建网格图：按实际 batch 大小取列数；squeeze=False 保证 axes 恒为二维数组，
        # 避免 batch_size=1 时返回单个 Axes 导致遍历报错
        ncols = min(images.shape[0], 5)
        fig, axes = plt.subplots(1, ncols, figsize=(10, 2), squeeze=False)
        
        for i, ax in enumerate(axes[0]):
            img = images[i]
            # 如果是 (1, 32, 32) 需要转为 (32, 32) 或 (32, 32, 1)
            img = img.squeeze()
            if self.hparams.use_normalize:
                # 反归一化公式：img * std + mean，并截断到 [0,1] 便于显示
                img = (img * self.MNIST_STD + self.MNIST_MEAN).clip(0, 1)
            ax.imshow(img, cmap='viridis') # gray
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == "__main__":
    ds = MNISTDataModule(batch_size=4)
    ds.setup()
    print(ds.hparams)
