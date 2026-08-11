from src.data.mnist_datamodule import MNISTDataModule
import torch

"""
hydsen
hufangyang86@qq.com

韩帅
18686156062@163.com

lyf
2417500668@qq.com

Captain
hl942049624@163.com

zheng
haozheng260@gmail.com
"""


if __name__ == "__main__":
    ds = MNISTDataModule(batch_size=4,
                         use_normalize=True,
                         resize=True)
    # ds.setup()
    # print(ds.hparams)
    # # print(ds.class_to_idx)
    # # ds.plot_sample()

    # train_dl = ds.train_dataloader()
    # x, y = next(iter(train_dl))
    # print(x.shape)

    ckpt_fn = "/Users/mac/Personspace/repo/torch-cv-foundation-lightning/logs/train/runs/2026-08-10_17-14-10/checkpoints/epoch_008.ckpt"
    model = torch.load(ckpt_fn)
    print(model.keys())

