from src.data.mnist_datamodule import MNISTDataModule


if __name__ == "__main__":
    ds = MNISTDataModule(batch_size=4,
                         use_normalize=True,
                         resize=True)
    ds.setup()
    print(ds.hparams)
    # print(ds.class_to_idx)
    # ds.plot_sample()

    train_dl = ds.train_dataloader()
    x, y = next(iter(train_dl))
    print(x.shape)
