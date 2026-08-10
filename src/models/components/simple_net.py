import torch
from torch import nn


class SimpleLeNet(nn.Module):
    """A simple LeNet-style CNN for MNIST quick tests.

    Accepts 2D image tensors of shape ``(B, C, H, W)`` directly; no manual
    ``view``/``flatten`` is required at the input. The only flattening happens
    internally between the convolutional trunk and the fully-connected head,
    which is unavoidable in any conv classifier.

    The default layout follows LeNet-5 for a 32x32 input:

        conv5x5(1->6)  -> ReLU -> pool2  : 6x28x28 -> 6x14x14
        conv5x5(6->16) -> ReLU -> pool2  : 16x10x10 -> 16x5x5
        flatten                          : 400
        fc(400->120) -> ReLU
        fc(120->84)  -> ReLU
        fc(84->10)
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 32,
        conv1_channels: int = 6,
        conv2_channels: int = 16,
        fc1_size: int = 120,
        fc2_size: int = 84,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_channels: Number of input image channels (1 for MNIST).
        :param input_size: Spatial size of the (square) input, e.g. 32 for the
            LeNet-classic 32x32 input. Used to infer the flattened conv output
            size so the head is wired correctly for any input resolution.
        :param conv1_channels: Number of filters in the first conv layer.
        :param conv2_channels: Number of filters in the second conv layer.
        :param fc1_size: Width of the first fully-connected layer.
        :param fc2_size: Width of the second fully-connected layer.
        :param output_size: Number of output classes.
        """
        super().__init__()

        # LeNet-5 trunk: (conv5x5 -> ReLU -> maxpool2) x2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv1_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Infer the flattened feature size for the given input resolution.
        # Each (conv5x5, pool2) block maps s -> (s - 4) // 2.
        s = input_size
        s = (s - 5 + 1) // 2  # after conv1 + pool1
        s = (s - 5 + 1) // 2  # after conv2 + pool2
        flatten_size = conv2_channels * s * s

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: Input image tensor of shape ``(B, C, H, W)``.
        :return: A tensor of logits of shape ``(B, output_size)``.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # (B, C, H, W) -> (B, C*H*W), conv->FC transition
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    f"""A simple MLP neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 1024,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleMLP` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)



if __name__ == "__main__":
    _ = SimpleMLP()
