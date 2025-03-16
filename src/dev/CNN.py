import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for feature extraction from high-dimensional input.

    Attributes:
        net (nn.Sequential): The sequential model defining the CNN architecture.

    Methods:
        __init__(input_dimensions, output_dimensions):
            Initializes the CNN with the given input and output dimensions.
        forward(input):
            Defines the forward pass of the network.
    """
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        channel_n, height, width = input_dimensions

        if height != 84 or width != 84:
            raise ValueError(f"Invalid input ({height, width})-shape. Expected: (84, 84)")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channel_n, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, output_dimensions),
        )

    def forward(self, input):
        return self.net(input)