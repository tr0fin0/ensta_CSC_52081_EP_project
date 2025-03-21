import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class PolicyNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dim):
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
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(256, output_dim)  # Mean of actions
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Learnable log_std

    def forward(self, input):
        x = self.net(input)
        mu = self.mu_layer(x)  # Mean of action
        std = torch.exp(self.log_std)  # Convert log_std to std (ensures positivity)

        mu[:, 0] = torch.tanh(mu[:, 0])  # Steering: [-1, 1]
        mu[:, 1:] = torch.sigmoid(mu[:, 1:])  # Gas and brake: [0, 1]
        
        return mu, std


class ValueNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dim=None):
        """
        Args:
            input_dimensions (tuple): Shape of the observation space (C, H, W).
            output_dim (int): Action space dimension (only used in policy mode).
            mode (str): "policy" for action selection, "value" for value estimation.
        """
        super().__init__()
        channel_n, height, width = input_dimensions

        if height != 84 or width != 84:
            raise ValueError(f"Invalid input ({height, width})-shape. Expected: (84, 84)")

        self.conv1 = nn.Conv2d(channel_n, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.value_head = nn.Linear(256, output_dim)  # Predicts single value

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.value_head(x)  # Return scalar value
