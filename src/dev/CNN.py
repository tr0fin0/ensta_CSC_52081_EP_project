import torch
import torch.nn as nn

class CNN_PPO(nn.Module):
    """
    A Convolutional Neural Network (CNN) for both policy and value estimation.
    """

    def __init__(self, input_dimensions, output_dim=None, mode="policy"):
        """
        Args:
            input_dimensions (tuple): Shape of the observation space (C, H, W).
            output_dim (int): Action space dimension (only used in policy mode).
            mode (str): "policy" for action selection, "value" for value estimation.
        """
        super().__init__()
        self.mode = mode
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
        )

        # Output layers for policy network
        if mode == "policy":
            self.mu_layer = nn.Linear(256, 1)  # Continuous steering action (-1 to 1)
            self.log_std_layer = nn.Linear(256, 1)  # Log standard deviation for steering
            self.discrete_layer = nn.Linear(256, 2)  # Binary actions (gas and brake)

        # Output layer for value function
        elif mode == "value":
            self.value_layer = nn.Linear(256, 1)  # Single scalar value output

    def forward(self, input):
        features = self.net(input)

        if self.mode == "policy":
            # Continuous steering action
            mu = torch.tanh(self.mu_layer(features))  # Output between -1 and 1
            log_std = torch.clamp(self.log_std_layer(features), min=-20, max=2)  # Log std for stability
            std = torch.exp(log_std)  # Convert log_std to std

            # Discrete actions (gas and brake as probabilities)
            logits = self.discrete_layer(features)  # Raw logits for binary actions

            return mu, std, logits  # Return mean, std, and logits

        elif self.mode == "value":
            return self.value_layer(features)  # Return scalar value estimate
