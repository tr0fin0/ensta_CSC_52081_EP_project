import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Rede neural convolucional para extração de features de imagens.
    Usada como rede de política para controle contínuo.

    Atributos:
        net (nn.Sequential): Modelo sequencial que define a arquitetura.
    """
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        channel_n, height, width = input_dimensions

        if height != 84 or width != 84:
            raise ValueError(f"Formato de entrada inválido ({height, width}). Esperado: (84, 84)")

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
