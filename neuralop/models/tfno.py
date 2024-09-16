import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch
# from .spectral_convolution import FactorizedSpectralConv
# from .padding import DomainPadding
# from .fno_block import FNOBlocks, resample


class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x
