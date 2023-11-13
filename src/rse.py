import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class gelu(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.script
    def forward(x):
        return x * torch.sigmoid(1.702 * x)

class CustomLayerNorm(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, 1, num_params))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mean = torch.mean(x, dim=1, keepdim=True)
        x = x - mean
        d = x + self.offset
        x = d + x
        variance = torch.mean(x**2, dim=1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-10)
        x = x.permute(0, 2, 1)
        return x


class CustomLayerNorm1(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(1, 1, num_params))

    def forward(self, x):  # input (batch_size,32 features,~768 channels)
        mean = torch.mean(x, dim=1, keepdim=True)
        x = x - mean
        d = x + self.offset
        x = d + x
        variance = torch.mean(x**2, dim=1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-10)
        return x


@torch.jit.script
def dropout(d, len: int, is_training: bool):
    """Dropout dependent on sequence length"""
    prob = 0.1 / len
    d = F.dropout(d, p=prob, training=is_training, inplace=False)
    return d


class SwitchLayer(nn.Module):
    def __init__(self, m, r=0.9):
        super().__init__()
        self.unit = nn.Sequential(
            nn.Linear(2 * m, 4 * m, bias=False),
            CustomLayerNorm1(4 * m),
            gelu(),
            nn.Linear(4 * m, 2 * m, bias=True),
        )

        self.s = nn.Parameter(torch.empty(2 * m))
        self.h = 0.25 * np.sqrt(1 - r**2)

        nn.init.constant_(self.s, np.log(r / (1 - r)) / 2)

    def forward(self, x):
        n_bits = int(x.shape[1] - 1).bit_length()
        x = x.view(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        c = self.unit(x)
        out = torch.mul(torch.sigmoid(self.s * 2), x) + torch.mul(self.h, c)
        return dropout(
            out.view(out.shape[0], out.shape[1] * 2, out.shape[2] // 2),
            n_bits,
            self.training,
        )


class ShuffleLayer(nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    @staticmethod
    @torch.jit.script
    def ror(x: int, n: int, p: int = 1):
        """Bitwise rotation right p positions
        n is the bit length of the number
        """
        return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))

    @staticmethod
    @torch.jit.script
    def rol(x: int, n: int, p: int = 1):
        """Bitwise rotation left p positions
        n is the bit length of the number
        """
        return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))

    def forward(self, x):
        length = x.shape[1]
        n_bits = int(length - 1).bit_length()
        if self.reverse:
            rev_indices = [self.ror(i, n_bits) for i in range(length)]
        else:
            rev_indices = [self.rol(i, n_bits) for i in range(length)]
        return x[..., rev_indices, :]


class BenesBlock(nn.Module):
    def __init__(self, m, r=0.9):
        super().__init__()
        self.regular_switch = SwitchLayer(m, r)
        self.regular_shuffle = ShuffleLayer(reverse=False)
        self.reverse_switch = SwitchLayer(m, r)
        self.reverse_shuffle = ShuffleLayer(reverse=True)

    def forward(self, x):
        k = int(x.shape[1] - 1).bit_length()
        for _ in range(k - 1):
            x = self.regular_switch(x)
            x = self.regular_shuffle(x)
        for _ in range(k - 1):
            x = self.reverse_switch(x)
            x = self.reverse_shuffle(x)
        return x


class ResidualShuffleExchangeNetwork(nn.Module):
    def __init__(self, m, n_blocks=1, r=0.9):
        super().__init__()
        self.blocks = nn.Sequential(
            OrderedDict({f"benes_block_{i}": BenesBlock(m, r) for i in range(n_blocks)})
        )
        self.final_switch = SwitchLayer(m, r)

    def forward(self, x):
        n = 1 << int(x.shape[1] - 1).bit_length()
        x = F.pad(x, (0, 0, 0, n - x.shape[1]))
        x = self.blocks(x)
        x = self.final_switch(x)
        return x
