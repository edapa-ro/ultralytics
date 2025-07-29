import torch
import torch.nn as nn

from .conv import Conv


class HourglassConv(nn.Module):
    def __init__(self, c, k=3, dwn=[2, 2, 2], skip=0):
        """
        Initialize Hourglass module with Convs.

        Args:
            c (int): Number of channels (input, internal and output).
            k (int): Kernel size for all convolutions.
            dwn (list of ints): Factors by which to downscale at each step
            skip (int): Number of upsamples to skip.
        """
        assert skip<=len(dwn)
        super().__init__()
        self.skip=skip
        self.downs = nn.ModuleList([Conv(c, c, k, factor) for factor in dwn])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=dwn[len(dwn)-i-1], mode='nearest') for i in range(len(dwn)-skip)])
    
    def forward(self, x):
        residuals = []
        for i, m in enumerate(self.downs):
            if i >= self.skip:
                residuals.append(x)
            x = m(x)
        for i, m in enumerate(self.ups):
            x = m(x) + residuals[-1]
            residuals.pop()
        return x
        