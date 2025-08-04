import torch
import torch.nn as nn

from .conv import Conv

def custom_pad(k, s):
    '''
    Return padding p such that dim_out = dim_in/stride.

    This holds unless k-s<0 (in which case None is returned) or (k-s)%2==1 and dim_in%s==s-1, 
    in which case dim_out will be (dim_in+1)/s
    '''
    if k-s<0:
        return None
    return (k-s + (k-s)%2)//2


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
        self.downs = nn.ModuleList([Conv(c, c, k, factor, custom_pad(k, factor)) for factor in dwn])
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


class Downscale(nn.Module):
    def __init__(self, c, k=3, dwn=[2, 2, 2]):
        """
        Initialize Downscale module.

        Args:
            c (int): Number of channels (input, internal and output).
            k (int): Kernel size for all convolutions.
            dwn (list of ints): Factors by which to downscale at each step
        """
        super().__init__()
        self.downs = nn.ModuleList([Conv(c, c, k, factor, custom_pad(k, factor)) for factor in dwn])
        self.sides = nn.ModuleList([nn.AvgPool2d(factor, factor) for factor in dwn])
        self.downscale_factor = 1
        for d in dwn:
            self.downscale_factor *= d
    
    def forward(self, x):
        residual = torch.zeros((x.shape[0], x.shape[1], x.shape[2]//self.downscale_factor, x.shape[3]//self.downscale_factor), 
                               dtype=x.dtype, device=x.device)
        for i, m in enumerate(self.downs):
            r = x
            for j in range(i, len(self.sides)):
                r = self.sides[j](r)
            residual += r
            x=m(x)
        return x + residual
        