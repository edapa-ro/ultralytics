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
    def __init__(self, c, k=3, dwn=[2, 2, 2], pooling='avg'):
        """
        Initialize Downscale module.

        Args:
            c (int): Number of channels (input, internal and output).
            k (int): Kernel size for all convolutions.
            dwn (list of ints): Factors by which to downscale at each step
            pooling ('avg' or 'max'): type of pooling to use. default avg
        """
        super().__init__()
        if pooling == 'max':
            pool = nn.MaxPool2d
        else:
            pool = nn.AvgPool2d
        self.downs = nn.ModuleList([Conv(c, c, k, factor, custom_pad(k, factor)) for factor in dwn])
        self.sides = nn.ModuleList([pool(factor, factor) for factor in dwn])
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


class DenseBlock(nn.Module):
    def __init__(self, ch, k=3, depth=4, growth=16, bottleneck=False, bottleneck_factor=4.0):
        """
        Initialize DenseBlock module.

        Args:
            ch (int): input channel count
            k (int or tuple(int, int)): kernel size
            depth (int): internal layer count
            growth (int): growth factor. the final output will have ch+depth*growth features
            bottleneck (bool): whether or not to use 1x1 convolutions as bottlenecks before internal layers
            bottleneck_factor (float): if bottleneck is True, 
                each bottleneck convolution will produce growth*bottleneck_factor features
        """
        super().__init__()
        self.layers = nn.ModuleList(Conv(ch+d*growth, growth, k) if not bottleneck 
                                    else nn.Sequential(Conv(ch+d*growth, int(growth*bottleneck_factor), 1), 
                                                       Conv(int(growth*bottleneck_factor), growth, k))
                                    for d in range(depth))
    
    def forward(self, x):
        for m in self.layers:
            x = torch.concat([x, m(x)], 1)
        return x
    
    def get_output_ch_count(ch, k=3, depth=4, growth=16, bottleneck=False, bottleneck_factor=4.0):
        """
        Return the number of output channels from init args.
        """
        return ch + depth*growth


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, kdense=3, ktrans=3, depth=4, growth=16, bottleneck=False, bottleneck_factor=4.0):
        """
        Initialize Fusion First CSPBlock module.

        Args:
            c1 (int): input channel count
            c2 (int): output channel count
            k (int or tuple(int, int)): kernel size
            depth (int): internal layer count
            growth (int): growth factor. the final output will have ch+depth*growth features
            bottleneck (bool): whether or not to use 1x1 convolutions as bottlenecks before internal layers
            bottleneck_factor (float): if bottleneck is True, 
                each bottleneck convolution will produce growth*bottleneck_factor features
        """
        super().__init__()
        self.dense = DenseBlock(c1//2+c1%2, kdense, depth, growth, bottleneck, bottleneck_factor)
        self.transition = Conv(c1+growth*depth, c2, ktrans)
    
    def forward(self, x):
        x0, x1 = torch.chunk(x, 2, 1)
        return self.transition( torch.concat((self.dense(x0), x1), 1) )


class ProperChannelAttention(nn.Module):
    def __init__(self, ch, use_mlp=True, r=16.0):
        """
        Initialize ChannelAttention module with both AvgPooling and MaxPooling, as well as an MLP option.

        Args:
            ch (int): input channel count
            use_mlp (bool): True to use an MLP with one hidden layer (like in the paper) before the summation,
                False to use a single layer
            r (float): reduction ratio. if use_mlp is True, the hidden layer of the mlp will have a size of ch/r.
        """
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.m = nn.Sequential(Conv(ch, int(ch/r), 1), nn.Conv2d(int(ch/r), ch, 1)) if use_mlp else nn.Conv2d(ch, ch, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.act( self.m(self.avg(x)) + self.m(self.max(x)) )


class ProperSpatialAttention(nn.Module):
    def __init__(self, k=7):
        """
        Initialize a Spatial Attention module that can (hopefully) be compiled for Coral EdgeTPU.

        Args:
            k (int): kernel size
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, custom_pad(k, 1))
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        att = self.act(self.conv(torch.concat((torch.mean(x, 1, True), torch.max(x, 1, True)[0]), 1)))
        # this is probably terrible for performance, but it should be compilable for Coral EdgeTPU
        return x * torch.concat([att for _ in range(x.shape[1])], 1)
        # return x * att.expand(x.shape)


class ProperCBAM(nn.Module):
    def __init__(self, ch, k=7, use_mlp=False, r=4.0):
        """
        Initialize a CBAM module with both avgpool and maxpool (as well as an MLP option) in ChannelAttention and
        with a Coral EdgeTPU-compilable SpatialAttention.

        Args:
            ch (int): input channel count (for ChannelAttention)
            k (int): kernel size for SpatialAttention
            use_mlp (bool): for ChannelAttention: True to use an MLP with one hidden layer (like in the paper)
                before the summation, False to use a single layer
            r (float): for ChannelAttention: reduction ratio.
                if use_mlp is True, the hidden layer of the mlp will have a size of ch/r.
        """
        super().__init__()
        self.chatt=ProperChannelAttention(ch, use_mlp, r)
        self.spatt=ProperSpatialAttention(k)
    
    def forward(self, x):
        return self.spatt(self.chatt(x))