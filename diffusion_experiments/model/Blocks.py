import torch
from torch import nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels=None, timeEmbDim=None):
        super().__init__()
        if outChannels == None or outChannels == inChannels:
            outChannels = inChannels
            self.residual = None
        else:
            self.residual = nn.Conv2d(inChannels, outChannels, 1)

        if timeEmbDim == None:
            self.timeEmbProj = None
        else:
            self.timeEmbProj = nn.Linear(timeEmbDim, outChannels)


        self.norm1 = nn.GroupNorm(32, inChannels)
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding=1)
        

    def forward(self, input, timeEmb=None):
        x = self.norm1(input)
        x = self.conv1(F.silu(x))

        if(timeEmb is not None):
            assert self.timeEmbProj is not None
            # temb: (1, timeEmbDim) -> (batch, outChannels, 1, 1)
            x = x + self.timeEmbProj(F.silu(timeEmb))[:, :, None, None]

        x = self.norm2(x)
        x = self.conv2(F.silu(x))

        if(self.residual == None):
            return input + x
        else:
            return self.residual(input) + x
        

class DownBlock(nn.Module):
    def __init__(self, inChannels, blockChannels, numRes, timeEmbDim=None, attention=False, noDownsample=False):
        super().__init__()
        self.blockChannels = blockChannels
        self.resBlocks = nn.ModuleList(
            [ResidualBlock(inChannels, blockChannels, timeEmbDim)] +
            [ResidualBlock(blockChannels, timeEmbDim=timeEmbDim) for _ in range(numRes - 1)]
        )
        if attention:
            self.atten = AttentionBlock(blockChannels, 4)
        else:
            self.atten = None

        if (not noDownsample):
            self.down = nn.Conv2d(blockChannels, blockChannels, 3, stride=2, padding=1)
        else:
            self.down = None

    def forward(self, input, timeEmb=None):
        residuals = []
        x = input
        for resBlock in self.resBlocks:
            x = resBlock(x, timeEmb)
            residuals.append(x)

        if(self.atten is not None):
            x = self.atten(x)

        if(self.down is not None):
            x = self.down(x)
        return x, residuals


class UpBlock(nn.Module):
    def __init__(self, inChannels, blockChannels, numRes, timeEmbDim=None, attention=False, noUpsample=False):
        super().__init__()
        self.inChannels = inChannels
        if(not noUpsample):
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = None
        
        if attention:
            self.atten = AttentionBlock(inChannels, 4)
        else:
            self.atten = None
        # Residual blocks take in concatenated input
        self.resBlocks = nn.ModuleList(
            [ResidualBlock(inChannels + blockChannels, blockChannels, timeEmbDim)] +
            [ResidualBlock(blockChannels * 2, blockChannels, timeEmbDim=timeEmbDim) for _ in range(numRes - 1)]
        )
    
    def forward(self, input, residuals, timeEmb=None):
        x = input

        if(self.up is not None):
            x = self.up(x)

        if(self.atten is not None):
            x = self.atten(x)

        residuals = reversed(residuals)
        for residual, resBlock in zip(residuals, self.resBlocks, strict=True):
            x = torch.cat((x, residual), dim=1)
            x = resBlock(x, timeEmb)
        return x
        

class AttentionBlock(nn.Module):
    def __init__(self, embedDim, numHeads):
        super().__init__()
        self.embedDim = embedDim
        self.norm = nn.GroupNorm(32, embedDim)
        self.atten = nn.MultiheadAttention(embedDim, numHeads, batch_first=True)

        self.proj = nn.Linear(embedDim, embedDim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input):
        batch, channels, height, width = input.shape
        assert self.embedDim == channels
        x = self.norm(input)
        x = x.flatten(2).transpose(1, 2)
        # x : (batch, height * width, channel)
        x, _ = self.atten(x, x, x, need_weights=False)
        x = self.proj(x)

        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        # x : (batch, channel, height, width)
        return input + x