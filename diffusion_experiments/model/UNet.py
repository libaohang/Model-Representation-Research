from torch import nn
from Blocks import ResidualBlock, UpBlock, DownBlock
from TimeEmbedding import timestepEmbedding, TimeMLP

class UNet(nn.Module):
    def __init__(self, inChannel, channels, attention, numRes, neckLength, timeEmbDim=64):
        super().__init__()
        assert len(channels) >= 3
        assert len(attention) == len(channels)

        self.timeEmbDim = timeEmbDim

        self.timeMLP = TimeMLP(timeEmbDim, timeEmbDim)

        self.stem = nn.Conv2d(inChannel, channels[0], 3, padding=1)

        self.downBlocks = nn.ModuleList(
            [DownBlock(channels[0], channels[0], numRes, timeEmbDim, attention[0])] +
            [DownBlock(channels[i-1], channels[i], numRes, timeEmbDim, attention[i], i==len(channels)-1) for i in range(1, len(channels))]
        )

        self.bottleneck = nn.ModuleList(
            [ResidualBlock(channels[-1], timeEmbDim=timeEmbDim) for _ in range(neckLength)]
        )

        self.upBlocks = nn.ModuleList(
            [UpBlock(channels[-1], channels[-1], numRes, timeEmbDim, attention[-1], True)] +
            [UpBlock(channels[i+1], channels[i], numRes, timeEmbDim, attention[i]) for i in range(len(channels)-2, -1, -1)]
        )

        self.head = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], inChannel, 3, padding=1)
        )
        
    def forward(self, image, time):
        timeEmb = timestepEmbedding(time, self.timeEmbDim)
        timeEmb = self.timeMLP(timeEmb)

        residuals = []
        image = self.stem(image)

        for block in self.downBlocks:
            #print(image.shape, "down block")
            image, residual = block(image, timeEmb)
            # residual is a list of all residuals in a block
            residuals.append(residual)
        
        for block in self.bottleneck:
            #print(image.shape, "bottleneck")
            image = block(image, timeEmb)

        residuals = reversed(residuals)

        for block, residual in zip(self.upBlocks, residuals, strict=True):
            #print(image.shape, "up block")
            image = block(image, residual, timeEmb)

        image = self.head(image)
        #print(image.shape, "end")
        return image