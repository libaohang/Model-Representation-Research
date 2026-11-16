import torch
from torch import nn
import math

def timestepEmbedding(t, embDim, period=10000):
    # t : (batch,)
    half = embDim // 2
    freqs = torch.exp(
        -math.log(period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    t = t.to(torch.float32)
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embDim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    # emb : (batch, embDim)
    return emb

class TimeMLP(nn.Module):
    def __init__(self, embDim, outDim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embDim, outDim * 2),
            nn.SiLU(),
            nn.Linear(outDim * 2, outDim)
        )
    def forward(self, emb):
        return self.mlp(emb)