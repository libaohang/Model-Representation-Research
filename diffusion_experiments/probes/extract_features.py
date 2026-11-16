import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from UNet import UNet
from EMA import EMA
from SaveLoad import loadEMA
from NoiseSchedule import cosineSchedule

DATA_ROOT = ROOT / "data"
FEATURE_DIR = DATA_ROOT / "features"

SPLITS = ["valid", "test"]
TIMESTEPS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

# site name -> (channels, resolution), in network order
SITE_SHAPES = {
    "down0": (64, 64),
    "down1": (128, 32),
    "down2": (256, 16),
    "down3": (512, 8),
    "down4": (512, 4),
    "mid": (512, 4),
    "up0": (512, 4),
    "up1": (512, 8),
    "up2": (256, 16),
    "up3": (128, 32),
    "up4": (64, 64),
}
SITE_NAMES = list(SITE_SHAPES.keys())

STEP = 229407
T = 1000
BATCH_SIZE = 256
NOISE_SEED = 0


def loadNetwork(device):
    net = UNet(3, [64, 128, 256, 512, 512], [False, False, True, True, False], 3, 2, 64).to(device)
    # loadEMA builds "{dataset}checkpoints/...", so an absolute prefix makes it independent of the cwd
    ema = loadEMA(STEP, EMA(net), str(ROOT / "diffusion_model" / "CelebA"), device=device)
    ema.applyShadow(net)
    net.eval()
    return net


def siteModules(net):
    return {
        "down0": net.downBlocks[0].resBlocks[-1],
        "down1": net.downBlocks[1].resBlocks[-1],
        "down2": net.downBlocks[2].atten,
        "down3": net.downBlocks[3].atten,
        "down4": net.downBlocks[4].resBlocks[-1],
        "mid": net.bottleneck[-1],
        "up0": net.upBlocks[0],
        "up1": net.upBlocks[1],
        "up2": net.upBlocks[2],
        "up3": net.upBlocks[3],
        "up4": net.upBlocks[4],
    }


def registerHooks(net, pool=True):
    feats = {}

    def makeHook(name):
        def hook(module, inputs, output):
            if pool:
                feats[name] = output.float().mean(dim=(2, 3)).cpu()
            else:
                feats[name] = output
        return hook

    handles = [module.register_forward_hook(makeHook(name)) for name, module in siteModules(net).items()]
    return feats, handles


def makeDataset(split):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return datasets.CelebA(root=str(DATA_ROOT), split=split, target_type=["attr", "identity"],
                           download=True, transform=transform)


def sanityChecks(net, x0, device):
    sqrtA, sqrt1mA, *_ = cosineSchedule(T, device=device)
    x0 = x0.to(device)
    batchSize = x0.size(0)

    # weights loaded: a network without the EMA weights predicts noise with MSE near 1
    eps = torch.randn_like(x0)
    tBatch = torch.full((batchSize,), 500, device=device, dtype=torch.long)
    xt = sqrtA[500] * x0 + sqrt1mA[500] * eps
    mse = F.mse_loss(net(xt, tBatch), eps).item()
    print(f"weight check: noise MSE at t=500 = {mse:.4f}")
    assert mse < 0.2, "noise MSE is too high, the EMA weights were probably not applied"

    # hook shapes before pooling
    raw, handles = registerHooks(net, pool=False)
    net(xt, tBatch)
    for name, (channels, resolution) in SITE_SHAPES.items():
        expected = (batchSize, channels, resolution, resolution)
        assert tuple(raw[name].shape) == expected, f"{name}: got {tuple(raw[name].shape)}, expected {expected}"
    for handle in handles:
        handle.remove()
    print("hook shapes: all 11 sites match")

    # determinism
    feats, handles = registerHooks(net)
    net(xt, tBatch)
    first = dict(feats)
    net(xt, tBatch)
    maxDiff = max((first[name] - feats[name]).abs().max().item() for name in SITE_NAMES)
    for handle in handles:
        handle.remove()
    print(f"determinism: max feature difference between two passes = {maxDiff:.2e}")
    assert maxDiff < 1e-5, "features differ between two passes over the same batch"


def extractSplit(net, split, loader, device, seed):
    sqrtA, sqrt1mA, *_ = cosineSchedule(T, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    feats, handles = registerHooks(net)
    store = {t: {name: [] for name in SITE_NAMES} for t in TIMESTEPS}

    for batchIdx, (x0, _) in enumerate(loader):
        x0 = x0.to(device)
        # one noise sample per batch, reused at every timestep
        eps = torch.randn(x0.shape, generator=gen, device=device)
        for t in TIMESTEPS:
            xt = sqrtA[t] * x0 + sqrt1mA[t] * eps
            tBatch = torch.full((x0.size(0),), t, device=device, dtype=torch.long)
            net(xt, tBatch)
            for name in SITE_NAMES:
                store[t][name].append(feats[name])
        if(batchIdx % 10 == 0):
            print(f"  {split}: batch {batchIdx + 1}/{len(loader)}")

    for handle in handles:
        handle.remove()

    outDir = FEATURE_DIR / split
    outDir.mkdir(parents=True, exist_ok=True)
    for t in TIMESTEPS:
        torch.save({name: torch.cat(store[t][name]) for name in SITE_NAMES}, outDir / f"t{t:04d}.pt")


def saveLabels(dataset, split):
    outDir = FEATURE_DIR / split
    outDir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "attr": dataset.attr.to(torch.uint8),
        "identity": dataset.identity.reshape(-1).to(torch.long),
        "attrNames": [name for name in dataset.attr_names if name],
    }, outDir / "labels.pt")


def extractFeatures():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with torch.no_grad():
        net = loadNetwork(device)

        for splitIdx, split in enumerate(SPLITS):
            dataset = makeDataset(split)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"{split}: {len(dataset)} images")

            if(splitIdx == 0):
                sanityChecks(net, torch.stack([dataset[i][0] for i in range(16)]), device)

            saveLabels(dataset, split)
            extractSplit(net, split, loader, device, NOISE_SEED + splitIdx)


if __name__ == "__main__":
    extractFeatures()
