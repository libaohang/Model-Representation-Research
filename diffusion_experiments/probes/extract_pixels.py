import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import torch
from torch.utils.data import DataLoader

from probes.extract_features import SPLITS, BATCH_SIZE, NOISE_SEED, PIXEL_FEATURE_PATH, makeDataset
from probes.probe_data import SPLIT_OF, PROBE_SPLITS

PCA_WIDTH = 512
PCA_SEED = 0


def loadImages():
    # one (N, 12288) tensor per split, in the order the trained features were extracted
    images = []
    for split in SPLITS:
        loader = DataLoader(makeDataset(split), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        images.append(torch.cat([x0.flatten(1) for x0, _ in loader]))
        print(f"{split}: {images[-1].size(0)} images")
    return images


def fitDirections(images, device):
    # fitted on probe-train rows only, so the baseline never sees probe val or test images
    trainRows = torch.from_numpy(SPLIT_OF == PROBE_SPLITS.index("TRAIN"))
    trainX = torch.cat(images)[trainRows].to(device)
    print(f"PCA fitted on {trainX.size(0)} probe-train images")
    torch.manual_seed(PCA_SEED)
    _, _, directions = torch.pca_lowrank(trainX, q=PCA_WIDTH, center=True, niter=4)
    return directions


def projectSplit(images, directions, splitIdx, device):
    # replays the noise of extractSplit: same seed, same batch shapes, so the same eps for every image
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED + splitIdx)
    z0, zEps = [], []
    for start in range(0, images.size(0), BATCH_SIZE):
        x0 = images[start:start + BATCH_SIZE].to(device)
        eps = torch.randn((x0.size(0), 3, 64, 64), generator=gen, device=device)
        z0.append((x0 @ directions).cpu())
        zEps.append((eps.flatten(1) @ directions).cpu())
    return torch.cat(z0), torch.cat(zEps)


def extractPixels(device):
    # CPU and GPU generators give different streams, and the trained features were noised on the GPU
    assert device.type == "cuda", "the noise replay only matches the trained features on the GPU"

    images = loadImages()
    directions = fitDirections(images, device)

    projected = [projectSplit(images[splitIdx], directions, splitIdx, device) for splitIdx in range(len(SPLITS))]
    z0 = torch.cat([z for z, _ in projected])
    zEps = torch.cat([z for _, z in projected])
    assert z0.size(0) == len(SPLIT_OF) and zEps.size(0) == len(SPLIT_OF), "pixel rows do not match the label rows"

    PIXEL_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z0": z0, "zEps": zEps, "directions": directions.cpu()}, PIXEL_FEATURE_PATH)
    print(f"saved {tuple(z0.shape)} projected images and noise to {PIXEL_FEATURE_PATH}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with torch.no_grad():
        extractPixels(device)
