import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from probes.extract_features import FEATURE_DIR, SPLITS, makeDataset
from probes.make_control import ATTR, ATTR_NAMES, ATTRIBUTES, BALANCED, IDENTITY, PROBE_SPLITS, SPLIT_OF, makeProbeData

PIXELS_PATH = FEATURE_DIR / "pixels8.pt"
PIXEL_GRID = 8
PCA_COMPONENTS = 32
KMEANS_ITERATIONS = 50


# clean images average-pooled to 8x8 and flattened, in pooled order (all of valid, then all of test)
def loadPixels():
    if(PIXELS_PATH.exists()):
        return torch.load(PIXELS_PATH)

    parts = []
    for split in SPLITS:
        loader = DataLoader(makeDataset(split), batch_size=512, shuffle=False, num_workers=4)
        for x0, _ in loader:
            parts.append(F.adaptive_avg_pool2d(x0, PIXEL_GRID).flatten(1))
        print(f"  {split}: pixels loaded")
    pixels = torch.cat(parts)
    assert len(pixels) == len(IDENTITY), "pixel rows do not match the pooled label rows"
    torch.save(pixels, PIXELS_PATH)
    return pixels


def _kMeans(points, k, seed):
    gen = torch.Generator(device=points.device).manual_seed(seed)

    # k-means++ start: each new centroid is drawn in proportion to its squared distance from the ones chosen so far
    first = torch.randint(len(points), (1,), generator=gen, device=points.device)
    centroids = points[first]
    closest = torch.cdist(points, centroids).squeeze(1) ** 2
    for _ in range(k - 1):
        pick = torch.multinomial(closest / closest.sum(), 1, generator=gen)
        centroids = torch.cat([centroids, points[pick]])
        closest = torch.minimum(closest, torch.cdist(points, points[pick]).squeeze(1) ** 2)

    for _ in range(KMEANS_ITERATIONS):
        assignment = torch.cdist(points, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, assignment, points)
        counts = torch.bincount(assignment, minlength=k)
        # a cluster that lost all its points keeps its previous centroid
        filled = counts > 0
        centroids[filled] = sums[filled] / counts[filled, None]
    return centroids


# clusters are defined by probe-train images only; every other image goes to its nearest centroid
def fitClusters(pixels, k, seed=0, device="cpu"):
    trainRows = torch.from_numpy(np.flatnonzero(SPLIT_OF == PROBE_SPLITS.index("TRAIN"))).to(device)
    X = pixels.to(device)
    pixelMean = X[trainRows].mean(dim=0)
    pixelStd = X[trainRows].std(dim=0).clamp_min(1e-6)
    X = (X - pixelMean) / pixelStd

    # PCA directions from the standardized (so already centered) probe-train rows
    _, _, components = torch.linalg.svd(X[trainRows], full_matrices=False)
    projected = X @ components[:PCA_COMPONENTS].T

    centroids = _kMeans(projected[trainRows], k, seed)
    return torch.cdist(projected, centroids).argmin(dim=1).cpu().numpy()


def makeClusterLabels(clusterIds, k, seed):
    # one random 0/1 label per cluster, inherited by every image in it
    rng = np.random.default_rng([seed, k])
    labelOfCluster = rng.integers(0, 2, size=k)
    return labelOfCluster[clusterIds]


# swaps the control labels of an existing makeProbeData dict, so features are standardized only once per cell
def applyClusterLabels(data, attribute, clusterLabels):
    device = data["PROBE_TRAIN_X"].device
    for split in PROBE_SPLITS:
        rows = BALANCED[attribute][split]
        data[f"PROBE_{split}_Y_CONTROL"] = torch.from_numpy(clusterLabels[rows]).float().to(device)
    return data


def makeClusterProbeData(features, attribute, site, clusterLabels, device="cpu"):
    return applyClusterLabels(makeProbeData(features, attribute, site, device), attribute, clusterLabels)


def describeClusters(clusterIds, k):
    trainCode = PROBE_SPLITS.index("TRAIN")
    trainSizes = np.bincount(clusterIds[SPLIT_OF == trainCode], minlength=k)
    summary = {
        "k": k,
        "minSize": int(trainSizes.min()),
        "medianSize": float(np.median(trainSizes)),
        "maxSize": int(trainSizes.max()),
        "emptyClusters": int((trainSizes == 0).sum()),
    }
    for split in ["VAL", "TEST"]:
        rows = SPLIT_OF == PROBE_SPLITS.index(split)
        summary[f"{split.lower()}Seen"] = float((trainSizes[clusterIds[rows]] > 0).mean())

    # how well the cluster id alone predicts each attribute: majority vote per cluster on balanced train, scored on balanced val
    for attribute in ATTRIBUTES:
        column = ATTR_NAMES.index(attribute)
        trainRows, valRows = BALANCED[attribute]["TRAIN"], BALANCED[attribute]["VAL"]
        positives = np.bincount(clusterIds[trainRows], weights=ATTR[trainRows, column], minlength=k)
        totals = np.bincount(clusterIds[trainRows], minlength=k)
        vote = positives > 0.5 * totals
        summary[attribute] = float((vote[clusterIds[valRows]] == ATTR[valRows, column]).mean())
    return summary
