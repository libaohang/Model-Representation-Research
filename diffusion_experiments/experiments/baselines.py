import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv

import torch

from NoiseSchedule import cosineSchedule

from probes.extract_features import (TIMESTEPS, SITE_NAMES, SITE_SHAPES, BATCH_SIZE, NOISE_SEED, T, FEATURE_DIR,
                                     RANDOM_FEATURE_DIR, RANDOM_SEED1_FEATURE_DIR, PIXEL_FEATURE_PATH,
                                     loadNetwork, registerHooks, makeDataset, extractFeatures)
from probes.extract_pixels import extractPixels
from probes.probe_data import makeBalancedSets
from experiments.probe_grid import RESULTS_DIR, runGrid

RANDOM_PATH = RESULTS_DIR / "probe_grid_random.csv"
RANDOM_SEED1_PATH = RESULTS_DIR / "probe_grid_random_seed1.csv"
PIXELS_PATH = RESULTS_DIR / "probe_grid_pixels.csv"
WIDTH64_PATH = RESULTS_DIR / "probe_width64.csv"
HELDOUT_PATH = RESULTS_DIR / "probe_heldout.csv"

PIXEL_SITES = ["pix512", "pix64"]
# the second random seed only measures how much the baseline varies between initializations
SEED1_TIMESTEPS = [100, 500]
CHECK_TIMESTEP = 100
WIDE_SITES = [name for name, (channels, _) in SITE_SHAPES.items() if channels > 64]
HELDOUT_SITES = ["mid", "up1", "up2"]


def _firstBatchFeatures(net, x0, device):
    # the first batch of valid at one timestep, noised exactly as extractSplit noised it
    sqrtA, sqrt1mA, *_ = cosineSchedule(T, device=device)
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
    eps = torch.randn(x0.shape, generator=gen, device=device)
    xt = sqrtA[CHECK_TIMESTEP] * x0 + sqrt1mA[CHECK_TIMESTEP] * eps
    tBatch = torch.full((x0.size(0),), CHECK_TIMESTEP, device=device, dtype=torch.long)

    feats, handles = registerHooks(net)
    net(xt, tBatch)
    for handle in handles:
        handle.remove()
    return dict(feats)


def _maxDifference(first, second):
    return max((first[name] - second[name]).abs().max().item() for name in SITE_NAMES)


def runChecks(device):
    dataset = makeDataset("valid")
    x0 = torch.stack([dataset[i][0] for i in range(BATCH_SIZE)]).to(device)

    # noise replay: if this fails, the baselines are not seeing the noise the trained network saw
    trained = _firstBatchFeatures(loadNetwork(device), x0, device)
    cached = torch.load(FEATURE_DIR / "valid" / f"t{CHECK_TIMESTEP:04d}.pt")
    replayDiff = _maxDifference(trained, {name: cached[name][:BATCH_SIZE].float() for name in SITE_NAMES})
    print(f"noise replay: max difference from the cached trained features = {replayDiff:.2e}")
    assert replayDiff < 1e-4, "replayed noise does not reproduce the cached trained features"

    # random network: the same seed gives the same network, and it is not the trained one
    first = _firstBatchFeatures(loadNetwork(device, 0), x0, device)
    second = _firstBatchFeatures(loadNetwork(device, 0), x0, device)
    seedDiff = _maxDifference(first, second)
    trainedDiff = _maxDifference(first, trained)
    print(f"random network: max difference between two builds = {seedDiff:.2e}, from the trained network = {trainedDiff:.2e}")
    assert seedDiff < 1e-5, "two random networks built with the same seed differ"
    assert trainedDiff > 1e-2, "the random network gives the trained features"


# each extraction is skipped when its last output file already exists
def extractBaselines(device):
    if(not (RANDOM_FEATURE_DIR / "test" / f"t{TIMESTEPS[-1]:04d}.pt").exists()):
        extractFeatures(randomSeed=0, outDir=RANDOM_FEATURE_DIR)
    if(not (RANDOM_SEED1_FEATURE_DIR / "test" / f"t{SEED1_TIMESTEPS[-1]:04d}.pt").exists()):
        extractFeatures(randomSeed=1, outDir=RANDOM_SEED1_FEATURE_DIR, timesteps=SEED1_TIMESTEPS)
    if(not PIXEL_FEATURE_PATH.exists()):
        extractPixels(device)


def _readRows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def printReferenceChecks():
    for name, path in [("random weights", RANDOM_PATH), ("noised pixels", PIXELS_PATH)]:
        noise = [float(row["testAcc"]) for row in _readRows(path) if int(row["timestep"]) == 999]
        print(f"{name}, t=999 (expect about 0.5): testAcc {min(noise):.4f} to {max(noise):.4f}")

    pix512 = [row for row in _readRows(PIXELS_PATH) if row["site"] == "pix512"]

    print("pix512, t=0 (expect well above chance)")
    for row in pix512:
        if(int(row["timestep"]) == 0 and row["attribute"] in ["Male", "Blond_Hair"]):
            print(f"{row['attribute']:<12} testAcc {float(row['testAcc']):.4f}")

    # a large gap here means the L2 list needs a stronger value for every run, not only this one
    print("pix512, largest train minus test accuracy on the rare attributes")
    for attribute in ["Eyeglasses", "Wearing_Hat"]:
        worst = max((row for row in pix512 if row["attribute"] == attribute), key=lambda row: float(row["gap"]))
        print(f"{attribute:<12} gap {float(worst['gap']):.4f} at t={worst['timestep']} (l2 {worst['l2']})")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with torch.no_grad():
        runChecks(device)
        extractBaselines(device)

    # each run writes its own file, so a finished run can be commented out
    runGrid(device, source="random", outPath=RANDOM_PATH)
    runGrid(device, source="random_seed1", outPath=RANDOM_SEED1_PATH, timesteps=SEED1_TIMESTEPS)
    runGrid(device, source="pixels", sites=PIXEL_SITES, outPath=PIXELS_PATH)
    runGrid(device, sites=WIDE_SITES, outPath=WIDTH64_PATH, timesteps=[CHECK_TIMESTEP], pcaWidth=64)
    runGrid(device, sites=HELDOUT_SITES, outPath=HELDOUT_PATH, timesteps=[CHECK_TIMESTEP],
            balanced=makeBalancedSets("people"))

    printReferenceChecks()
