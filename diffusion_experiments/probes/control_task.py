import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import torch
import torch.nn.functional as F
from torch import nn

from probes.make_control import loadFeatures, makeProbeData

PROBE_WIDTHS = [32, 64, 128, 256]
WEIGHT_DECAYS = [0.0, 1e-2, 1e-1, 1.0]

# weight penalties tried for the linear probe, strongest first; the best one differs by cell
# (large image sets want almost none, rare attributes at 512-dimensional sites want much more)
LINEAR_L2S = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
MLP_EPOCHS = 60


def makeProbe(inputDim, hiddenWidth=None):
    if(hiddenWidth is None):
        return nn.Linear(inputDim, 1)
    return nn.Sequential(nn.Linear(inputDim, hiddenWidth), nn.GELU(), nn.Linear(hiddenWidth, 1))


# full-batch logistic regression solved with L-BFGS from zero weights: no learning rate, epoch count or
# stopping rule to tune, and the same answer on every run
def _trainLinear(X, Y, l2):
    X, Y = X.double(), Y.double()
    probe = makeProbe(X.size(1)).to(X.device).double()
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=1000, history_size=20, tolerance_grad=1e-6,
                                  tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(X).squeeze(1), Y) + 0.5 * l2 * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


# fixed number of epochs with no early stopping, so every run gets the same budget
def _trainMLP(probe, X, Y, weightDecay, epochs, batchSize, lr):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weightDecay)
    n = X.size(0)
    for _ in range(epochs):
        indices = torch.randperm(n, device=X.device)
        for start in range(0, n, batchSize):
            batch = indices[start:start + batchSize]
            loss = F.binary_cross_entropy_with_logits(probe(X[batch]).squeeze(1), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def _evaluate(probe, X, Y):
    # the linear probe is trained in double precision, the MLP in single
    logits = probe(X.to(next(probe.parameters()).dtype)).squeeze(1)
    return ((logits > 0) == (Y > 0.5)).float().mean().item()


# one probe with one regularization value: the L2 penalty for the linear probe (hiddenWidth=None),
# AdamW weight decay for the MLP
def trainProbe(data, hiddenWidth=None, regularization=0.0, evalSplit="TEST", epochs=MLP_EPOCHS, batchSize=256, lr=1e-3, seed=0):
    trainX, trainY = data["PROBE_TRAIN_X"], data["PROBE_TRAIN_Y"]

    if(hiddenWidth is None):
        probe = _trainLinear(trainX, trainY, regularization)
    else:
        torch.manual_seed(seed)
        probe = makeProbe(trainX.size(1), hiddenWidth).to(trainX.device)
        _trainMLP(probe, trainX, trainY, regularization, epochs, batchSize, lr)

    trainAcc = _evaluate(probe, trainX, trainY)
    evalAcc = _evaluate(probe, data[f"PROBE_{evalSplit}_X"], data[f"PROBE_{evalSplit}_Y"])

    return {
        "evalAcc": evalAcc,
        "trainAcc": trainAcc,
        # accuracy the probe gains on images it was trained on: its memorization
        "gap": trainAcc - evalAcc,
        "nTrain": trainX.size(0),
        "regularization": regularization,
    }


# linear probe for the grid: the L2 penalty is chosen on probe val, and the chosen probe is scored on probe test
def trainSelectedLinear(data):
    trainX, trainY = data["PROBE_TRAIN_X"], data["PROBE_TRAIN_Y"]
    best = None
    for l2 in LINEAR_L2S:
        probe = _trainLinear(trainX, trainY, l2)
        valAcc = _evaluate(probe, data["PROBE_VAL_X"], data["PROBE_VAL_Y"])
        # strict comparison, so ties go to the stronger penalty
        if(best is None or valAcc > best[0]):
            best = (valAcc, l2, probe)

    valAcc, l2, probe = best
    trainAcc = _evaluate(probe, trainX, trainY)
    testAcc = _evaluate(probe, data["PROBE_TEST_X"], data["PROBE_TEST_Y"])
    return {"testAcc": testAcc, "valAcc": valAcc, "trainAcc": trainAcc, "gap": trainAcc - testAcc,
            "l2": l2, "nTrain": trainX.size(0)}


def _probeName(hiddenWidth):
    return "linear" if hiddenWidth is None else f"mlp{hiddenWidth}"


HEADER = f"{'Site':<6} {'Probe':<8} {'reg':>6} {'trainAcc':>9} {'valAcc':>8} {'gap':>8}"


def _printRow(site, hiddenWidth, r):
    print(f"{site:<6} {_probeName(hiddenWidth):<8} {r['regularization']:>6g} {r['trainAcc']:>9.4f} {r['evalAcc']:>8.4f} {r['gap']:>8.4f}")


# every probe and regularization value on probe val (probe test stays untouched for the grid),
# then each probe at its best regularization, where the gap column shows how much it memorizes
def compareProbes(features, attribute, sites, widths, device):
    nTrain = None
    best = []
    print(f"\n{attribute}, t=100: all settings")
    print(HEADER)
    for site in sites:
        data = makeProbeData(features, attribute, site, device)
        for hiddenWidth in [None] + widths:
            results = []
            for regularization in (LINEAR_L2S if hiddenWidth is None else WEIGHT_DECAYS):
                results.append(trainProbe(data, hiddenWidth, regularization, evalSplit="VAL"))
                _printRow(site, hiddenWidth, results[-1])
            best.append((site, hiddenWidth, max(results, key=lambda r: r["evalAcc"])))
            nTrain = results[-1]["nTrain"]

    print(f"\n{attribute}, t=100: each probe at its best regularization ({nTrain} training images)")
    print(HEADER)
    for site, hiddenWidth, result in best:
        _printRow(site, hiddenWidth, result)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    features = loadFeatures(100)
    compareProbes(features, "Smiling", ["mid", "up1", "up2"], PROBE_WIDTHS, device)
    # few balanced images at 512-dimensional sites is where memorization is most likely
    compareProbes(features, "Eyeglasses", ["mid", "up1"], PROBE_WIDTHS, device)
