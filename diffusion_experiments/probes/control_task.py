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


def makeProbe(inputDim, hiddenWidth=None):
    if(hiddenWidth is None):
        return nn.Linear(inputDim, 1)
    return nn.Sequential(nn.Linear(inputDim, hiddenWidth), nn.GELU(), nn.Linear(hiddenWidth, 1))


def _trainOne(probe, X, Y, weightDecay, maxEpochs, batchSize, lr, tolerance):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weightDecay)
    n = X.size(0)
    previousLoss = None

    for epoch in range(maxEpochs):
        indices = torch.randperm(n, device=X.device)
        totalLoss = torch.zeros((), device=X.device)
        for start in range(0, n, batchSize):
            batch = indices[start:start + batchSize]
            loss = F.binary_cross_entropy_with_logits(probe(X[batch]).squeeze(1), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.detach() * batch.numel()

        epochLoss = totalLoss.item() / n
        if(previousLoss is not None and abs(previousLoss - epochLoss) < tolerance):
            break
        previousLoss = epochLoss

    return epoch + 1


@torch.no_grad()
def _evaluate(probe, X, Y):
    logits = probe(X).squeeze(1)
    accuracy = ((logits > 0).float() == Y).float().mean().item()
    # logit signed positive when it favors the true class
    margin = ((2 * Y - 1) * logits).mean().item()
    return accuracy, margin


def trainProbes(data, hiddenWidth=None, weightDecay=1e-2, evalSplit="TEST", maxEpochs=50, batchSize=256, lr=1e-3, tolerance=0.005, seed=0):
    trainX = data["PROBE_TRAIN_X"]
    evalX = data[f"PROBE_{evalSplit}_X"]
    device = trainX.device

    torch.manual_seed(seed)
    normalProbe = makeProbe(trainX.size(1), hiddenWidth).to(device)
    controlProbe = makeProbe(trainX.size(1), hiddenWidth).to(device)

    normalEpochs = _trainOne(normalProbe, trainX, data["PROBE_TRAIN_Y"], weightDecay, maxEpochs, batchSize, lr, tolerance)
    controlEpochs = _trainOne(controlProbe, trainX, data["PROBE_TRAIN_Y_CONTROL"], weightDecay, maxEpochs, batchSize, lr, tolerance)

    normalAcc, normalMargin = _evaluate(normalProbe, evalX, data[f"PROBE_{evalSplit}_Y"])
    controlAcc, controlMargin = _evaluate(controlProbe, evalX, data[f"PROBE_{evalSplit}_Y_CONTROL"])
    trainAcc, _ = _evaluate(normalProbe, trainX, data["PROBE_TRAIN_Y"])

    return {
        "normalAcc": normalAcc,
        "controlAcc": controlAcc,
        "selectivity": normalAcc - controlAcc,
        "normalMargin": normalMargin,
        "controlMargin": controlMargin,
        "trainAcc": trainAcc,
        "nTrain": trainX.size(0),
        "normalEpochs": normalEpochs,
        "controlEpochs": controlEpochs,
    }


def _probeName(hiddenWidth):
    return "linear" if hiddenWidth is None else f"mlp{hiddenWidth}"


def compareProbes(features, device):
    # probe type and weight decay are chosen on probe val, so probe test stays untouched for the grid
    header = f"{'Site':<6} {'Probe':<8} {'decay':>6} {'normalAcc':>10} {'controlAcc':>11} {'selectivity':>12} {'trainAcc':>9} {'epochs':>7}"
    print("Smiling, t=100")
    print(header)
    for site in ["mid", "up1", "up2"]:
        data = makeProbeData(features, "Smiling", site, device)
        for hiddenWidth in [None] + PROBE_WIDTHS:
            for weightDecay in WEIGHT_DECAYS:
                r = trainProbes(data, hiddenWidth, weightDecay, evalSplit="VAL")
                print(f"{site:<6} {_probeName(hiddenWidth):<8} {weightDecay:>6g} {r['normalAcc']:>10.4f} {r['controlAcc']:>11.4f} {r['selectivity']:>12.4f} {r['trainAcc']:>9.4f} {r['normalEpochs']:>7}")


def checkRareAttribute(features, device):
    # few balanced images at 512-dimensional sites is where overfitting is most likely
    header = f"{'Site':<6} {'decay':>6} {'trainAcc':>9} {'valAcc':>8} {'gap':>8} {'nTrain':>7}"
    print("Eyeglasses, t=100, linear probe")
    print(header)
    for site in ["mid", "up1"]:
        data = makeProbeData(features, "Eyeglasses", site, device)
        for weightDecay in WEIGHT_DECAYS:
            r = trainProbes(data, None, weightDecay, evalSplit="VAL")
            print(f"{site:<6} {weightDecay:>6g} {r['trainAcc']:>9.4f} {r['normalAcc']:>8.4f} {r['trainAcc'] - r['normalAcc']:>8.4f} {r['nTrain']:>7}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    features = loadFeatures(100)
    compareProbes(features, device)
    checkRareAttribute(features, device)
