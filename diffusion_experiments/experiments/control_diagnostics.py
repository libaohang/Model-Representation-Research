import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from probes.extract_features import SITE_NAMES
from probes.make_control import (ATTRIBUTES, BALANCED, CONTROL_Y, IDENTITY, PROBE_SPLITS, SPLIT_OF,
                                 loadFeatures, makeProbeData)
from probes.control_task import makeProbe, trainProbes, _evaluate
from experiments.analyze_grid import AXIS, SECONDARY_INK, SERIES_COLORS, SURFACE

GRID_PATH = ROOT / "results" / "probe_grid.csv"
OUT_DIR = ROOT / "results" / "control_diagnostics"

MAIN_CELL = ("Smiling", "up1")
RARE_CELL = ("Wearing_Hat", "up1")
LOW_NOISE_T = 100
HIGH_NOISE_T = 900

HISTORY_EPOCHS = 200
IDENTITY_EPOCHS = 100
MLP_WIDTH = 256
CONVERGED_L2 = 1e-4

# (attribute, site, timestep): two strong cells, the bottleneck, and two weak real-task cells
CONVERGED_CELLS = [
    ("Smiling", "up1", LOW_NOISE_T),
    ("Wearing_Hat", "up1", LOW_NOISE_T),
    ("Smiling", "mid", LOW_NOISE_T),
    ("Smiling", "down0", HIGH_NOISE_T),
    ("Eyeglasses", "down0", HIGH_NOISE_T),
]

IDENTITY_COUNTS = [100, 300, 1000, None]
IDENTITY_SITES = ["up1", "up2", "up4"]


def saveRows(name, rows):
    with open(OUT_DIR / name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def loadGrid():
    with open(GRID_PATH, newline="") as file:
        return {(row["attribute"], row["site"], int(row["timestep"])): row for row in csv.DictReader(file)}


# same minibatch AdamW loop as control_task._trainOne, but it never stops early and records every epoch
def trainWithHistory(probe, X, Y, evalX, evalY, epochs, weightDecay, batchSize=256, lr=1e-3, tolerance=0.005, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weightDecay)
    n = X.size(0)
    history = []
    previousLoss, stopEpoch = None, None

    for epoch in range(epochs):
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
        # the epoch at which the grid's rule would have ended training
        if(stopEpoch is None and previousLoss is not None and abs(previousLoss - epochLoss) < tolerance):
            stopEpoch = epoch + 1
        previousLoss = epochLoss

        trainAcc, _ = _evaluate(probe, X, Y)
        testAcc, testMargin = _evaluate(probe, evalX, evalY)
        history.append({"epoch": epoch + 1, "trainLoss": epochLoss, "trainAcc": trainAcc, "testAcc": testAcc, "testMargin": testMargin})

    return history, stopEpoch


# full-batch logistic regression from zero weights, run to convergence
def fitConverged(X, Y, l2=CONVERGED_L2):
    X, Y = X.double(), Y.double()
    probe = makeProbe(X.size(1)).to(X.device).double()
    torch.nn.init.zeros_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=2000, history_size=20, tolerance_grad=1e-9,
                                  tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(X).squeeze(1), Y) + 0.5 * l2 * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


def evaluateConverged(probe, X, Y):
    return _evaluate(probe, X.double(), Y.double())


def checkEpochCounts(features, grid, device):
    print("\n=== 1. Epochs before the grid's stopping rule fired (all cells at t=100) ===")
    rows = []
    for attribute in ATTRIBUTES:
        for site in SITE_NAMES:
            data = makeProbeData(features, attribute, site, device)
            result = trainProbes(data, None, 1e-2, evalSplit="TEST")
            gridControlAcc = float(grid[(attribute, site, LOW_NOISE_T)]["controlAcc"])
            rows.append({"attribute": attribute, "site": site, "normalEpochs": result["normalEpochs"],
                         "controlEpochs": result["controlEpochs"], "controlAcc": result["controlAcc"],
                         "gridControlAcc": gridControlAcc})
    saveRows("epoch_counts.csv", rows)

    maxDiff = max(abs(row["controlAcc"] - row["gridControlAcc"]) for row in rows)
    print(f"largest difference from the grid's controlAcc: {maxDiff:.4f}")
    assert maxDiff < 0.01, "the diagnostic does not reproduce the grid's control probes"

    for key in ["normalEpochs", "controlEpochs"]:
        counts = Counter(row[key] for row in rows)
        print(f"{key:<14} " + "  ".join(f"{epochs} epochs: {count} cells" for epochs, count in sorted(counts.items())))

    header = f"{'Attribute':<22} {'steps/epoch':>12} {'normalEpochs':>13} {'controlEpochs':>14}"
    print(header)
    for attribute in ATTRIBUTES:
        mine = [row for row in rows if row["attribute"] == attribute]
        stepsPerEpoch = int(np.ceil(len(BALANCED[attribute]["TRAIN"]) / 256))
        print(f"{attribute:<22} {stepsPerEpoch:>12} {np.mean([r['normalEpochs'] for r in mine]):>13.1f} {np.mean([r['controlEpochs'] for r in mine]):>14.1f}")


def checkLossCurves(features, device):
    print(f"\n=== 2 and 4. Training for {HISTORY_EPOCHS} epochs with no stopping (t=100) ===")
    runs = [
        ("Smiling control, linear", MAIN_CELL, "_CONTROL", None, 1e-2),
        ("Wearing_Hat control, linear", RARE_CELL, "_CONTROL", None, 1e-2),
        (f"Smiling control, MLP-{MLP_WIDTH}", MAIN_CELL, "_CONTROL", MLP_WIDTH, 0.0),
        ("Smiling real, linear", MAIN_CELL, "", None, 1e-2),
    ]
    histories, rows = {}, []
    for label, (attribute, site), suffix, hiddenWidth, weightDecay in runs:
        data = makeProbeData(features, attribute, site, device)
        torch.manual_seed(0)
        probe = makeProbe(data["PROBE_TRAIN_X"].size(1), hiddenWidth).to(device)
        history, stopEpoch = trainWithHistory(probe, data["PROBE_TRAIN_X"], data[f"PROBE_TRAIN_Y{suffix}"],
                                              data["PROBE_TEST_X"], data[f"PROBE_TEST_Y{suffix}"], HISTORY_EPOCHS, weightDecay)
        histories[label] = (history, stopEpoch)
        rows += [{"run": label, **entry} for entry in history]

        print(f"\n{label} at {site} (grid rule would stop at epoch {stopEpoch})")
        print(f"{'epoch':>6} {'trainLoss':>10} {'trainAcc':>9} {'testAcc':>8} {'testMargin':>11}")
        for entry in history:
            if(entry["epoch"] in [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]):
                print(f"{entry['epoch']:>6} {entry['trainLoss']:>10.4f} {entry['trainAcc']:>9.4f} {entry['testAcc']:>8.4f} {entry['testMargin']:>11.4f}")
    saveRows("histories.csv", rows)
    plotHistories(histories)


def plotHistories(histories):
    panels = [("trainLoss", "Training loss"), ("trainAcc", "Training accuracy"), ("testAcc", "Test accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (key, title) in zip(axes, panels):
        for k, (label, (history, stopEpoch)) in enumerate(histories.items()):
            epochs = [entry["epoch"] for entry in history]
            values = [entry[key] for entry in history]
            ax.plot(epochs, values, color=SERIES_COLORS[k], linewidth=2, label=label)
            if(stopEpoch is not None):
                # where the grid's stopping rule ended training
                ax.plot(stopEpoch, values[stopEpoch - 1], marker="o", markersize=8, color=SERIES_COLORS[k],
                        markeredgecolor=SURFACE, markeredgewidth=2)
        chance = np.log(2) if key == "trainLoss" else 0.5
        ax.axhline(chance, color=AXIS, linewidth=1, linestyle="--")
        # the accuracy curves sit above their chance line and the loss curves below theirs, so the label goes on the empty side
        ax.text(HISTORY_EPOCHS, chance, "chance", ha="right", va="bottom" if key == "trainLoss" else "top",
                fontsize=9, color=SECONDARY_INK)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Epoch (log scale)")
        ax.grid(axis="y", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Probe training without early stopping, t = 100 (dot = where the grid's rule stopped)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)


def checkConvergedSolver(featuresByT, grid, device):
    print("\n=== 3. Grid's AdamW probes against a converged solver (full-batch L-BFGS, test split) ===")
    header = (f"{'Attribute':<12} {'Site':<6} {'t':>4} {'labels':<8} {'gridAcc':>8} {'convAcc':>8} {'gridMargin':>11} "
              f"{'convMargin':>11} {'convTrainAcc':>13}")
    print(header)
    rows = []
    for attribute, site, t in CONVERGED_CELLS:
        data = makeProbeData(featuresByT[t], attribute, site, device)
        gridRow = grid[(attribute, site, t)]
        for labels, suffix, prefix in [("real", "", "normal"), ("control", "_CONTROL", "control")]:
            probe = fitConverged(data["PROBE_TRAIN_X"], data[f"PROBE_TRAIN_Y{suffix}"])
            trainAcc, _ = evaluateConverged(probe, data["PROBE_TRAIN_X"], data[f"PROBE_TRAIN_Y{suffix}"])
            testAcc, testMargin = evaluateConverged(probe, data["PROBE_TEST_X"], data[f"PROBE_TEST_Y{suffix}"])
            row = {"attribute": attribute, "site": site, "timestep": t, "labels": labels,
                   "gridAcc": float(gridRow[f"{prefix}Acc"]), "convergedAcc": testAcc,
                   "gridMargin": float(gridRow[f"{prefix}Margin"]), "convergedMargin": testMargin,
                   "convergedTrainAcc": trainAcc}
            rows.append(row)
            print(f"{attribute:<12} {site:<6} {t:>4} {labels:<8} {row['gridAcc']:>8.4f} {testAcc:>8.4f} "
                  f"{row['gridMargin']:>11.4f} {testMargin:>11.4f} {trainAcc:>13.4f}")
    saveRows("converged.csv", rows)


# control set that does not depend on any attribute: every kept image of the K people with the most images
def makeIdentityData(features, site, numIdentities, device):
    kept = SPLIT_OF >= 0
    people, counts = np.unique(IDENTITY[kept], return_counts=True)
    if(numIdentities is not None):
        people = people[np.argsort(-counts, kind="stable")[:numIdentities]]
    chosen = kept & np.isin(IDENTITY, people)

    X = features[site]
    trainRows = np.flatnonzero(chosen & (SPLIT_OF == PROBE_SPLITS.index("TRAIN")))
    testRows = np.flatnonzero(chosen & (SPLIT_OF == PROBE_SPLITS.index("TEST")))
    featureMean = X[trainRows].mean(dim=0)
    featureStd = X[trainRows].std(dim=0).clamp_min(1e-6)

    def tensors(rows):
        return ((X[rows] - featureMean) / featureStd).to(device), torch.from_numpy(CONTROL_Y[rows]).float().to(device)

    return tensors(trainRows), tensors(testRows), len(people)


def checkIdentityCount(features, device):
    print("\n=== 5. Control task with fewer identities and all of their images (t=100, test split) ===")
    header = (f"{'Site':<5} {'dim':>4} {'people':>7} {'nTrain':>7} {'img/person':>11} {'nTest':>6} {'majority':>9} "
              f"{'linTrain':>9} {'linTest':>8} {'mlpTrain':>9} {'mlpTest':>8}")
    print(header)
    rows = []
    for site in IDENTITY_SITES:
        for numIdentities in IDENTITY_COUNTS:
            (trainX, trainY), (testX, testY), numPeople = makeIdentityData(features, site, numIdentities, device)
            # accuracy of always guessing the more common label, the floor for this subset
            majority = max(testY.mean().item(), 1 - testY.mean().item())

            linear = fitConverged(trainX, trainY)
            linearTrain, _ = evaluateConverged(linear, trainX, trainY)
            linearTest, _ = evaluateConverged(linear, testX, testY)

            torch.manual_seed(0)
            mlp = makeProbe(trainX.size(1), MLP_WIDTH).to(device)
            history, _ = trainWithHistory(mlp, trainX, trainY, testX, testY, IDENTITY_EPOCHS, 0.0)

            row = {"site": site, "dim": trainX.size(1), "people": numPeople, "nTrain": trainX.size(0),
                   "imagesPerPerson": trainX.size(0) / numPeople, "nTest": testX.size(0), "majority": majority,
                   "linearTrainAcc": linearTrain, "linearTestAcc": linearTest,
                   "mlpTrainAcc": history[-1]["trainAcc"], "mlpTestAcc": history[-1]["testAcc"]}
            rows.append(row)
            print(f"{site:<5} {row['dim']:>4} {numPeople:>7} {row['nTrain']:>7} {row['imagesPerPerson']:>11.1f} {row['nTest']:>6} "
                  f"{majority:>9.4f} {linearTrain:>9.4f} {linearTest:>8.4f} {row['mlpTrainAcc']:>9.4f} {row['mlpTestAcc']:>8.4f}")
    saveRows("identity_count.csv", rows)


def checkImagesPerIdentity():
    print("\n=== 6. Images per identity in each attribute's balanced training set ===")
    header = f"{'Attribute':<22} {'nTrain':>7} {'people':>7} {'img/person':>11} {'nTest':>6} {'testPeopleInTrain':>18}"
    print(header)
    rows = []
    for attribute in ATTRIBUTES:
        trainPeople = IDENTITY[BALANCED[attribute]["TRAIN"]]
        testPeople = IDENTITY[BALANCED[attribute]["TEST"]]
        people = np.unique(trainPeople)
        row = {"attribute": attribute, "nTrain": len(trainPeople), "people": len(people),
               "imagesPerPerson": len(trainPeople) / len(people), "nTest": len(testPeople),
               "testPeopleInTrain": float(np.isin(testPeople, people).mean())}
        rows.append(row)
        print(f"{attribute:<22} {row['nTrain']:>7} {row['people']:>7} {row['imagesPerPerson']:>11.2f} {row['nTest']:>6} {row['testPeopleInTrain']:>18.3f}")
    saveRows("images_per_identity.csv", rows)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = loadGrid()

    features = loadFeatures(LOW_NOISE_T)
    checkEpochCounts(features, grid, device)
    checkLossCurves(features, device)
    checkIdentityCount(features, device)
    checkImagesPerIdentity()

    featuresByT = {LOW_NOISE_T: features, HIGH_NOISE_T: loadFeatures(HIGH_NOISE_T)}
    checkConvergedSolver(featuresByT, grid, device)
