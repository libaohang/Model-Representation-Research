import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from probes.make_control import ATTRIBUTES, BALANCED, loadFeatures, makeProbeData
from probes.control_task import makeProbe
from probes.make_cluster_control import applyClusterLabels, describeClusters, fitClusters, loadPixels, makeClusterLabels
from experiments.control_diagnostics import evaluateConverged, fitConverged, trainWithHistory
from experiments.analyze_grid import AXIS, SECONDARY_INK, SERIES_COLORS, SERIES_MARKERS, SURFACE

OUT_DIR = ROOT / "results" / "cluster_control"

K_VALUES = [50, 100, 200, 500, 1000]
SITES = ["down0", "down3", "up1", "up2"]
PILOT_ATTRIBUTES = ["Smiling", "Wearing_Hat"]
LABEL_SEEDS = [0, 1, 2]
TIMESTEP = 100

MLP_WIDTH = 256
MLP_EPOCHS = 60

# second table: one k at one site across noise levels
NOISE_K = 200
NOISE_SITE = "up1"
NOISE_ATTRIBUTE = "Smiling"
NOISE_TIMESTEPS = [0, 100, 500, 900]


def saveRows(name, rows):
    with open(OUT_DIR / name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# trains the converged linear probe and the MLP on one label set, scored on probe val so probe test stays untouched
def fitBoth(data, suffix):
    trainX, trainY = data["PROBE_TRAIN_X"], data[f"PROBE_TRAIN_Y{suffix}"]
    valX, valY = data["PROBE_VAL_X"], data[f"PROBE_VAL_Y{suffix}"]

    linear = fitConverged(trainX, trainY)
    linearTrain, _ = evaluateConverged(linear, trainX, trainY)
    linearVal, _ = evaluateConverged(linear, valX, valY)

    torch.manual_seed(0)
    mlp = makeProbe(trainX.size(1), MLP_WIDTH).to(trainX.device)
    history, _ = trainWithHistory(mlp, trainX, trainY, valX, valY, MLP_EPOCHS, 0.0)

    return {
        "linearTrain": linearTrain,
        "linearVal": linearVal,
        "mlpTrain": history[-1]["trainAcc"],
        "mlpVal": history[-1]["testAcc"],
        "mlpBestVal": max(entry["testAcc"] for entry in history),
    }


# control results for one cell and one clustering, averaged over the random label seeds
def fitControl(data, attribute, clusterIds, k):
    runs = []
    floors = []
    for seed in LABEL_SEEDS:
        applyClusterLabels(data, attribute, makeClusterLabels(clusterIds, k, seed))
        runs.append(fitBoth(data, "_CONTROL"))
        # accuracy of always guessing the more common label
        positiveRate = data["PROBE_VAL_Y_CONTROL"].mean().item()
        floors.append(max(positiveRate, 1 - positiveRate))

    result = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
    for key in ["linearVal", "mlpVal"]:
        result[f"{key}Min"] = min(run[key] for run in runs)
        result[f"{key}Max"] = max(run[key] for run in runs)
    result["floor"] = float(np.mean(floors))
    return result


def coverage(attribute, clusterIds, k):
    trainRows, valRows = BALANCED[attribute]["TRAIN"], BALANCED[attribute]["VAL"]
    trainSizes = np.bincount(clusterIds[trainRows], minlength=k)
    imagesPerCluster = len(trainRows) / k
    # share of val images whose cluster has at least one training image in this attribute's balanced set
    seen = float((trainSizes[clusterIds[valRows]] > 0).mean())
    return imagesPerCluster, seen


def describeAllClusters(clusters):
    print("\n=== Clusters (sizes are probe-train images per cluster) ===")
    header = f"{'k':>5} {'min':>5} {'median':>7} {'max':>5} {'empty':>6} {'valSeen':>8} {'testSeen':>9}"
    print(header)
    rows = [describeClusters(clusters[k], k) for k in K_VALUES]
    for row in rows:
        print(f"{row['k']:>5} {row['minSize']:>5} {row['medianSize']:>7.0f} {row['maxSize']:>5} {row['emptyClusters']:>6} "
              f"{row['valSeen']:>8.3f} {row['testSeen']:>9.3f}")

    print("\nAttribute accuracy from the cluster id alone (balanced val, chance 0.5)")
    print(f"{'k':>5} " + " ".join(f"{attribute[:11]:>11}" for attribute in ATTRIBUTES))
    for row in rows:
        print(f"{row['k']:>5} " + " ".join(f"{row[attribute]:>11.3f}" for attribute in ATTRIBUTES))
    saveRows("clusters.csv", rows)


def runPilot(features, clusters, device):
    print(f"\n=== Cluster control pilot, t={TIMESTEP}, probe val, mean over {len(LABEL_SEEDS)} label seeds ===")
    header = (f"{'Site':<6} {'k':>5} {'img/cl':>7} {'seen':>6} {'floor':>6} {'linTrain':>9} {'linVal':>7} {'(min-max)':>13} "
              f"{'mlpTrain':>9} {'mlpVal':>7} {'(min-max)':>13} {'mlpBest':>8} {'linSel':>7} {'mlpSel':>7}")
    rows = []
    for attribute in PILOT_ATTRIBUTES:
        print(f"\n{attribute}")
        for site in SITES:
            data = makeProbeData(features, attribute, site, device)
            real = fitBoth(data, "")
            print(f"{site}: real accuracy linear {real['linearVal']:.4f}, MLP-{MLP_WIDTH} {real['mlpVal']:.4f}")
            print(header)
            for k in K_VALUES:
                control = fitControl(data, attribute, clusters[k], k)
                imagesPerCluster, seen = coverage(attribute, clusters[k], k)
                row = {"attribute": attribute, "site": site, "k": k, "imagesPerCluster": imagesPerCluster, "seen": seen,
                       "realLinearVal": real["linearVal"], "realMlpVal": real["mlpVal"], **control,
                       "linearSelectivity": real["linearVal"] - control["linearVal"],
                       "mlpSelectivity": real["mlpVal"] - control["mlpVal"]}
                rows.append(row)
                linearRange = f"({row['linearValMin']:.3f}-{row['linearValMax']:.3f})"
                mlpRange = f"({row['mlpValMin']:.3f}-{row['mlpValMax']:.3f})"
                print(f"{site:<6} {k:>5} {imagesPerCluster:>7.1f} {seen:>6.3f} {row['floor']:>6.3f} {row['linearTrain']:>9.4f} "
                      f"{row['linearVal']:>7.4f} {linearRange:>13} {row['mlpTrain']:>9.4f} {row['mlpVal']:>7.4f} {mlpRange:>13} "
                      f"{row['mlpBestVal']:>8.4f} {row['linearSelectivity']:>7.4f} {row['mlpSelectivity']:>7.4f}")
    saveRows("pilot.csv", rows)
    return rows


def runNoiseCheck(featuresAtPilotT, clusters, device):
    print(f"\n=== Control accuracy by noise level: {NOISE_ATTRIBUTE}, {NOISE_SITE}, k={NOISE_K}, probe val ===")
    header = f"{'t':>4} {'realLin':>8} {'realMlp':>8} {'ctrlLin':>8} {'ctrlMlp':>8} {'linSel':>7} {'mlpSel':>7}"
    print(header)
    rows = []
    for t in NOISE_TIMESTEPS:
        features = featuresAtPilotT if t == TIMESTEP else loadFeatures(t)
        data = makeProbeData(features, NOISE_ATTRIBUTE, NOISE_SITE, device)
        real = fitBoth(data, "")
        control = fitControl(data, NOISE_ATTRIBUTE, clusters[NOISE_K], NOISE_K)
        row = {"timestep": t, "realLinearVal": real["linearVal"], "realMlpVal": real["mlpVal"],
               "controlLinearVal": control["linearVal"], "controlMlpVal": control["mlpVal"],
               "linearSelectivity": real["linearVal"] - control["linearVal"],
               "mlpSelectivity": real["mlpVal"] - control["mlpVal"]}
        rows.append(row)
        print(f"{t:>4} {row['realLinearVal']:>8.4f} {row['realMlpVal']:>8.4f} {row['controlLinearVal']:>8.4f} "
              f"{row['controlMlpVal']:>8.4f} {row['linearSelectivity']:>7.4f} {row['mlpSelectivity']:>7.4f}")
    saveRows("noise_check.csv", rows)


def plotControlByK(rows):
    attribute = PILOT_ATTRIBUTES[0]
    series = [("linear", "linearVal", "Linear probe"), ("mlp", "mlpVal", f"MLP-{MLP_WIDTH} probe")]
    fig, axes = plt.subplots(1, len(SITES), figsize=(4.2 * len(SITES), 4.6), sharey=True)
    for ax, site in zip(axes, SITES):
        mine = [row for row in rows if row["attribute"] == attribute and row["site"] == site]
        ks = [row["k"] for row in mine]
        for index, (_, key, label) in enumerate(series):
            # band spans the random label seeds
            ax.fill_between(ks, [row[f"{key}Min"] for row in mine], [row[f"{key}Max"] for row in mine],
                            color=SERIES_COLORS[index], alpha=0.18, linewidth=0)
            ax.plot(ks, [row[key] for row in mine], color=SERIES_COLORS[index], marker=SERIES_MARKERS[index],
                    linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--")
        ax.text(ks[-1], 0.5, "chance", ha="right", va="top", fontsize=9, color=SECONDARY_INK)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_ylim(0.45, 1.0)
        ax.set_title(f"{site} ({mine[0]['realLinearVal']:.2f} real accuracy, linear)", loc="left", fontsize=11)
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(axis="y", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Control accuracy on probe val")
    fig.suptitle(f"Cluster control accuracy by number of clusters ({attribute} image set, t = {TIMESTEP})",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_DIR / "control_by_k.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pixels = loadPixels()
    print(f"pixels: {tuple(pixels.shape)}")
    clusters = {k: fitClusters(pixels, k, device=device) for k in K_VALUES}
    describeAllClusters(clusters)

    features = loadFeatures(TIMESTEP)
    rows = runPilot(features, clusters, device)
    plotControlByK(rows)
    runNoiseCheck(features, clusters, device)
