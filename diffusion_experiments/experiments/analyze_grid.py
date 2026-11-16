import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from NoiseSchedule import cosineSchedule

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "probe_grid.csv"
EMERGENCE_PATH = RESULTS_DIR / "emergence.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

SLICE_TIMESTEPS = [100, 300, 500, 700]
EMERGENCE_SITE_TIMESTEP = 100

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed categorical order: color follows the attribute, markers back it up
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

SEQUENTIAL = LinearSegmentedColormap.from_list("sequentialBlue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("divergingBlueRed", ["#1c5cab", "#86b6ef", "#f0efec", "#ec8a8a", "#b02a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": AXIS,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "grid.color": GRIDLINE,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def loadResults():
    with open(RESULTS_PATH, newline="") as file:
        rows = list(csv.DictReader(file))

    # first-appearance order keeps sites in network order and timesteps ascending
    attributes = list(dict.fromkeys(row["attribute"] for row in rows))
    sites = list(dict.fromkeys(row["site"] for row in rows))
    timesteps = sorted({int(row["timestep"]) for row in rows})

    accuracy = {a: np.full((len(timesteps), len(sites)), np.nan) for a in attributes}
    selectivity = {a: np.full((len(timesteps), len(sites)), np.nan) for a in attributes}
    for row in rows:
        i, j = timesteps.index(int(row["timestep"])), sites.index(row["site"])
        accuracy[row["attribute"]][i, j] = float(row["normalAcc"])
        selectivity[row["attribute"]][i, j] = float(row["selectivity"])
    return attributes, sites, timesteps, accuracy, selectivity


def logSNR(timesteps):
    alphaBars = cosineSchedule(1000, device="cpu")[-1].double().numpy()[timesteps]
    with np.errstate(divide="ignore"):
        return np.log(alphaBars / (1 - alphaBars))


def plotHeatmaps(attributes, sites, timesteps, accuracy, selectivity):
    for attribute in attributes:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        panels = [
            (accuracy[attribute], SEQUENTIAL, 0.5, 1.0, "Probe accuracy"),
            (selectivity[attribute], DIVERGING, -0.5, 0.5, "Selectivity (real − control accuracy)"),
        ]
        for ax, (values, cmap, low, high, title) in zip(axes, panels):
            image = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xticks(range(len(sites)), sites)
            ax.set_yticks(range(len(timesteps)), timesteps)
            ax.set_xlabel("Site (network order)")
            # white gaps separate the cells
            ax.set_xticks(np.arange(-0.5, len(sites)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(timesteps)), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)
        axes[0].set_ylabel("Timestep (0 = clean, 999 = noise)")
        fig.suptitle(attribute, x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"heatmap_{attribute}.png", dpi=150)
        plt.close(fig)


def _plotAttributeLines(ax, attributes, x, valuesOf):
    for k, attribute in enumerate(attributes):
        ax.plot(x, valuesOf(attribute), color=SERIES_COLORS[k], marker=SERIES_MARKERS[k],
                linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, label=attribute)
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)


def _addLegend(fig, ax, columns):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=columns, frameon=False)


def plotTimestepSlices(attributes, sites, timesteps, accuracy):
    present = [t for t in SLICE_TIMESTEPS if t in timesteps]
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.8), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], present):
        row = timesteps.index(t)
        _plotAttributeLines(ax, attributes, range(len(sites)), lambda a: accuracy[a][row])
        ax.set_xticks(range(len(sites)), sites, rotation=45)
        ax.set_title(f"t = {t}", loc="left", fontsize=11)
        ax.set_xlabel("Site (network order)")
    axes[0][0].set_ylabel("Probe accuracy")
    fig.suptitle("Accuracy by site at fixed timesteps", x=0.01, ha="left", fontsize=14, fontweight="bold")
    _addLegend(fig, axes[0][0], len(attributes))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOTS_DIR / "slices_by_timestep.png", dpi=150)
    plt.close(fig)


def plotSiteSlices(attributes, sites, timesteps, accuracy):
    upSites = [site for site in sites if site.startswith("up")]
    meanAccuracy = np.mean([accuracy[a] for a in attributes], axis=0).mean(axis=0)
    bestUp = max(upSites, key=lambda site: meanAccuracy[sites.index(site)])

    # t = 0 has infinite log-SNR, so it is left off this axis
    noisy = [i for i, t in enumerate(timesteps) if t > 0]
    snr = logSNR([timesteps[i] for i in noisy])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, site in zip(axes, ["mid", bestUp]):
        column = sites.index(site)
        _plotAttributeLines(ax, attributes, snr, lambda a: accuracy[a][noisy, column])
        ax.set_title(site, loc="left", fontsize=11)
        ax.set_xlabel("log-SNR (higher = cleaner image)")
    axes[0].set_ylabel("Probe accuracy")
    fig.suptitle("Accuracy by noise level at fixed sites", x=0.01, ha="left", fontsize=14, fontweight="bold")
    _addLegend(fig, axes[0], 4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(PLOTS_DIR / "slices_by_site.png", dpi=150)
    plt.close(fig)


def emergencePoints(attributes, sites, timesteps, accuracy):
    snr = logSNR(timesteps)
    siteRow = timesteps.index(EMERGENCE_SITE_TIMESTEP)
    points = []
    for attribute in attributes:
        values = accuracy[attribute]
        bestAcc = float(np.nanmax(values))
        threshold = 0.5 + 0.9 * (bestAcc - 0.5)
        bestRow, bestColumn = np.unravel_index(np.nanargmax(values), values.shape)

        # noisiest timestep at which the best site still reaches the threshold
        reached = [i for i in range(len(timesteps)) if values[i, bestColumn] >= threshold]
        emergenceRow = max(reached)
        # first site in network order that reaches the threshold at a low-noise timestep
        reachedSites = [j for j in range(len(sites)) if values[siteRow, j] >= threshold]

        points.append({
            "attribute": attribute,
            "bestAcc": bestAcc,
            "bestSite": sites[bestColumn],
            "bestTimestep": timesteps[bestRow],
            "threshold": threshold,
            "emergenceTimestep": timesteps[emergenceRow],
            "emergenceLogSNR": float(snr[emergenceRow]),
            "emergenceSite": sites[reachedSites[0]] if reachedSites else "none",
        })
    return points


def saveEmergence(points):
    with open(EMERGENCE_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    header = f"{'Attribute':<22} {'bestAcc':>8} {'bestSite':>9} {'bestT':>6} {'threshold':>10} {'emergeT':>8} {'logSNR':>8} {'emergeSite':>11}"
    print(header)
    for p in points:
        print(f"{p['attribute']:<22} {p['bestAcc']:>8.4f} {p['bestSite']:>9} {p['bestTimestep']:>6} {p['threshold']:>10.4f} "
              f"{p['emergenceTimestep']:>8} {p['emergenceLogSNR']:>8.2f} {p['emergenceSite']:>11}")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    attributes, sites, timesteps, accuracy, selectivity = loadResults()
    plotHeatmaps(attributes, sites, timesteps, accuracy, selectivity)
    plotTimestepSlices(attributes, sites, timesteps, accuracy)
    plotSiteSlices(attributes, sites, timesteps, accuracy)
    saveEmergence(emergencePoints(attributes, sites, timesteps, accuracy))
