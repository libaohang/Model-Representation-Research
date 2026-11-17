import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from experiments.analyze_grid import (RESULTS_DIR, RESULTS_PATH, PLOTS_DIR, SURFACE, AXIS, SERIES_COLORS, DIVERGING,
                                      loadResults, logSNR, drawHeatmap, emergencePoints)

RANDOM_PATH = RESULTS_DIR / "probe_grid_random.csv"
RANDOM_SEED1_PATH = RESULTS_DIR / "probe_grid_random_seed1.csv"
PIXELS_PATH = RESULTS_DIR / "probe_grid_pixels.csv"
WIDTH64_PATH = RESULTS_DIR / "probe_width64.csv"
HELDOUT_PATH = RESULTS_DIR / "probe_heldout.csv"

GAIN_PATH = RESULTS_DIR / "gain.csv"
EMERGENCE_PATH = RESULTS_DIR / "emergence_pixels.csv"

COMPARE_SITE = "up1"
PIXEL_SITE = "pix512"
CHECK_TIMESTEP = 100

# color follows the feature source; the second random seed shares its source's color and differs by line style
SOURCE_STYLES = {
    "trained": {"color": SERIES_COLORS[0], "marker": "o", "linestyle": "-", "markeredgecolor": SURFACE},
    "random weights": {"color": SERIES_COLORS[1], "marker": "s", "linestyle": "-", "markeredgecolor": SURFACE},
    "random weights, seed 1": {"color": SERIES_COLORS[1], "marker": "s", "linestyle": "--", "markerfacecolor": SURFACE},
    "noised pixels (pix512)": {"color": SERIES_COLORS[2], "marker": "^", "linestyle": "-", "markeredgecolor": SURFACE},
}


def computeGains(attributes, trained, random, pixelCurve):
    gains = {}
    for attribute in attributes:
        # the pixel baseline has one value per timestep, compared against every site
        overRandom = trained[attribute] - random[attribute]
        overPixels = trained[attribute] - pixelCurve[attribute][:, None]
        gains[attribute] = {"gainRandom": overRandom, "gainPixels": overPixels, "gain": np.minimum(overRandom, overPixels)}
    return gains


def saveGains(attributes, sites, timesteps, trained, random, pixelCurve, gains):
    with open(GAIN_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["attribute", "site", "timestep", "trainedAcc", "randomAcc", "pixelAcc", "gainRandom", "gainPixels", "gain"])
        for attribute in attributes:
            for i, t in enumerate(timesteps):
                for j, site in enumerate(sites):
                    writer.writerow([attribute, site, t, trained[attribute][i, j], random[attribute][i, j], pixelCurve[attribute][i],
                                     gains[attribute]["gainRandom"][i, j], gains[attribute]["gainPixels"][i, j], gains[attribute]["gain"][i, j]])


def plotGainHeatmaps(attributes, sites, timesteps, gains):
    # one symmetric scale for every panel and attribute, so the figures can be compared directly
    largest = max(np.abs(values).max() for gain in gains.values() for values in gain.values())
    limit = math.ceil(largest * 20) / 20

    panels = [("gainRandom", "Trained − random weights"), ("gainPixels", "Trained − noised pixels"), ("gain", "Gain (smaller of the two)")]
    for attribute in attributes:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, layout="constrained")
        for ax, (key, title) in zip(axes, panels):
            image = drawHeatmap(ax, gains[attribute][key], DIVERGING, -limit, limit, sites, timesteps)
            ax.set_title(title, loc="left", fontsize=11)
        axes[0].set_ylabel("Timestep (0 = clean, 999 = noise)")
        colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
        colorbar.outline.set_visible(False)
        colorbar.set_label("Accuracy difference")
        fig.suptitle(f"{attribute}: probe accuracy gained over the baselines", x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.savefig(PLOTS_DIR / f"gain_{attribute}.png", dpi=150)
        plt.close(fig)


def _plotSourceLines(ax, x, series):
    for label, values in series.items():
        ax.plot(x, values, linewidth=2, markersize=6, markeredgewidth=1, label=label, **SOURCE_STYLES[label])
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle="--")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)


def plotBaselineCurves(attributes, sites, timesteps, trained, random, pixelCurve):
    column = sites.index(COMPARE_SITE)
    # t = 0 has infinite log-SNR, so it is left off this axis and reported in the t = 0 table
    noisy = [i for i, t in enumerate(timesteps) if t > 0]
    snr = logSNR([timesteps[i] for i in noisy])

    for attribute in attributes:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        _plotSourceLines(ax, snr, {
            "trained": trained[attribute][noisy, column],
            "random weights": random[attribute][noisy, column],
            "noised pixels (pix512)": pixelCurve[attribute][noisy],
        })
        ax.set_title(f"Probe accuracy at {COMPARE_SITE} against the two baselines", loc="left", fontsize=11)
        ax.set_xlabel("log-SNR (higher = cleaner image)")
        ax.set_ylabel("Probe accuracy")
        ax.legend(frameon=False, loc="upper left")
        fig.suptitle(attribute, x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"baselines_{attribute}.png", dpi=150)
        plt.close(fig)


def plotSiteProfile(attributes, sites, timesteps, trained, random, seed1, seed1Timesteps):
    row, seed1Row = timesteps.index(CHECK_TIMESTEP), seed1Timesteps.index(CHECK_TIMESTEP)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharey=True)
    for ax, attribute in zip(axes.flat, attributes):
        _plotSourceLines(ax, range(len(sites)), {
            "trained": trained[attribute][row],
            "random weights": random[attribute][row],
            "random weights, seed 1": seed1[attribute][seed1Row],
        })
        ax.set_xticks(range(len(sites)), sites, rotation=45)
        ax.set_title(attribute, loc="left", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Probe accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("Site (network order)")
    fig.suptitle(f"Accuracy by site at t = {CHECK_TIMESTEP}: trained network against random weights", x=0.01, ha="left", fontsize=14, fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(PLOTS_DIR / "site_profile.png", dpi=150)
    plt.close(fig)


def _noisiestReaching(values, timesteps, threshold):
    reached = [t for t, value in zip(timesteps, values) if value >= threshold]
    return max(reached) if reached else "none"


def saveEmergence(attributes, sites, timesteps, trained, pixelCurve):
    trainedPoints = emergencePoints(attributes, sites, timesteps, trained)
    # the same rule on the pixel baseline alone: its threshold comes from its own best accuracy
    pixelPoints = emergencePoints(attributes, [PIXEL_SITE], timesteps, {a: pixelCurve[a][:, None] for a in attributes})

    points = []
    for attribute, t, p in zip(attributes, trainedPoints, pixelPoints):
        points.append({
            "attribute": attribute,
            "trainedBestAcc": t["bestAcc"],
            "trainedThreshold": t["threshold"],
            "trainedEmergenceTimestep": t["emergenceTimestep"],
            "pixelBestAcc": p["bestAcc"],
            "pixelThreshold": p["threshold"],
            "pixelEmergenceTimestep": p["emergenceTimestep"],
            # noisiest timestep at which the pixels are as decodable as the trained threshold asks
            "pixelReachesTrainedThreshold": _noisiestReaching(pixelCurve[attribute], timesteps, t["threshold"]),
        })

    with open(EMERGENCE_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    print("\nEmergence timestep: trained network against noised pixels")
    print(f"{'Attribute':<22} {'trainBest':>10} {'trainThr':>9} {'trainT':>7} {'pixBest':>8} {'pixThr':>7} {'pixT':>6} {'pixAtTrainThr':>14}")
    for p in points:
        print(f"{p['attribute']:<22} {p['trainedBestAcc']:>10.4f} {p['trainedThreshold']:>9.4f} {p['trainedEmergenceTimestep']:>7} "
              f"{p['pixelBestAcc']:>8.4f} {p['pixelThreshold']:>7.4f} {p['pixelEmergenceTimestep']:>6} {str(p['pixelReachesTrainedThreshold']):>14}")


def printCleanImageTable(attributes, sites, timesteps, trained, random, pixelCurve):
    column, clean, next_ = sites.index(COMPARE_SITE), timesteps.index(0), timesteps.index(50)
    print(f"\nt = 0 at {COMPARE_SITE}: accuracy on the clean image, and the dip (accuracy at t=50 minus t=0)")
    print(f"{'Attribute':<22} {'trained':>8} {'trainDip':>9} {'random':>8} {'randDip':>8} {'pix512':>8}")
    for attribute in attributes:
        t, r = trained[attribute][:, column], random[attribute][:, column]
        print(f"{attribute:<22} {t[clean]:>8.4f} {t[next_] - t[clean]:>9.4f} {r[clean]:>8.4f} {r[next_] - r[clean]:>8.4f} {pixelCurve[attribute][clean]:>8.4f}")


def printSeedVariation(attributes, timesteps, random, seed1, seed1Timesteps):
    print("\nRandom weights: accuracy difference between seed 0 and seed 1, over all attributes and sites")
    print(f"{'Timestep':<10} {'mean':>8} {'largest':>8}")
    for i, t in enumerate(seed1Timesteps):
        differences = np.abs(np.stack([random[a][timesteps.index(t)] - seed1[a][i] for a in attributes]))
        print(f"{t:<10} {differences.mean():>8.4f} {differences.max():>8.4f}")


def _printMatrix(title, attributes, columns, valueOf):
    print(f"\n{title}")
    print(f"{'Attribute':<22}" + "".join(f"{column:>8}" for column in columns))
    for attribute in attributes:
        print(f"{attribute:<22}" + "".join(f"{valueOf(attribute, column):>8.4f}" for column in columns))


def printWidthCheck(attributes, sites, timesteps, trained):
    _, wideSites, _, width64 = loadResults(WIDTH64_PATH)
    row = timesteps.index(CHECK_TIMESTEP)

    # sites that are already 64 wide keep their main-grid accuracy
    def atWidth64(attribute, site):
        if(site in wideSites):
            return width64[attribute][0, wideSites.index(site)]
        return trained[attribute][row, sites.index(site)]

    _printMatrix(f"Width check, t = {CHECK_TIMESTEP}: accuracy with every site at 64 dimensions", attributes, sites, atWidth64)
    _printMatrix("Width check: 64-dimension accuracy minus full-width accuracy", attributes, wideSites,
                 lambda a, site: atWidth64(a, site) - trained[a][row, sites.index(site)])


def printHeldOutCheck(attributes, sites, timesteps, trained):
    _, heldSites, _, heldOut = loadResults(HELDOUT_PATH)
    row = timesteps.index(CHECK_TIMESTEP)

    _printMatrix(f"Held-out people, t = {CHECK_TIMESTEP}: accuracy when no test person is in probe train", attributes, heldSites,
                 lambda a, site: heldOut[a][0, heldSites.index(site)])
    _printMatrix("Held-out people: held-out accuracy minus main-grid accuracy", attributes, heldSites,
                 lambda a, site: heldOut[a][0, heldSites.index(site)] - trained[a][row, sites.index(site)])


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    attributes, sites, timesteps, trained = loadResults(RESULTS_PATH)
    _, randomSites, randomTimesteps, random = loadResults(RANDOM_PATH)
    _, _, seed1Timesteps, seed1 = loadResults(RANDOM_SEED1_PATH)
    _, pixelSites, pixelTimesteps, pixels = loadResults(PIXELS_PATH)
    assert randomSites == sites and randomTimesteps == timesteps and pixelTimesteps == timesteps, "the baseline grids do not line up with the trained grid"
    pixelCurve = {a: pixels[a][:, pixelSites.index(PIXEL_SITE)] for a in attributes}

    gains = computeGains(attributes, trained, random, pixelCurve)
    saveGains(attributes, sites, timesteps, trained, random, pixelCurve, gains)
    plotGainHeatmaps(attributes, sites, timesteps, gains)
    plotBaselineCurves(attributes, sites, timesteps, trained, random, pixelCurve)
    plotSiteProfile(attributes, sites, timesteps, trained, random, seed1, seed1Timesteps)

    saveEmergence(attributes, sites, timesteps, trained, pixelCurve)
    printCleanImageTable(attributes, sites, timesteps, trained, random, pixelCurve)
    printSeedVariation(attributes, timesteps, random, seed1, seed1Timesteps)
    printWidthCheck(attributes, sites, timesteps, trained)
    printHeldOutCheck(attributes, sites, timesteps, trained)
