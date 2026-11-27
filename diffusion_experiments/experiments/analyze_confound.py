import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from experiments.analyze_grid import RESULTS_DIR, SURFACE, AXIS, SERIES_COLORS, DIVERGING, logSNR, drawHeatmap

GRID_PATH = RESULTS_DIR / "confound_grid.csv"
PIXEL_PATH = RESULTS_DIR / "confound_grid_pixels.csv"
OUT_DIR = RESULTS_DIR / "confound"
LAYER_TABLE_PATH = OUT_DIR / "layer_table.csv"
SUMMARY_PATH = OUT_DIR / "summary.csv"

COMPARE_SITE = "up1"
PIXEL_SITE = "pix512"
# t = 0 is skipped as the clean reference because of the dip at t = 0 in experiment 1
CLEAN_TIMESTEP = 50
# heavy noise where every probe is still well above chance, to see which attribute fades first
NOISY_TIMESTEP = 700
SEPARATED = 0.05
# a site only counts for "best site" if its aligned accuracy is within this of the best site at that timestep,
# so a near-chance site (where both groups are near 0.5 and the gap is trivially small) can't win
USABLE_MARGIN = 0.10
MODES = ["natural", "decorrelated"]
NUMERIC = ["alignedAcc", "conflictAcc", "gap", "alignedCI", "conflictCI", "confoundReliance", "testAcc", "valAcc",
           "trainAcc", "l2", "confoundValAcc", "confoundL2", "overlap"]

# color follows the training set, line style the test group
SET_STYLES = {
    "natural": {"color": SERIES_COLORS[0]},
    "decorrelated": {"color": SERIES_COLORS[1]},
    "pixels": {"color": SERIES_COLORS[2], "linewidth": 1.4},
}
GROUP_STYLES = {"aligned": {"linestyle": "-", "marker": "o"}, "conflict": {"linestyle": "--", "marker": "s", "markerfacecolor": SURFACE}}

TRANSFORMER_ROW = {"pair": "transformer (attractor mismatch)", "naturalSeparatedSite": "layer 3"}

# from looking at conflict_<target>.png; a pair noted here should not be read as a finding about the model
LABEL_NOTES = {
    "Arched_Eyebrows": "conflict-cell labels visibly noisy (makeup worn but Heavy_Makeup=0, arched brows labeled 0)",
}


def loadRows(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["timestep"] = int(row["timestep"])
        for key in NUMERIC:
            row[key] = float(row[key])
    return rows


def pairName(pair):
    target, confound, subset = pair
    return f"{target} vs {confound}" + (" (women)" if subset == "women" else "")


class Grid:
    # one value per (timestep, site) for every pair and training set
    def __init__(self, rows):
        self.rows = rows
        self.pairs = list(dict.fromkeys((r["target"], r["confound"], r["subset"]) for r in rows))
        self.sites = list(dict.fromkeys(r["site"] for r in rows))
        self.timesteps = sorted({r["timestep"] for r in rows})
        self.index = {(r["target"], r["mode"], r["site"], r["timestep"]): r for r in rows}

    def row(self, target, mode, site, t):
        return self.index[(target, mode, site, t)]

    def matrix(self, target, mode, key):
        return np.array([[self.row(target, mode, s, t)[key] for s in self.sites] for t in self.timesteps])

    def curve(self, target, mode, key, site, timesteps):
        return np.array([self.row(target, mode, site, t)[key] for t in timesteps])


def _aboveChance(row):
    # a gap only means something where the probe reads the target at all; at chance both groups are near 0.5
    return row["alignedAcc"] - 0.5 > row["alignedCI"]


def gapCI(row):
    # aligned and conflict are separate image groups, so their intervals add in quadrature
    return math.hypot(row["alignedCI"], row["conflictCI"])


def usableSites(grid, target, mode, t):
    rows = [grid.row(target, mode, s, t) for s in grid.sites]
    best = max(r["alignedAcc"] for r in rows)
    return [r for r in rows if _aboveChance(r) and r["alignedAcc"] >= best - USABLE_MARGIN]


def bestSite(grid, target, mode):
    # the usable site at CLEAN_TIMESTEP whose gap is closest to zero
    return min(usableSites(grid, target, mode, CLEAN_TIMESTEP), key=lambda r: abs(r["gap"]))


def firstSeparatedSite(grid, target, mode):
    # first usable site, in network order, with the gap below SEPARATED
    for row in usableSites(grid, target, mode, CLEAN_TIMESTEP):
        if(abs(row["gap"]) < SEPARATED):
            return row["site"]
    return "none"


def lowestConflict(grid, target, mode):
    # usable sites only, and not pure noise, where one of ~22 chance-level cells passes _aboveChance by luck
    rows = [r for t in grid.timesteps if t < max(grid.timesteps) for r in usableSites(grid, target, mode, t)]
    worst = min(rows, key=lambda r: r["conflictAcc"])
    return worst["conflictAcc"], f"{worst['site']} t={worst['timestep']}"


def plotGapHeatmaps(grid):
    largest = max(np.abs(grid.matrix(p[0], mode, "gap")).max() for p in grid.pairs for mode in MODES)
    limit = math.ceil(largest * 20) / 20
    for pair in grid.pairs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, layout="constrained")
        for ax, mode in zip(axes, MODES):
            image = drawHeatmap(ax, grid.matrix(pair[0], mode, "gap"), DIVERGING, -limit, limit, grid.sites, grid.timesteps)
            ax.set_title(f"{mode.capitalize()} training set", loc="left", fontsize=11)
        axes[0].set_ylabel("Timestep (0 = clean, 999 = noise)")
        bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
        bar.outline.set_visible(False)
        bar.set_label("Aligned minus conflict accuracy")
        fig.suptitle(f"{pairName(pair)}: accuracy gap", x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.savefig(OUT_DIR / f"gap_{pair[0]}.png", dpi=150)
        plt.close(fig)


def plotAccuracyCurves(grid, pixels):
    noisy = [t for t in grid.timesteps if t > 0]
    snr = logSNR(noisy)
    for pair in grid.pairs:
        fig, ax = plt.subplots(figsize=(8.5, 5.8), layout="constrained")
        lines = [(mode, grid, COMPARE_SITE, mode) for mode in MODES] + [("pixels", pixels, PIXEL_SITE, "natural")]
        for label, source, site, mode in lines:
            for group in ("aligned", "conflict"):
                values = source.curve(pair[0], mode, f"{group}Acc", site, noisy)
                ax.plot(snr, values, markersize=5, linewidth=SET_STYLES[label].get("linewidth", 2),
                        color=SET_STYLES[label]["color"], label=f"{label}, {group}", **GROUP_STYLES[group])
                if(group == "conflict"):
                    ci = source.curve(pair[0], mode, "conflictCI", site, noisy)
                    ax.fill_between(snr, values - ci, values + ci, color=SET_STYLES[label]["color"], alpha=0.12, linewidth=0)
        ax.axhline(0.5, color=AXIS, linewidth=1, linestyle=":")
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("log-SNR (higher = cleaner image)")
        ax.set_ylabel("Probe accuracy")
        ax.grid(axis="y", linewidth=0.8)
        ax.set_axisbelow(True)
        # below the plot, where no curve can sit under it
        fig.legend(frameon=False, fontsize=9, ncol=3, loc="outside lower center")
        ax.set_title(f"{COMPARE_SITE} (network) and {PIXEL_SITE} (pixels); shading = 95% interval on conflict",
                     loc="left", fontsize=10)
        fig.suptitle(pairName(pair), x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.savefig(OUT_DIR / f"accuracy_{pair[0]}.png", dpi=150)
        plt.close(fig)


def plotOverlap(grid):
    noisy = [t for t in grid.timesteps if t > 0]
    snr = logSNR(noisy)
    for pair in grid.pairs:
        fig, (bySite, byNoise) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        for mode in MODES:
            style = {"color": SET_STYLES[mode]["color"], "marker": "o", "linewidth": 2, "markersize": 5, "label": mode}
            bySite.plot(range(len(grid.sites)), [grid.row(pair[0], mode, s, CLEAN_TIMESTEP)["overlap"] for s in grid.sites], **style)
            byNoise.plot(snr, grid.curve(pair[0], mode, "overlap", COMPARE_SITE, noisy), **style)
        bySite.set_xticks(range(len(grid.sites)), grid.sites, rotation=45)
        bySite.set_xlabel("Site (network order)")
        bySite.set_title(f"By site, t = {CLEAN_TIMESTEP}", loc="left", fontsize=11)
        byNoise.set_xlabel("log-SNR (higher = cleaner image)")
        byNoise.set_title(f"By noise level, {COMPARE_SITE}", loc="left", fontsize=11)
        bySite.set_ylabel("Cosine of target and confound probe weights")
        for ax in (bySite, byNoise):
            ax.axhline(0, color=AXIS, linewidth=1)
            ax.grid(axis="y", linewidth=0.8)
            ax.set_axisbelow(True)
        byNoise.legend(frameon=False)
        fig.suptitle(f"{pairName(pair)}: direction overlap", x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"overlap_{pair[0]}.png", dpi=150)
        plt.close(fig)


def saveLayerTable(grid):
    rows = []
    for pair in grid.pairs:
        print(f"\n{pairName(pair)}, t = {CLEAN_TIMESTEP}")
        print(f"{'Site':<6} " + " ".join(f"{mode[:3] + ' ' + k:>12}" for mode in MODES for k in ("aligned", "conflict", "gap")))
        for site in grid.sites:
            values = {mode: grid.row(pair[0], mode, site, CLEAN_TIMESTEP) for mode in MODES}
            print(f"{site:<6} " + " ".join(f"{values[mode][k]:>12.4f}" for mode in MODES for k in ("alignedAcc", "conflictAcc", "gap")))
            for mode in MODES:
                r = values[mode]
                rows.append({"target": pair[0], "confound": pair[1], "subset": pair[2], "mode": mode, "site": site,
                             "alignedAcc": r["alignedAcc"], "conflictAcc": r["conflictAcc"], "gap": r["gap"],
                             "conflictCI": r["conflictCI"], "nConflict": r["nConflict"]})
    with open(LAYER_TABLE_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def saveSummary(grid, pixels):
    # gap = aligned minus conflict accuracy. Natural gap = decorrelated gap (the floor: conflict images that are
    # harder or mislabeled, which no training set removes) + the shortcut part (what decorrelating training removes)
    rows = []
    for pair in grid.pairs:
        target = pair[0]
        natural = grid.row(target, "natural", COMPARE_SITE, CLEAN_TIMESTEP)
        decorrelated = grid.row(target, "decorrelated", COMPARE_SITE, CLEAN_TIMESTEP)
        best = bestSite(grid, target, "decorrelated")
        lowest, where = lowestConflict(grid, target, "natural")
        rows.append({
            "pair": pairName(pair),
            "naturalAligned": round(natural["alignedAcc"], 4),
            "decorrelatedAligned": round(decorrelated["alignedAcc"], 4),
            "naturalGap": round(natural["gap"], 4),
            "decorrelatedGap": round(decorrelated["gap"], 4),
            "shortcutPart": round(natural["gap"] - decorrelated["gap"], 4),
            "gapCI": round(max(gapCI(natural), gapCI(decorrelated)), 4),
            "pixelGap": round(pixels.row(target, "natural", PIXEL_SITE, CLEAN_TIMESTEP)["gap"], 4),
            "naturalSeparatedSite": firstSeparatedSite(grid, target, "natural"),
            "decorrelatedSeparatedSite": firstSeparatedSite(grid, target, "decorrelated"),
            "decorrelatedBestSite": best["site"],
            "decorrelatedBestGap": round(best["gap"], 4),
            "decorrelatedBestAligned": round(best["alignedAcc"], 4),
            "naturalGapNoisy": round(grid.row(target, "natural", COMPARE_SITE, NOISY_TIMESTEP)["gap"], 4),
            "decorrelatedGapNoisy": round(grid.row(target, "decorrelated", COMPARE_SITE, NOISY_TIMESTEP)["gap"], 4),
            "lowestConflictAcc": round(lowest, 4),
            "lowestConflictWhere": where,
            "labelNote": LABEL_NOTES.get(target, ""),
        })
    rows.append({key: TRANSFORMER_ROW.get(key, "") for key in rows[0]})

    with open(SUMMARY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary at {COMPARE_SITE}, t = {CLEAN_TIMESTEP} (gaps are aligned minus conflict accuracy; "
          f"shortcut = natural gap minus decorrelated gap)")
    print(f"{'Pair':<42} {'natGap':>7} {'decGap':>7} {'shortct':>7} {'±CI':>6} {'pixGap':>7} "
          f"{'t=' + str(NOISY_TIMESTEP) + ' nat/dec':>15}")
    for r in rows[:-1]:
        print(f"{r['pair']:<42} {r['naturalGap']:>+7.3f} {r['decorrelatedGap']:>+7.3f} {r['shortcutPart']:>+7.3f} "
              f"{r['gapCI']:>6.3f} {r['pixelGap']:>+7.3f} {r['naturalGapNoisy']:>+7.3f}/{r['decorrelatedGapNoisy']:+.3f}")

    print(f"\nSites at t = {CLEAN_TIMESTEP}, counting only sites within {USABLE_MARGIN:.2f} of the best aligned accuracy "
          f"(separated = |gap| below {SEPARATED:.2f})")
    print(f"{'Pair':<42} {'nat separated':>13} {'dec separated':>13} {'dec best site (gap, aligned)':>30}")
    for r in rows[:-1]:
        bestText = f"{r['decorrelatedBestSite']} ({r['decorrelatedBestGap']:+.3f}, {r['decorrelatedBestAligned']:.3f})"
        print(f"{r['pair']:<42} {r['naturalSeparatedSite']:>13} {r['decorrelatedSeparatedSite']:>13} {bestText:>30}")
    for r in rows[:-1]:
        if(r["labelNote"]):
            print(f"note, {r['pair']}: {r['labelNote']}")


def printChecks(grid):
    noise = [r for r in grid.rows if r["timestep"] == max(grid.timesteps)]
    for key in ["alignedAcc", "conflictAcc"]:
        values = [r[key] for r in noise]
        print(f"t={max(grid.timesteps)} {key} (expect about 0.5): {min(values):.4f} to {max(values):.4f}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, pixels = Grid(loadRows(GRID_PATH)), Grid(loadRows(PIXEL_PATH))
    printChecks(grid)
    plotGapHeatmaps(grid)
    plotAccuracyCurves(grid, pixels)
    plotOverlap(grid)
    saveLayerTable(grid)
    saveSummary(grid, pixels)
