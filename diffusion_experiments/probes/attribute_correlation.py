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
from matplotlib.colors import LinearSegmentedColormap, Normalize

RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PAIRS_PATH = RESULTS_DIR / "attribute_pairs.csv"

# counted on probe train, which holds about twice as many images as probe test,
# so 600 here leaves roughly 300 in the rarest test cell
MIN_CELL = 600
NUM_LABELED = 10
# only pairs at least this strong get a text label on the selection plot
LABEL_MIN_PHI = 0.15
# correlations beyond this get the darkest color; few CelebA pairs exceed it
COLOR_LIMIT = 0.6

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
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


def correlationMatrix(attr):
    # phi coefficient: the Pearson correlation of two 0/1 columns
    x = attr.astype(np.float64)
    x = (x - x.mean(axis=0)) / x.std(axis=0).clip(min=1e-12)
    return (x.T @ x) / len(x)


def cellCounts(attr):
    # counts[i, j, a, b] = number of images with attribute i == a and attribute j == b
    x = attr.astype(np.int64)
    both = x.T @ x
    ones = x.sum(axis=0)
    n = len(x)
    counts = np.empty((x.shape[1], x.shape[1], 2, 2), dtype=np.int64)
    counts[:, :, 1, 1] = both
    counts[:, :, 1, 0] = ones[:, None] - both
    counts[:, :, 0, 1] = ones[None, :] - both
    counts[:, :, 0, 0] = n - ones[:, None] - ones[None, :] + both
    return counts


def clusterOrder(phi):
    # attributes with strong correlations of either sign end up next to each other
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        distance = 1 - np.abs(phi)
        np.fill_diagonal(distance, 0)
        return leaves_list(linkage(squareform(distance, checks=False), method="average"))
    except ImportError:
        return np.arange(len(phi))


def plotCorrelogram(phi, counts, names, order):
    n = len(names)
    minCell = counts.min(axis=(2, 3))
    norm = Normalize(-COLOR_LIMIT, COLOR_LIMIT)

    rows, cols = np.tril_indices(n, k=-1)
    i, j = order[rows], order[cols]
    # circle area grows with the rarest cell, capped so common pairs don't swamp the plot
    sizes = 4 + 140 * np.sqrt(np.minimum(minCell[i, j], 4000) / 4000)

    usable = minCell[i, j] >= MIN_CELL
    fig, ax = plt.subplots(figsize=(13, 12))
    # pairs too rare to test are faded, so the eye goes to the usable ones
    ax.scatter(cols[~usable], rows[~usable], s=sizes[~usable], c=phi[i, j][~usable], cmap=DIVERGING, norm=norm,
               alpha=0.3, linewidths=0)
    points = ax.scatter(cols[usable], rows[usable], s=sizes[usable], c=phi[i, j][usable], cmap=DIVERGING, norm=norm,
                        linewidths=0)
    ax.set_xticks(range(n), [names[k] for k in order], rotation=90, fontsize=8)
    ax.set_yticks(range(n), [names[k] for k in order], fontsize=8)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)
    ax.set_aspect("equal")
    ax.grid(linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.02, extend="both")
    bar.outline.set_visible(False)
    bar.set_label("Correlation (phi)")
    fig.suptitle("Attribute correlations", x=0.01, ha="left", fontsize=14, fontweight="bold")
    ax.set_title(f"Circle size = images in the pair's rarest cell; faded = fewer than {MIN_CELL}",
                 loc="left", fontsize=10, color=SECONDARY_INK)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "attribute_correlogram.png", dpi=150)
    plt.close(fig)


def pairTable(phi, counts, names):
    rows = []
    for a in range(len(names)):
        for b in range(a):
            c = counts[a, b]
            rows.append({
                "attributeA": names[a],
                "attributeB": names[b],
                "phi": phi[a, b],
                "minCell": int(c.min()),
                "A1_B1": int(c[1, 1]), "A1_B0": int(c[1, 0]), "A0_B1": int(c[0, 1]), "A0_B0": int(c[0, 0]),
                # how much knowing B shifts the rate of A, and the other way round
                "P(A|B=1)": c[1, 1] / max(c[:, 1].sum(), 1),
                "P(A|B=0)": c[1, 0] / max(c[:, 0].sum(), 1),
                "P(B|A=1)": c[1, 1] / max(c[1, :].sum(), 1),
                "P(B|A=0)": c[0, 1] / max(c[0, :].sum(), 1),
            })
    return sorted(rows, key=lambda r: -abs(r["phi"]))


def plotSelection(pairs):
    strength = np.array([abs(p["phi"]) for p in pairs])
    minCell = np.array([p["minCell"] for p in pairs])
    usable = minCell >= MIN_CELL

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(minCell[~usable], strength[~usable], s=14, color=AXIS, label=f"rarest cell < {MIN_CELL}")
    ax.scatter(minCell[usable], strength[usable], s=18, color="#2a78d6", label=f"rarest cell ≥ {MIN_CELL}")
    ax.axvline(MIN_CELL, color=AXIS, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Images in the rarest of the four cells (log scale)")
    ax.set_ylabel("|Correlation (phi)|")
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)

    # label the strongest usable pairs: these are the candidates
    candidates = [p for p in pairs if p["minCell"] >= MIN_CELL and abs(p["phi"]) >= LABEL_MIN_PHI][:NUM_LABELED]
    # alternate the label side so neighboring labels overlap less; the CSV has the full list
    for k, p in enumerate(candidates):
        ax.annotate(f"{p['attributeA']} × {p['attributeB']}", (p["minCell"], abs(p["phi"])),
                    xytext=(6, 3) if k % 2 == 0 else (6, -10), textcoords="offset points",
                    fontsize=7.5, color=SECONDARY_INK)

    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle("Candidate pairs: strong correlation and enough images", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "attribute_pair_selection.png", dpi=150)
    plt.close(fig)


def savePairs(pairs):
    with open(PAIRS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Strongest pairs with at least {MIN_CELL} images in every cell:")
    print(f"{'Attribute A':<22} {'Attribute B':<22} {'phi':>6} {'minCell':>8} {'P(A|B=1)':>9} {'P(A|B=0)':>9}")
    for p in [p for p in pairs if p["minCell"] >= MIN_CELL][:NUM_LABELED]:
        print(f"{p['attributeA']:<22} {p['attributeB']:<22} {p['phi']:>6.3f} {p['minCell']:>8} "
              f"{p['P(A|B=1)']:>9.3f} {p['P(A|B=0)']:>9.3f}")


def analyze(attr, names):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    phi = correlationMatrix(attr)
    counts = cellCounts(attr)
    plotCorrelogram(phi, counts, names, clusterOrder(phi))
    pairs = pairTable(phi, counts, names)
    plotSelection(pairs)
    savePairs(pairs)


if __name__ == "__main__":
    from probes.probe_data import ATTR, ATTR_NAMES, SPLIT_OF, PROBE_SPLITS

    # probe-train rows only, so the test images play no part in choosing the pairs
    train = SPLIT_OF == PROBE_SPLITS.index("TRAIN")
    print(f"computing correlations on {int(train.sum())} probe-train images")
    analyze(ATTR[train], ATTR_NAMES)