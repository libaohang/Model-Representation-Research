import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from probes.extract_features import DATA_ROOT, SPLITS
from probes.probe_data import ATTR, ATTR_NAMES
from experiments.confound_grid import PAIRS, RESULTS_DIR, buildIndices

OUT_DIR = RESULTS_DIR / "confound"
PER_CELL = 25
COLUMNS = 10


# label-noise check: conflict cells collect mislabeled images, so look at them before reading low conflict accuracy
def loadImages():
    # pooled rows are all of valid, then all of test, as in probe_data; the crop is the one the network saw, at full resolution
    crop = transforms.CenterCrop(178)
    return [datasets.CelebA(root=str(DATA_ROOT), split=split, target_type="attr", transform=crop, download=False)
            for split in SPLITS]


def imageAt(splitSets, row):
    first = len(splitSets[0])
    return splitSets[0][row][0] if row < first else splitSets[1][row - first][0]


def saveConflictGrid(splitSets, pair, conflictRows, seed=0):
    t, c = ATTR_NAMES.index(pair["target"]), ATTR_NAMES.index(pair["confound"])
    rng = np.random.default_rng([seed, t, c])
    # PER_CELL from each of the two conflict cells, one cell after the other
    shown = []
    for value in (1, 0):
        cell = conflictRows[ATTR[conflictRows, t] == value]
        shown += [(row, value, int(ATTR[row, c])) for row in rng.choice(cell, PER_CELL, replace=False)]

    rows = int(np.ceil(len(shown) / COLUMNS))
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(COLUMNS * 1.6, rows * 2.0))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (row, targetValue, confoundValue) in zip(axes.flat, shown):
        ax.imshow(imageAt(splitSets, row))
        ax.set_title(f"T={targetValue} C={confoundValue}  #{row}", fontsize=7)
    subset = " (women only)" if pair["subset"] == "women" else ""
    fig.suptitle(f"Conflict-cell test images{subset}: T = {pair['target']}, C = {pair['confound']}. "
                 f"Check how many labels look wrong.", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"conflict_{pair['target']}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splitSets = loadImages()
    indices = buildIndices()
    for pair in PAIRS:
        # the test groups are the same for both training sets
        saveConflictGrid(splitSets, pair, indices[(pair["target"], "natural")]["TEST_CONFLICT"])
        print(f"saved {OUT_DIR / ('conflict_' + pair['target'] + '.png')}")
