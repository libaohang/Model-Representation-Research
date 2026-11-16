import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv

import torch

from probes.extract_features import TIMESTEPS, SITE_NAMES
from probes.make_control import ATTRIBUTES, loadFeatures, makeProbeData
from probes.control_task import trainProbes

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "probe_grid.csv"

# set both from the pilot in probes/control_task.py before running the grid
HIDDEN_WIDTH = None
WEIGHT_DECAY = 1e-2

COLUMNS = ["attribute", "site", "timestep", "normalAcc", "controlAcc", "selectivity",
           "normalMargin", "controlMargin", "trainAcc", "nTrain"]


def runGrid(device):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for t in TIMESTEPS:
            features = loadFeatures(t)
            for attribute in ATTRIBUTES:
                for site in SITE_NAMES:
                    data = makeProbeData(features, attribute, site, device)
                    result = trainProbes(data, HIDDEN_WIDTH, WEIGHT_DECAY, evalSplit="TEST")
                    row = {"attribute": attribute, "site": site, "timestep": t, **result}
                    writer.writerow(row)
                    rows.append(row)
                file.flush()
                best = max(rows[-len(SITE_NAMES):], key=lambda row: row["normalAcc"])
                print(f"t={t:<4} {attribute:<22} best site {best['site']:<6} normalAcc {best['normalAcc']:.4f} selectivity {best['selectivity']:.4f}")
    return rows


def printReferenceChecks(rows):
    noise = [row for row in rows if row["timestep"] == 999]
    print(f"t=999 (expect about 0.5): normalAcc {min(r['normalAcc'] for r in noise):.4f} to {max(r['normalAcc'] for r in noise):.4f}, "
          f"controlAcc {min(r['controlAcc'] for r in noise):.4f} to {max(r['controlAcc'] for r in noise):.4f}")

    header = f"{'Attribute':<10} {'Site':<6} {'normalAcc':>10} {'controlAcc':>11} {'selectivity':>12}"
    print("t=0 (expect well above chance)")
    print(header)
    for row in rows:
        if(row["timestep"] == 0 and row["attribute"] in ["Male", "Smiling"] and (row["site"] == "mid" or row["site"].startswith("up"))):
            print(f"{row['attribute']:<10} {row['site']:<6} {row['normalAcc']:>10.4f} {row['controlAcc']:>11.4f} {row['selectivity']:>12.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rows = runGrid(device)
    printReferenceChecks(rows)
