import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv

import torch

from probes.extract_features import TIMESTEPS, SITE_NAMES
from probes.probe_data import ATTRIBUTES, BALANCED, loadFeatures, makeProbeData
from probes.probe_training import trainSelectedLinear

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "probe_grid.csv"

# the grid uses the linear probe, chosen in the pilot in probes/probe_training.py;
# valAcc and l2 record the penalty selection, testAcc is the reported number
COLUMNS = ["attribute", "site", "timestep", "testAcc", "valAcc", "trainAcc", "gap", "l2", "nTrain"]


# the defaults are the trained-network grid; the baselines change the feature source and its sites,
# and every run writes the same columns so the files join on attribute, site and timestep
def runGrid(device, source="trained", sites=SITE_NAMES, outPath=RESULTS_PATH, timesteps=TIMESTEPS,
            balanced=BALANCED, pcaWidth=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(outPath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for t in timesteps:
            features = loadFeatures(t, source)
            for attribute in ATTRIBUTES:
                for site in sites:
                    data = makeProbeData(features, attribute, site, device, balanced, pcaWidth)
                    row = {"attribute": attribute, "site": site, "timestep": t, **trainSelectedLinear(data)}
                    writer.writerow(row)
                    rows.append(row)
                file.flush()
                best = max(rows[-len(sites):], key=lambda row: row["testAcc"])
                print(f"t={t:<4} {attribute:<22} best site {best['site']:<6} testAcc {best['testAcc']:.4f} gap {best['gap']:.4f}")
    return rows


def printReferenceChecks(rows):
    noise = [row for row in rows if row["timestep"] == 999]
    print(f"t=999 (expect about 0.5): testAcc {min(r['testAcc'] for r in noise):.4f} to {max(r['testAcc'] for r in noise):.4f}")

    header = f"{'Attribute':<10} {'Site':<6} {'testAcc':>8} {'trainAcc':>9} {'gap':>8}"
    print("t=0 (expect well above chance)")
    print(header)
    for row in rows:
        if(row["timestep"] == 0 and row["attribute"] in ["Male", "Smiling"] and (row["site"] == "mid" or row["site"].startswith("up"))):
            print(f"{row['attribute']:<10} {row['site']:<6} {row['testAcc']:>8.4f} {row['trainAcc']:>9.4f} {row['gap']:>8.4f}")

    gaps = sorted(row["gap"] for row in rows)
    print(f"train minus test accuracy over all cells: median {gaps[len(gaps) // 2]:.4f}, largest {gaps[-1]:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rows = runGrid(device)
    printReferenceChecks(rows)
