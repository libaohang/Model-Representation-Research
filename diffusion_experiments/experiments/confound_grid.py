import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import csv
import math

import numpy as np
import torch

from probes.extract_features import TIMESTEPS, SITE_NAMES
from probes.probe_data import (ATTR, ATTR_NAMES, PROBE_SPLITS, WOMEN, PAIR_MODES, splitByPeople, checkSplits,
                               loadFeatures, makePairIndices, makePairProbeData)
from probes.probe_training import trainSelectedLinear, evaluate
from experiments.probe_grid import RESULTS_DIR

GRID_PATH = RESULTS_DIR / "confound_grid.csv"
PIXEL_PATH = RESULTS_DIR / "confound_grid_pixels.csv"
PAIRS_PATH = RESULTS_DIR / "confound_pairs.csv"
HELDOUT_PATH = RESULTS_DIR / "probe_heldout.csv"

# held-out people: an attribute that is stable within a person (Male, Big_Nose) can't be read by recognizing them
SPLIT = splitByPeople(seed=1)
SUBSETS = {"all": None, "women": WOMEN}

# chosen from attribute_correlation.py; confound_pairs.csv records the reasons. The makeup pairs are women only:
# among men their correlation is about zero, so over everyone it mostly runs through Male
PAIRS = [
    {"target": "Mouth_Slightly_Open", "confound": "Smiling", "subset": "all"},
    {"target": "Smiling", "confound": "High_Cheekbones", "subset": "all"},
    {"target": "Attractive", "confound": "Wearing_Lipstick", "subset": "women"},
    {"target": "Arched_Eyebrows", "confound": "Heavy_Makeup", "subset": "women"},
    {"target": "Big_Nose", "confound": "Male", "subset": "all"},
]

COLUMNS = ["target", "confound", "subset", "mode", "site", "timestep",
           "alignedAcc", "conflictAcc", "gap", "alignedCI", "conflictCI", "confoundReliance",
           "testAcc", "valAcc", "trainAcc", "l2", "nTrain", "nAligned", "nConflict",
           "confoundValAcc", "confoundL2", "overlap"]

CHECK_TIMESTEP = 100
CHECK_SITES = ["mid", "up1", "up2"]


def ci95(accuracy, n):
    # half-width of the normal-approximation 95% interval
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)


def buildIndices():
    return {(p["target"], mode): makePairIndices(p["target"], p["confound"], mode, SPLIT, SUBSETS[p["subset"]])
            for p in PAIRS for mode in PAIR_MODES}


def savePairTable(indices):
    rows = []
    for p in PAIRS:
        t, c = ATTR_NAMES.index(p["target"]), ATTR_NAMES.index(p["confound"])
        keep = np.ones(len(ATTR), dtype=bool) if SUBSETS[p["subset"]] is None else SUBSETS[p["subset"]]
        row = {"target": p["target"], "confound": p["confound"], "subset": p["subset"]}
        for code, split in enumerate(PROBE_SPLITS):
            r = (SPLIT == code) & keep
            for a in (1, 0):
                for b in (1, 0):
                    row[f"{split.lower()}_T{a}_C{b}"] = int(((ATTR[r, t] == a) & (ATTR[r, c] == b)).sum())
        # correlation and conditional rates on probe train only, as in attribute_correlation.py
        r = (SPLIT == PROBE_SPLITS.index("TRAIN")) & keep
        x, y = ATTR[r, t].astype(float), ATTR[r, c].astype(float)
        row.update({
            "phi": np.corrcoef(x, y)[0, 1],
            "P(T|C=1)": x[y == 1].mean(), "P(T|C=0)": x[y == 0].mean(),
            "P(C|T=1)": y[x == 1].mean(), "P(C|T=0)": y[x == 0].mean(),
        })
        for mode in PAIR_MODES:
            row[f"nTrain_{mode}"] = len(indices[(p["target"], mode)]["TRAIN"])
        row["nAligned"] = len(indices[(p["target"], "natural")]["TEST_ALIGNED"])
        row["nConflict"] = len(indices[(p["target"], "natural")]["TEST_CONFLICT"])
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Target':<20} {'Confound':<17} {'subset':<6} {'phi':>6} {'trainMin':>8} {'testMin':>8} {'aligned':>8} {'conflict':>8}")
    for row in rows:
        trainMin = min(v for k, v in row.items() if k.startswith("train_T"))
        testMin = min(v for k, v in row.items() if k.startswith("test_T"))
        print(f"{row['target']:<20} {row['confound']:<17} {row['subset']:<6} {row['phi']:>6.3f} {trainMin:>8} {testMin:>8} "
              f"{row['nAligned']:>8} {row['nConflict']:>8}")


def _view(data, label):
    # the shape trainSelectedLinear expects, for the target (label "Y") or the confound (label "C")
    return {f"PROBE_{split}_{axis}": data[f"PROBE_{group}_{label if axis == 'Y' else 'X'}"]
            for split, group in (("TRAIN", "TRAIN"), ("VAL", "VAL"), ("TEST", "TEST_ALL")) for axis in ("X", "Y")}


def scorePair(data, usualWithConfound):
    result, probe = trainSelectedLinear(_view(data, "Y"), returnProbe=True)
    aligned = evaluate(probe, data["PROBE_TEST_ALIGNED_X"], data["PROBE_TEST_ALIGNED_Y"])
    conflict = evaluate(probe, data["PROBE_TEST_CONFLICT_X"], data["PROBE_TEST_CONFLICT_Y"])
    nAligned, nConflict = data["PROBE_TEST_ALIGNED_Y"].numel(), data["PROBE_TEST_CONFLICT_Y"].numel()

    # the confound probe sees the same images, with its own L2 chosen on probe val; under the natural set
    # the confound labels are not balanced, so its accuracy is not on a 50% chance scale
    confoundResult, confoundProbe = trainSelectedLinear(_view(data, "C"), returnProbe=True)
    w, wc = probe.weight.flatten(), confoundProbe.weight.flatten()
    cosine = (w @ wc / (w.norm() * wc.norm()).clamp_min(1e-12)).item()

    return {
        "alignedAcc": aligned, "conflictAcc": conflict, "gap": aligned - conflict,
        "alignedCI": ci95(aligned, nAligned), "conflictCI": ci95(conflict, nConflict),
        # every conflict image the probe gets wrong is given the label the confound implies
        "confoundReliance": 1 - conflict,
        "testAcc": result["testAcc"], "valAcc": result["valAcc"], "trainAcc": result["trainAcc"],
        "l2": result["l2"], "nTrain": result["nTrain"], "nAligned": nAligned, "nConflict": nConflict,
        "confoundValAcc": confoundResult["valAcc"], "confoundL2": confoundResult["l2"],
        # signed so that positive means the two probes agree on aligned images
        "overlap": cosine if usualWithConfound else -cosine,
    }


def runConfoundGrid(device, indices, source="trained", sites=SITE_NAMES, modes=PAIR_MODES, outPath=GRID_PATH,
                    timesteps=TIMESTEPS):
    rows = []
    with open(outPath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for t in timesteps:
            features = loadFeatures(t, source)
            for p in PAIRS:
                for mode in modes:
                    pairIndices = indices[(p["target"], mode)]
                    for site in sites:
                        data = makePairProbeData(features, pairIndices, p["target"], p["confound"], site, device)
                        row = {**p, "mode": mode, "site": site, "timestep": t,
                               **scorePair(data, pairIndices["usualWithConfound"])}
                        writer.writerow(row)
                        rows.append(row)
                    file.flush()
                    showSite = "up1" if "up1" in sites else sites[0]
                    show = next(r for r in rows[-len(sites):] if r["site"] == showSite)
                    print(f"t={t:<4} {p['target']:<20} {mode:<12} {show['site']:<6} aligned {show['alignedAcc']:.4f} "
                          f"conflict {show['conflictAcc']:.4f} gap {show['gap']:+.4f} overlap {show['overlap']:+.3f}")
    return rows


def printChecks(rows):
    noise = [r for r in rows if r["timestep"] == 999]
    for key in ["alignedAcc", "conflictAcc"]:
        values = [r[key] for r in noise]
        print(f"t=999 {key} (expect about 0.5): {min(values):.4f} to {max(values):.4f}")

    # the natural set reuses the experiment 2 held-out-people images, so these should match exactly
    with open(HELDOUT_PATH, newline="") as file:
        heldOut = {(r["attribute"], r["site"]): float(r["testAcc"]) for r in csv.DictReader(file) if int(r["timestep"]) == CHECK_TIMESTEP}
    print(f"natural testAcc against the experiment 2 held-out-people grid, t = {CHECK_TIMESTEP}")
    for r in rows:
        key = (r["target"], r["site"])
        if(r["mode"] == "natural" and r["timestep"] == CHECK_TIMESTEP and r["subset"] == "all" and key in heldOut
           and r["site"] in CHECK_SITES):
            print(f"{r['target']:<20} {r['site']:<5} here {r['testAcc']:.4f}  experiment 2 {heldOut[key]:.4f}  "
                  f"difference {r['testAcc'] - heldOut[key]:+.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # no person in two probe splits, checked before anything is trained
    checkSplits(SPLIT, {}, "people")
    indices = buildIndices()
    savePairTable(indices)

    # the pixel run is short, so it comes first and shows the pipeline works before the long run
    runConfoundGrid(device, indices, source="pixels", sites=["pix512"], modes=["natural"], outPath=PIXEL_PATH)
    rows = runConfoundGrid(device, indices)
    printChecks(rows)
