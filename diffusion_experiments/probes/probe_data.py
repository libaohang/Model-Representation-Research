import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import numpy as np
import torch

from NoiseSchedule import cosineSchedule

from probes.extract_features import (FEATURE_DIR, RANDOM_FEATURE_DIR, RANDOM_SEED1_FEATURE_DIR,
                                     PIXEL_FEATURE_PATH, SPLITS, T)

ATTRIBUTES = ["Male", "Young", "Blond_Hair", "Bangs", "Smiling", "Mouth_Slightly_Open", "Eyeglasses", "Wearing_Hat"]

PROBE_SPLITS = ["TRAIN", "VAL", "TEST"]
MIN_IMAGES_PER_IDENTITY = 4
HELD_OUT_FRACTION = 0.25

# feature sources stored as one file per timestep; "pixels" is built in loadFeatures instead
FEATURE_DIRS = {
    "trained": FEATURE_DIR,
    "random": RANDOM_FEATURE_DIR,
    "random_seed1": RANDOM_SEED1_FEATURE_DIR,
}


def loadLabels():
    attrs, identities, attrNames = [], [], None
    for split in SPLITS:
        labels = torch.load(FEATURE_DIR / split / "labels.pt")
        attrs.append(labels["attr"].numpy())
        identities.append(labels["identity"].numpy())
        attrNames = labels["attrNames"]
    return np.concatenate(attrs), np.concatenate(identities), attrNames


# rows are in pooled order: all of valid, then all of test, matching loadFeatures
ATTR, IDENTITY, ATTR_NAMES = loadLabels()


def splitByIdentity(seed):
    # splits each person's images about 50/25/25, so the same people appear in every probe split
    rng = np.random.default_rng(seed)
    splitOf = np.full(len(IDENTITY), -1, dtype=np.int64)
    for person in np.unique(IDENTITY):
        rows = np.flatnonzero(IDENTITY == person)
        if(len(rows) < MIN_IMAGES_PER_IDENTITY):
            continue
        rows = rng.permutation(rows)
        numHeld = max(1, round(HELD_OUT_FRACTION * len(rows)))
        splitOf[rows[:numHeld]] = PROBE_SPLITS.index("VAL")
        splitOf[rows[numHeld:2 * numHeld]] = PROBE_SPLITS.index("TEST")
        splitOf[rows[2 * numHeld:]] = PROBE_SPLITS.index("TRAIN")
    return splitOf


def splitByPeople(seed):
    # splits the people about 50/25/25, so nobody appears in two probe splits;
    # the same small identities are dropped as in splitByIdentity, so both modes draw from the same images
    rng = np.random.default_rng(seed)
    splitOf = np.full(len(IDENTITY), -1, dtype=np.int64)
    people, counts = np.unique(IDENTITY, return_counts=True)
    people = rng.permutation(people[counts >= MIN_IMAGES_PER_IDENTITY])
    numHeld = round(HELD_OUT_FRACTION * len(people))
    groups = {"VAL": people[:numHeld], "TEST": people[numHeld:2 * numHeld], "TRAIN": people[2 * numHeld:]}
    for split, group in groups.items():
        splitOf[np.isin(IDENTITY, group)] = PROBE_SPLITS.index(split)
    return splitOf


SPLITTERS = {"images": splitByIdentity, "people": splitByPeople}

SPLIT_OF = splitByIdentity(seed=1)


def makeBalancedIndices(attribute, splitOf=SPLIT_OF, seed=1):
    column = ATTR_NAMES.index(attribute)
    rng = np.random.default_rng([seed, column])
    indices = {}
    for code, split in enumerate(PROBE_SPLITS):
        rows = np.flatnonzero(splitOf == code)
        positives = rows[ATTR[rows, column] == 1]
        negatives = rows[ATTR[rows, column] == 0]
        numEach = min(len(positives), len(negatives))
        positives = rng.choice(positives, numEach, replace=False)
        negatives = rng.choice(negatives, numEach, replace=False)
        indices[split] = np.sort(np.concatenate([positives, negatives]))
    return indices


def checkSplits(splitOf, balanced, mode):
    people = {split: set(IDENTITY[splitOf == code].tolist()) for code, split in enumerate(PROBE_SPLITS)}
    if(mode == "images"):
        for split in ["VAL", "TEST"]:
            assert people[split] <= people["TRAIN"], f"probe {split} has identities missing from probe train"
    else:
        for first, second in [("TRAIN", "VAL"), ("TRAIN", "TEST"), ("VAL", "TEST")]:
            assert not (people[first] & people[second]), f"probe {first} and probe {second} share identities"
    for indices in balanced.values():
        allRows = np.concatenate([indices[split] for split in PROBE_SPLITS])
        assert len(np.unique(allRows)) == len(allRows), "an image appears in two probe splits"


# mode "images" splits each person's images across the probe splits; mode "people" holds whole people out
def makeBalancedSets(mode, seed=1):
    splitOf = SPLITTERS[mode](seed)
    balanced = {attribute: makeBalancedIndices(attribute, splitOf) for attribute in ATTRIBUTES}
    checkSplits(splitOf, balanced, mode)
    return balanced


BALANCED = {attribute: makeBalancedIndices(attribute) for attribute in ATTRIBUTES}

checkSplits(SPLIT_OF, BALANCED, "images")


def loadFeatures(t, source="trained"):
    if(source == "pixels"):
        # a PCA projection is linear, so the projected noised image is rebuilt from the projected
        # clean image and the projected noise
        pixels = torch.load(PIXEL_FEATURE_PATH)
        sqrtA, sqrt1mA, *_ = cosineSchedule(T, device="cpu")
        z = sqrtA[t] * pixels["z0"] + sqrt1mA[t] * pixels["zEps"]
        return {"pix512": z, "pix64": z[:, :64]}

    perSplit = [torch.load(FEATURE_DIRS[source] / split / f"t{t:04d}.pt") for split in SPLITS]
    return {name: torch.cat([feats[name] for feats in perSplit]).float() for name in perSplit[0]}


def _standardize(splitX):
    mean = splitX["TRAIN"].mean(dim=0)
    std = splitX["TRAIN"].std(dim=0).clamp_min(1e-6)
    return {split: (x - mean) / std for split, x in splitX.items()}, mean, std


def _pcaDirections(trainX, width):
    # trainX is standardized, so it is already centered
    trainX = trainX.double()
    _, vectors = torch.linalg.eigh(trainX.T @ trainX)
    # eigh sorts eigenvalues ascending, so the leading directions are the last columns
    return vectors[:, -width:].flip(1).float()


# pcaWidth reduces the site to that many dimensions with PCA fitted on the probe-train rows,
# so sites of different channel counts can be compared at equal width
def makeProbeData(features, attribute, site, device="cpu", balanced=BALANCED, pcaWidth=None):
    column = ATTR_NAMES.index(attribute)
    indices = balanced[attribute]
    X = features[site]

    splitX, featureMean, featureStd = _standardize({split: X[indices[split]] for split in PROBE_SPLITS})
    if(pcaWidth is not None):
        directions = _pcaDirections(splitX["TRAIN"], pcaWidth)
        splitX, _, _ = _standardize({split: x @ directions for split, x in splitX.items()})

    data = {"featureMean": featureMean, "featureStd": featureStd}
    for split in PROBE_SPLITS:
        data[f"PROBE_{split}_X"] = splitX[split].to(device)
        data[f"PROBE_{split}_Y"] = torch.from_numpy(ATTR[indices[split], column]).float().to(device)
    return data


def printSummary():
    kept = SPLIT_OF >= 0
    print(f"pooled images: {len(IDENTITY)}, kept: {int(kept.sum())}, identities kept: {len(np.unique(IDENTITY[kept]))}")
    for code, split in enumerate(PROBE_SPLITS):
        print(f"probe {split.lower()}: {int((SPLIT_OF == code).sum())} images")

    header = f"{'Attribute':<22} {'posRate':>8} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    for attribute in ATTRIBUTES:
        column = ATTR_NAMES.index(attribute)
        indices = BALANCED[attribute]
        posRate = ATTR[kept, column].mean()
        print(f"{attribute:<22} {posRate:>8.3f} {len(indices['TRAIN']):>8} {len(indices['VAL']):>8} {len(indices['TEST']):>8}")


if __name__ == "__main__":
    printSummary()
