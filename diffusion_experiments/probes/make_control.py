import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diffusion_model"))

import numpy as np
import torch

from probes.extract_features import FEATURE_DIR, SPLITS, TIMESTEPS, SITE_NAMES

ATTRIBUTES = ["Male", "Young", "Blond_Hair", "Bangs", "Smiling", "Mouth_Slightly_Open", "Eyeglasses", "Wearing_Hat"]

PROBE_SPLITS = ["TRAIN", "VAL", "TEST"]
MIN_IMAGES_PER_IDENTITY = 4
HELD_OUT_FRACTION = 0.25


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


def makeControlLabels(seed):
    # one random 0/1 label per identity, inherited by every image of that person
    rng = np.random.default_rng(seed)
    people = np.unique(IDENTITY)
    labels = rng.integers(0, 2, size=len(people))
    return labels[np.searchsorted(people, IDENTITY)]


SPLIT_OF = splitByIdentity(seed=1)
CONTROL_Y = makeControlLabels(seed=1)


def makeBalancedIndices(attribute, seed=1):
    column = ATTR_NAMES.index(attribute)
    rng = np.random.default_rng([seed, column])
    indices = {}
    for code, split in enumerate(PROBE_SPLITS):
        rows = np.flatnonzero(SPLIT_OF == code)
        positives = rows[ATTR[rows, column] == 1]
        negatives = rows[ATTR[rows, column] == 0]
        numEach = min(len(positives), len(negatives))
        positives = rng.choice(positives, numEach, replace=False)
        negatives = rng.choice(negatives, numEach, replace=False)
        indices[split] = np.sort(np.concatenate([positives, negatives]))
    return indices


BALANCED = {attribute: makeBalancedIndices(attribute) for attribute in ATTRIBUTES}


def checkSplits():
    trainPeople = set(IDENTITY[SPLIT_OF == PROBE_SPLITS.index("TRAIN")].tolist())
    for split in ["VAL", "TEST"]:
        people = set(IDENTITY[SPLIT_OF == PROBE_SPLITS.index(split)].tolist())
        assert people <= trainPeople, f"probe {split} has identities missing from probe train"
    for indices in BALANCED.values():
        allRows = np.concatenate([indices[split] for split in PROBE_SPLITS])
        assert len(np.unique(allRows)) == len(allRows), "an image appears in two probe splits"


checkSplits()


def loadFeatures(t):
    perSplit = [torch.load(FEATURE_DIR / split / f"t{t:04d}.pt") for split in SPLITS]
    return {name: torch.cat([feats[name] for feats in perSplit]).float() for name in SITE_NAMES}


def makeProbeData(features, attribute, site, device="cpu"):
    column = ATTR_NAMES.index(attribute)
    indices = BALANCED[attribute]
    X = features[site]

    trainX = X[indices["TRAIN"]]
    featureMean = trainX.mean(dim=0)
    featureStd = trainX.std(dim=0).clamp_min(1e-6)

    data = {"featureMean": featureMean, "featureStd": featureStd}
    for split in PROBE_SPLITS:
        rows = indices[split]
        data[f"PROBE_{split}_X"] = ((X[rows] - featureMean) / featureStd).to(device)
        data[f"PROBE_{split}_Y"] = torch.from_numpy(ATTR[rows, column]).float().to(device)
        data[f"PROBE_{split}_Y_CONTROL"] = torch.from_numpy(CONTROL_Y[rows]).float().to(device)
    return data


def printSummary():
    kept = SPLIT_OF >= 0
    print(f"pooled images: {len(IDENTITY)}, kept: {int(kept.sum())}, identities kept: {len(np.unique(IDENTITY[kept]))}")
    for code, split in enumerate(PROBE_SPLITS):
        print(f"probe {split.lower()}: {int((SPLIT_OF == code).sum())} images")

    header = f"{'Attribute':<22} {'posRate':>8} {'train':>8} {'val':>8} {'test':>8} {'ctrlPos':>8}"
    print(header)
    for attribute in ATTRIBUTES:
        column = ATTR_NAMES.index(attribute)
        indices = BALANCED[attribute]
        posRate = ATTR[kept, column].mean()
        controlRate = CONTROL_Y[indices["TRAIN"]].mean()
        print(f"{attribute:<22} {posRate:>8.3f} {len(indices['TRAIN']):>8} {len(indices['VAL']):>8} {len(indices['TEST']):>8} {controlRate:>8.3f}")


if __name__ == "__main__":
    printSummary()
