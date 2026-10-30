import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from dataset.sample import load_npz
from probes.make_control import (
    VOCAB,
    wordIdx,
    NUMBER_TO_LABEL,
    makeHiddenStates,
    gatherPreVerbState,
    model,
)
from dataset import generate


trainCube = load_npz(ROOT / "data/valGPT_trainProbe.npz")
valCube = load_npz(ROOT / "data/valProbe.npz")


# Same as probes.make_control.buildDataset but also tracks last_attractor_mismatch
# per row so val rows can be split into "mismatch" (True) vs "rest" (False/None).
def buildDataset(cube):
    tokens, subjectPosition, verbPosition, subjectNumber = [], [], [], []
    subjectTokenId, lastAttractorMismatch = [], []
    for idx in np.argwhere(cube["cell_valid"]):
        cell = tuple(idx)
        for k in range(len(cube["tokens"][cell])):
            slot = cell + (k,)
            sentence = cube["tokens"][slot]
            assert 14 - len(sentence) >= 0
            padding = ["<pad>"] * (14 - len(sentence))
            padded = ["<bos>"] + sentence + ["<eos>"] + padding
            tokens.append([wordIdx[word] for word in padded])

            subjPos = int(cube["subject_position"][slot]) + 1
            verbPos = int(cube["verb_position"][slot]) + 1
            subjectPosition.append(subjPos)
            verbPosition.append(verbPos)
            subjectNumber.append(NUMBER_TO_LABEL[cube["subject_number"][slot]])
            subjectTokenId.append(wordIdx[padded[subjPos]])
            lastAttractorMismatch.append(cube["last_attractor_mismatch"][slot] is True)

    data = {
        "tokens": np.array(tokens),
        "subjectPosition": np.array(subjectPosition),
        "verbPosition": np.array(verbPosition),
        "subjectNumber": np.array(subjectNumber),
        "subjectTokenId": np.array(subjectTokenId),
        "lastAttractorMismatch": np.array(lastAttractorMismatch, dtype=bool),
    }
    rows = np.arange(len(data["tokens"]))
    assert np.all(data["tokens"][rows, data["subjectPosition"]] == data["subjectTokenId"])
    for tokenId in np.unique(data["subjectTokenId"]):
        assert VOCAB[tokenId] in generate.NOUN_BY_FORM
    verbTokens = data["tokens"][rows, data["verbPosition"]]
    assert np.all(verbTokens >= 3), "verb position landed on a special token"
    return data


trainData = buildDataset(trainCube)
valData = buildDataset(valCube)

valMismatchMask = valData["lastAttractorMismatch"]
valRestMask = ~valMismatchMask

print(f"val split sizes: mismatch={int(valMismatchMask.sum())}  rest={int(valRestMask.sum())}")
assert valMismatchMask.sum() > 0 and valRestMask.sum() > 0


# Build the probe dataset at a chosen depth of the model. Trains on all rows,
# but returns the val set split into last-attractor-mismatch vs rest.
def makeProbeData(model, layer, batchSize=100):
    trainHidden = makeHiddenStates(model, trainData["tokens"], probeLayer=layer, batchSize=batchSize)
    valHidden = makeHiddenStates(model, valData["tokens"], probeLayer=layer, batchSize=batchSize)

    trainX = gatherPreVerbState(trainHidden, trainData["verbPosition"])
    valX = gatherPreVerbState(valHidden, valData["verbPosition"])

    featureMean = trainX.mean(axis=0)
    featureStd = np.maximum(trainX.std(axis=0), 1e-6)
    trainX = ((trainX - featureMean) / featureStd).astype(np.float32)
    valX = ((valX - featureMean) / featureStd).astype(np.float32)

    valY = valData["subjectNumber"]

    return {
        "PROBE_TRAIN_X": trainX,
        "PROBE_TRAIN_Y": trainData["subjectNumber"],
        "PROBE_VAL_X_MISMATCH": valX[valMismatchMask],
        "PROBE_VAL_Y_MISMATCH": valY[valMismatchMask],
        "PROBE_VAL_X_REST": valX[valRestMask],
        "PROBE_VAL_Y_REST": valY[valRestMask],
        "featureMean": featureMean,
        "featureStd": featureStd,
    }
