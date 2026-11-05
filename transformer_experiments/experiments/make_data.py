import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tranformer_model"))

import random

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
from dataset.generate import make_nonce


trainCube = load_npz(ROOT / "data/valGPT_trainProbe.npz")
valCube = load_npz(ROOT / "data/valProbe.npz")


# Same as probes.make_control.buildDataset but also tracks last_attractor_mismatch
# per row so val rows can be split into "mismatch" (True) vs "rest" (False/None).
def buildDataset(cube):
    tokens, subjectPosition, verbPosition, subjectNumber = [], [], [], []
    subjectTokenId, lastAttractorMismatch = [], []
    verbCorrectId, verbIncorrectId = [], []
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
            verbCorrectId.append(wordIdx[cube["verb_correct"][slot]])
            verbIncorrectId.append(wordIdx[cube["verb_incorrect"][slot]])

    data = {
        "tokens": np.array(tokens),
        "subjectPosition": np.array(subjectPosition),
        "verbPosition": np.array(verbPosition),
        "subjectNumber": np.array(subjectNumber),
        "subjectTokenId": np.array(subjectTokenId),
        "lastAttractorMismatch": np.array(lastAttractorMismatch, dtype=bool),
        "verbCorrectId": np.array(verbCorrectId),
        "verbIncorrectId": np.array(verbIncorrectId),
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


# Mirror of buildDataset that calls make_nonce on each record before padding.
# Also captures per-row verb_correct/verb_incorrect token ids so the LM can be
# evaluated at the pre-verb position without re-running make_nonce.
def buildNonceData(cube, seed):
    rng = random.Random(seed)
    tokens, subjectPosition, verbPosition, subjectNumber = [], [], [], []
    subjectTokenId, lastAttractorMismatch = [], []
    verbCorrectId, verbIncorrectId = [], []
    for idx in np.argwhere(cube["cell_valid"]):
        cell = tuple(idx)
        for k in range(len(cube["tokens"][cell])):
            slot = cell + (k,)
            orig = cube["tokens"][slot]
            subjPosRaw = int(cube["subject_position"][slot])
            verbPosRaw = int(cube["verb_position"][slot])
            subjNum = cube["subject_number"][slot]

            rec = {
                "tokens": list(orig),
                "content_tags": list(cube["content_tags"][slot]),
                "verb_position": verbPosRaw,
                "subject_number": subjNum,
                # Carried through dict(record) in make_nonce. Overwritten if the
                # main verb is successfully swapped; otherwise the originals
                # still agree with the (unchanged) token at verb_position.
                "verb_correct": cube["verb_correct"][slot],
                "verb_incorrect": cube["verb_incorrect"][slot],
            }
            nonceRec = make_nonce(rec, rng)
            sentence = nonceRec["tokens"]

            assert 14 - len(sentence) >= 0
            padding = ["<pad>"] * (14 - len(sentence))
            padded = ["<bos>"] + sentence + ["<eos>"] + padding
            tokens.append([wordIdx[word] for word in padded])

            subjPos = subjPosRaw + 1
            verbPos = verbPosRaw + 1
            subjectPosition.append(subjPos)
            verbPosition.append(verbPos)
            subjectNumber.append(NUMBER_TO_LABEL[subjNum])
            subjectTokenId.append(wordIdx[padded[subjPos]])
            lastAttractorMismatch.append(cube["last_attractor_mismatch"][slot] is True)
            verbCorrectId.append(wordIdx[nonceRec["verb_correct"]])
            verbIncorrectId.append(wordIdx[nonceRec["verb_incorrect"]])

    data = {
        "tokens": np.array(tokens),
        "subjectPosition": np.array(subjectPosition),
        "verbPosition": np.array(verbPosition),
        "subjectNumber": np.array(subjectNumber),
        "subjectTokenId": np.array(subjectTokenId),
        "lastAttractorMismatch": np.array(lastAttractorMismatch, dtype=bool),
        "verbCorrectId": np.array(verbCorrectId),
        "verbIncorrectId": np.array(verbIncorrectId),
    }
    rows = np.arange(len(data["tokens"]))
    assert np.all(data["tokens"][rows, data["subjectPosition"]] == data["subjectTokenId"])
    for tokenId in np.unique(data["subjectTokenId"]):
        assert VOCAB[tokenId] in generate.NOUN_BY_FORM
    verbTokens = data["tokens"][rows, data["verbPosition"]]
    assert np.all(verbTokens >= 3), "verb position landed on a special token"
    return data


nonceTrainData = buildNonceData(trainCube, seed=0)
nonceValData = buildNonceData(valCube, seed=1)

# make_nonce preserves all structural fields, so these should match row-for-row.
for _key in ("subjectNumber", "subjectPosition", "verbPosition", "lastAttractorMismatch"):
    assert np.array_equal(nonceTrainData[_key], trainData[_key])
    assert np.array_equal(nonceValData[_key], valData[_key])


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
        "PROBE_VAL_X": valX,
        "PROBE_VAL_Y": valY,
        "PROBE_VAL_X_MISMATCH": valX[valMismatchMask],
        "PROBE_VAL_Y_MISMATCH": valY[valMismatchMask],
        "PROBE_VAL_X_REST": valX[valRestMask],
        "PROBE_VAL_Y_REST": valY[valRestMask],
        "featureMean": featureMean,
        "featureStd": featureStd,
    }


# Parallel to makeProbeData but trains on nonce train and evaluates on nonce val.
# Standardization uses nonce-train statistics so train and val share one transform.
def makeNonceProbeData(model, layer, batchSize=100):
    trainHidden = makeHiddenStates(model, nonceTrainData["tokens"], probeLayer=layer, batchSize=batchSize)
    valHidden = makeHiddenStates(model, nonceValData["tokens"], probeLayer=layer, batchSize=batchSize)

    trainX = gatherPreVerbState(trainHidden, nonceTrainData["verbPosition"])
    valX = gatherPreVerbState(valHidden, nonceValData["verbPosition"])

    featureMean = trainX.mean(axis=0)
    featureStd = np.maximum(trainX.std(axis=0), 1e-6)
    trainX = ((trainX - featureMean) / featureStd).astype(np.float32)
    valX = ((valX - featureMean) / featureStd).astype(np.float32)

    return {
        "PROBE_TRAIN_X": trainX,
        "PROBE_TRAIN_Y": nonceTrainData["subjectNumber"],
        "PROBE_VAL_X": valX,
        "PROBE_VAL_Y": nonceValData["subjectNumber"],
        "featureMean": featureMean,
        "featureStd": featureStd,
    }
