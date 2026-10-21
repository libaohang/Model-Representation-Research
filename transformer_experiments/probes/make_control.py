import sys
from pathlib import Path

# transformer_experiments/, so `dataset` is importable however the script is launched
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The model files import each other by bare name (`from Layer import Layer`), so their folder must be on the path too
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from GPT import GPT
from dataset.sample import load_npz
from dataset import generate


VOCAB = ["<pad>", "<bos>", "<eos>"] + generate.VOCABULARY
wordIdx = {word: i for i, word in enumerate(VOCAB)}
padId = wordIdx["<pad>"]

NUMBER_TO_LABEL = {"singular": 0, "plural": 1}

trainCube = load_npz(ROOT / "data/valGPT_trainProbe.npz")
valCube = load_npz(ROOT / "data/valProbe.npz")

# One row per valid sentence, anchored at the subject noun:
#   tokens          (N, 16) ids: <bos> sentence <eos> <pad>...
#   subjectPosition (N,) index into the padded row of the subject noun
#                   (cube subject_position shifted by +1 for the <bos> prefix)
#   verbPosition    (N,) index into the padded row of the main verb
#                   (cube verb_position shifted by +1 for the <bos> prefix)
#   subjectNumber   (N,) int label: 0 = singular, 1 = plural
#   subjectTokenId  (N,) vocab id of the subject noun; used to look up its
#                   control-task label
def buildDataset(cube):
    tokens, subjectPosition, verbPosition, subjectNumber, subjectTokenId = [], [], [], [], []
    for idx in np.argwhere(cube["cell_valid"]):
        cell = tuple(idx)
        for k in range(len(cube["tokens"][cell])):
            slot = cell + (k,)
            sentence = cube["tokens"][slot]
            assert 14 - len(sentence) >= 0
            padding = ["<pad>"] * (14 - len(sentence))
            # <eos> goes right after the sentence so the model learns where it ends; padding stays on the right
            padded = ["<bos>"] + sentence + ["<eos>"] + padding
            tokens.append([wordIdx[word] for word in padded])

            # Subject and verb each sit one slot later in the padded row because of <bos>
            subjPos = int(cube["subject_position"][slot]) + 1
            verbPos = int(cube["verb_position"][slot]) + 1
            subjectPosition.append(subjPos)
            verbPosition.append(verbPos)
            subjectNumber.append(NUMBER_TO_LABEL[cube["subject_number"][slot]])
            subjectTokenId.append(wordIdx[padded[subjPos]])

    data = {
        "tokens": np.array(tokens),
        "subjectPosition": np.array(subjectPosition),
        "verbPosition": np.array(verbPosition),
        "subjectNumber": np.array(subjectNumber),
        "subjectTokenId": np.array(subjectTokenId),
    }
    rows = np.arange(len(data["tokens"]))
    assert np.all(data["tokens"][rows, data["subjectPosition"]] == data["subjectTokenId"])
    # The subject token must be a known noun form
    for tokenId in np.unique(data["subjectTokenId"]):
        assert VOCAB[tokenId] in generate.NOUN_BY_FORM
    # Verb position must land on a real verb token (not <bos>, <eos>, or <pad>)
    verbTokens = data["tokens"][rows, data["verbPosition"]]
    assert np.all(verbTokens >= 3), "verb position landed on a special token"
    return data


# Hewitt & Liang control labels: each noun surface form in the vocab gets a
# fixed random {0, 1} label. Sorting forms before sampling makes the mapping
# deterministic across Python runs regardless of dict iteration order.
def makeControlLabels(seed):
    rng = np.random.default_rng(seed)
    labels = {}
    for form in sorted(generate.NOUN_BY_FORM.keys()):
        labels[wordIdx[form]] = int(rng.integers(0, 2))
    return labels


# Project per-row subjectTokenId through the control dict.
def applyControlLabels(data, controlLabels):
    lookup = np.full(len(VOCAB), -1, dtype=np.int64)
    for tokenId, label in controlLabels.items():
        lookup[tokenId] = label
    out = lookup[data["subjectTokenId"]]
    assert np.all(out >= 0)
    return out


# Run model.probeLayer in batches and return the full hidden-state tensor.
def makeHiddenStates(model, tokens, probeLayer, batchSize=256):
    outputs = []
    for start in range(0, len(tokens), batchSize):
        batch = tokens[start:start + batchSize]
        outputs.append(model.probeLayer(batch, probeLayer))
    return np.concatenate(outputs, axis=0)


# Slice (N, 16, D) hidden states down to (N, D) at the subject position.
def gatherSubjectStates(hidden, subjectPosition):
    return hidden[np.arange(len(hidden)), subjectPosition]


# Return the single hidden state at the token immediately before the verb,
# (N, D). This is the state the model uses to predict the verb token itself.
def gatherPreVerbState(hidden, verbPosition):
    return hidden[np.arange(len(hidden)), verbPosition - 1]


trainData = buildDataset(trainCube)
valData = buildDataset(valCube)

model = GPT(vocabSize=len(VOCAB), contextLen=16, modelDim=32, numHeads=4, numLayers=4)

# Trained weights live under transformer_experiments/models/
modelDir = ROOT / "models/model_32dim_4heads"
weights = np.load(modelDir / "GPT_1epochs.npz")
for i, (p, _) in enumerate(model.parameters()):
    p[...] = weights[f"arr_{i}"]


# Build the normal + control probe datasets at a chosen depth of the model.
# layer=0 returns the state after the embeddings and before any transformer
# block; layer=k (1 <= k <= numLayers) returns the state after block k-1.
def makeProbeData(model, layer, controlSeed=1, batchSize=100):
    trainHidden = makeHiddenStates(model, trainData["tokens"], probeLayer=layer, batchSize=batchSize)
    valHidden = makeHiddenStates(model, valData["tokens"], probeLayer=layer, batchSize=batchSize)

    # Single hidden state right before the verb — what the model conditions on
    # when predicting it. Shape (N, modelDim).
    trainX = gatherPreVerbState(trainHidden, trainData["verbPosition"])
    valX = gatherPreVerbState(valHidden, valData["verbPosition"])

    # Standardize each feature using training-set statistics so train and val
    # share one transform. Clamp std to avoid divide-by-zero on dead features.
    featureMean = trainX.mean(axis=0)
    featureStd = np.maximum(trainX.std(axis=0), 1e-6)
    trainX = ((trainX - featureMean) / featureStd).astype(np.float32)
    valX = ((valX - featureMean) / featureStd).astype(np.float32)

    # Control task: same X, random-but-fixed per-noun-form Y (Hewitt & Liang 2019).
    randomLabels = makeControlLabels(controlSeed)

    return {
        "PROBE_TRAIN_X": trainX,
        "PROBE_VAL_X": valX,
        "PROBE_TRAIN_Y": trainData["subjectNumber"],
        "PROBE_VAL_Y": valData["subjectNumber"],
        "PROBE_TRAIN_Y_CONTROL": applyControlLabels(trainData, randomLabels),
        "PROBE_VAL_Y_CONTROL": applyControlLabels(valData, randomLabels),
        "featureMean": featureMean,
        "featureStd": featureStd,
    }

