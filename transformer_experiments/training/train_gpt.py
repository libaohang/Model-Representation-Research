import sys
from pathlib import Path

# transformer_experiments/, so `dataset` is importable however the script is launched
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The model files import each other by bare name (`from Layer import Layer`), so their folder must be on the path too
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from GPT import GPT
from Train import trainNetwork
from Test import testNetwork
from Loss import crossEntropyLoss
from Optimizer import AdamW
from dataset.sample import load_npz
from dataset import generate


VOCAB = ["<pad>", "<bos>", "<eos>"] + generate.VOCABULARY
wordIdx = {word: i for i, word in enumerate(VOCAB)}
padId = wordIdx["<pad>"]

trainCube = load_npz(ROOT / "data/trainGPT.npz")
valCube = load_npz(ROOT / "data/valGPT_trainProbe.npz")

# Flattens every valid cell of a cube into aligned arrays, one row per sentence:
#   tokens        (N, 16) ids: <bos> sentence <eos> <pad>...
#   verbPosition  (N,) index of the main verb in the sentence; with <bos> prepended this is also
#                 the logits position that predicts the verb (logits[:, t] predicts token t + 1)
#   verbCorrect / verbIncorrect  (N,) ids of the verb form that agrees / disagrees with the subject
#   lastMismatch  (N,) True when the attractor closest to the verb has the opposite number to the subject
def buildDataset(cube):
    tokens, verbPosition, verbCorrect, verbIncorrect, lastMismatch = [], [], [], [], []
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

            verbPosition.append(cube["verb_position"][slot])
            verbCorrect.append(wordIdx[cube["verb_correct"][slot]])
            verbIncorrect.append(wordIdx[cube["verb_incorrect"][slot]])
            # None (no attractors) and False both count as not mismatched
            lastMismatch.append(cube["last_attractor_mismatch"][slot] is True)

    data = {
        "tokens": np.array(tokens),
        "verbPosition": np.array(verbPosition),
        "verbCorrect": np.array(verbCorrect),
        "verbIncorrect": np.array(verbIncorrect),
        "lastMismatch": np.array(lastMismatch, dtype=bool),
    }
    # The verb sits one slot later in the padded row because of <bos>
    rows = np.arange(len(data["tokens"]))
    assert np.all(data["tokens"][rows, data["verbPosition"] + 1] == data["verbCorrect"])
    return data

trainData = buildDataset(trainCube)
valData = buildDataset(valCube)

small = False
if(small):
    # Train on a random subset for quick runs
    subsetFraction = 0.1
    rng = np.random.default_rng(0)
    trainKeep = rng.permutation(len(trainData["tokens"]))[:int(subsetFraction * len(trainData["tokens"]))]
    trainData = {key: arr[trainKeep] for key, arr in trainData.items()}
    # Validation runs every epoch, so shrink it by the same fraction to keep it from dominating the run time
    # One permutation for every array keeps each sentence's verb info aligned with its tokens
    valKeep = rng.permutation(len(valData["tokens"]))[:int(subsetFraction * len(valData["tokens"]))]
    valData = {key: arr[valKeep] for key, arr in valData.items()}
    print(f"Training on {len(trainData['tokens'])} sentences, validating on {len(valData['tokens'])}")

model = GPT(vocabSize=len(VOCAB), contextLen=16, modelDim=32, numHeads=4, numLayers=4)

optimizer = AdamW(model, lr=0.001, weightDecay=0.001)

# Weights go to transformer_experiments/model/, created if missing
modelDir = ROOT / "model_32dim_4heads"
modelDir.mkdir(exist_ok=True)

model = trainNetwork(model, crossEntropyLoss, optimizer, trainData["tokens"], padId, valData, epochs = 6, batchSize = 100, saveDir=modelDir)

params = [arr for (arr, name) in model.parameters()]
np.savez(modelDir / "GPT_final.npz", *params)

testCube = load_npz(ROOT / "data/testGPT.npz")
testData = buildDataset(testCube)
#data = np.load(modelDir / "GPT_3epochs.npz")
#for i, (p, _) in enumerate(model.parameters()):
#    p[...] = data[f"arr_{i}"]

testLoss, agreementAcc, lastMismatchAcc = testNetwork(model, testData, crossEntropyLoss, padId, batchSize = 100)
print(f"Test error is {testLoss}")
print(f"Test agreement accuracy is {agreementAcc}")
print(f"Test agreement accuracy (last attractor mismatched) is {lastMismatchAcc}")
