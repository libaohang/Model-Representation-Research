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
from Loss import crossEntropyLoss
from Optimizer import AdamW
from dataset.sample import load_npz
from dataset import generate


VOCAB = ["<pad>", "<bos>", "<eos>"] + generate.VOCABULARY
wordIdx = {word: i for i, word in enumerate(VOCAB)}
padId = wordIdx["<pad>"]

trainCube = load_npz(ROOT / "data/trainGPT.npz")
valCube = load_npz(ROOT / "data/valGPT_trainProbe.npz")

trainIdx = np.argwhere(trainCube["cell_valid"])
valIdx = np.argwhere(valCube["cell_valid"])

trainTokens = []
for idx in trainIdx:
    tg, subj, na, nm, lm = idx
    sentences = trainCube["tokens"][tg, subj, na, nm, lm]
    for sentence in sentences:
        assert 14 - len(sentence) >= 0
        padding = ["<pad>"] * (14 - len(sentence))
        # <eos> goes right after the sentence so the model learns where it ends; padding stays on the right
        padded = ["<bos>"] + sentence + ["<eos>"] + padding
        trainTokens.append([wordIdx[word] for word in padded])
trainTokens = np.array(trainTokens)

valTokens = []
for idx in valIdx:
    tg, subj, na, nm, lm = idx
    sentences = valCube["tokens"][tg, subj, na, nm, lm]
    for sentence in sentences:
        assert 14 - len(sentence) >= 0
        padding = ["<pad>"] * (14 - len(sentence))
        padded = ["<bos>"] + sentence + ["<eos>"] + padding
        valTokens.append([wordIdx[word] for word in padded])
valTokens = np.array(valTokens)

model = GPT(vocabSize=len(VOCAB), contextLen=16, modelDim=64, numHeads=4, numLayers=4)

optimizer = AdamW(model, lr=0.001, weightDecay=0.001)

model = trainNetwork(model, crossEntropyLoss, optimizer, trainTokens, padId, valTokens, epochs = 5, batchSize = 100)