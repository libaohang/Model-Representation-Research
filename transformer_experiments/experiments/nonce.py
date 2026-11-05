import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from Optimizer import AdamW
from Loss import crossEntropyLoss
from LinearLayer import Linear
from experiments.make_data import (
    model,
    makeProbeData,
    makeNonceProbeData,
    trainData,
    nonceTrainData,
)


def _trainOne(probe, X, Y, epochs, batchSize, lr, name):
    optimizer = AdamW(probe, lr=lr)
    n = len(X)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epochLoss = 0.0
        numBatches = 0
        for start in range(0, n, batchSize):
            rows = indices[start:start + batchSize]
            batchX = X[rows]
            batchY = Y[rows]

            logits = probe.forward(batchX)
            batchLoss, grad = crossEntropyLoss(logits, batchY)
            probe.backward(grad)
            optimizer.step()

            epochLoss += float(batchLoss)
            numBatches += 1

        print(f"Probe {name} epoch {1 + epoch}: loss={epochLoss / numBatches:.4f}")


def _evaluate(probe, X, Y, batchSize):
    n = len(X)
    allLogits = []
    for start in range(0, n, batchSize):
        allLogits.append(probe.forward(X[start:start + batchSize]))
    logits = np.concatenate(allLogits, axis=0)

    preds = logits.argmax(axis=-1)
    accuracy = float((preds == Y).mean())

    rows = np.arange(n)
    margin = float((logits[rows, Y] - logits[rows, 1 - Y]).mean())
    return accuracy, margin


def layerNormalVsNonce():
    results = []
    layerEpochs = [7, 6, 9, 8, 6]

    for epochs, layer in zip(layerEpochs, range(5)):
        print(f"\n=== Layer {layer} ===")

        normalData = makeProbeData(model, layer=layer)
        nonceData = makeNonceProbeData(model, layer=layer)

        normalProbe = Linear(32, 2)
        _trainOne(normalProbe, normalData["PROBE_TRAIN_X"], normalData["PROBE_TRAIN_Y"],
                  epochs=epochs, batchSize=100, lr=1e-3, name=f"L{layer}-normal")

        nonceProbe = Linear(32, 2)
        _trainOne(nonceProbe, nonceData["PROBE_TRAIN_X"], nonceData["PROBE_TRAIN_Y"],
                  epochs=epochs, batchSize=100, lr=1e-3, name=f"L{layer}-nonce")

        normalAcc, normalMargin = _evaluate(
            normalProbe, normalData["PROBE_VAL_X"], normalData["PROBE_VAL_Y"], batchSize=100)
        nonceAcc, nonceMargin = _evaluate(
            nonceProbe, nonceData["PROBE_VAL_X"], nonceData["PROBE_VAL_Y"], batchSize=100)

        print(f"Layer {layer}  normal acc={normalAcc:.4f}  nonce acc={nonceAcc:.4f}")
        results.append((layer, normalAcc, normalMargin, nonceAcc, nonceMargin))

    print("\n=== Summary ===")
    header = f"{'Layer':<8} {'normalAcc':>10} {'nonceAcc':>10} {'gap':>8} {'normalMarg':>12} {'nonceMarg':>11}"
    print(header)
    print("-" * len(header))
    for layer, nAcc, nMarg, noAcc, noMarg in results:
        gap = nAcc - noAcc
        print(f"{layer:<8} {nAcc:>10.4f} {noAcc:>10.4f} {gap:>8.4f} {nMarg:>12.4f} {noMarg:>11.4f}")


# Single pass of the trained GPT over `data` at position verb_position - 1.
# Returns (accuracy, meanMargin) for the correct-vs-incorrect verb pair.
def _lmAgreement(data, batchSize):
    tokens = data["tokens"]
    verbPosition = data["verbPosition"]
    verbCorrectId = data["verbCorrectId"]
    verbIncorrectId = data["verbIncorrectId"]
    n = len(tokens)

    allMargins = np.empty(n, dtype=np.float32)
    allCorrect = np.empty(n, dtype=bool)
    for start in range(0, n, batchSize):
        end = start + batchSize
        batch = tokens[start:end]
        logits = model.forward(batch)
        rows = np.arange(len(batch))
        preVerb = verbPosition[start:end] - 1
        logitsAtPreVerb = logits[rows, preVerb]
        c = verbCorrectId[start:end]
        w = verbIncorrectId[start:end]
        margin = logitsAtPreVerb[rows, c] - logitsAtPreVerb[rows, w]
        allMargins[start:end] = margin
        allCorrect[start:end] = margin > 0

    return float(allCorrect.mean()), float(allMargins.mean())


# Agreement accuracy of the trained GPT on the LM validation set
# (valGPT_trainProbe.npz) in both normal and nonce forms.
def lmNonceAgreement(batchSize=100):
    normalAcc, normalMargin = _lmAgreement(trainData, batchSize)
    nonceAcc, nonceMargin = _lmAgreement(nonceTrainData, batchSize)

    print(f"\n=== LM agreement on LM-val (valGPT_trainProbe.npz) ===")
    print(f"  normal: accuracy = {normalAcc:.4f}  mean margin = {normalMargin:.4f}")
    print(f"  nonce : accuracy = {nonceAcc:.4f}  mean margin = {nonceMargin:.4f}")
    print(f"  gap   : accuracy = {normalAcc - nonceAcc:+.4f}  mean margin = {normalMargin - nonceMargin:+.4f}")
    return normalAcc, normalMargin, nonceAcc, nonceMargin


layerNormalVsNonce()
lmNonceAgreement()
