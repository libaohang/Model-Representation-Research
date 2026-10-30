import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from Optimizer import AdamW
from Loss import crossEntropyLoss
from LinearLayer import Linear
from experiments.make_data import model, makeProbeData


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


def layerAttractorMismatchAccuracy():
    results = []
    layerEpochs = [7, 6, 9, 8, 6]

    for epochs, layer in zip(layerEpochs, range(5)):
        print(f"\n=== Layer {layer} ===")
        probeData = makeProbeData(model, layer=layer)

        probe = Linear(32, 2)
        _trainOne(probe, probeData["PROBE_TRAIN_X"], probeData["PROBE_TRAIN_Y"],
                  epochs=epochs, batchSize=100, lr=1e-3, name=f"L{layer}")

        mismatchAcc, mismatchMargin = _evaluate(
            probe, probeData["PROBE_VAL_X_MISMATCH"],
            probeData["PROBE_VAL_Y_MISMATCH"], batchSize=100)
        restAcc, restMargin = _evaluate(
            probe, probeData["PROBE_VAL_X_REST"],
            probeData["PROBE_VAL_Y_REST"], batchSize=100)

        print(f"Layer {layer}  mismatch acc={mismatchAcc:.4f}  rest acc={restAcc:.4f}")
        results.append((layer, mismatchAcc, mismatchMargin, restAcc, restMargin))

    print("\n=== Summary ===")
    header = f"{'Layer':<8} {'mismatchAcc':>12} {'restAcc':>10} {'gap':>8} {'mismatchMarg':>14} {'restMarg':>11}"
    print(header)
    print("-" * len(header))
    for layer, mAcc, mMarg, rAcc, rMarg in results:
        gap = rAcc - mAcc
        print(f"{layer:<8} {mAcc:>12.4f} {rAcc:>10.4f} {gap:>8.4f} {mMarg:>14.4f} {rMarg:>11.4f}")


layerAttractorMismatchAccuracy()
