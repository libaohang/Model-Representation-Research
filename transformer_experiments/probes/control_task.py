import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tranformer_model"))

import numpy as np
from Optimizer import AdamW
from Loss import crossEntropyLoss
from LinearLayer import Linear
from MultiLayerPerceptron import MLP
from Activations import GELU
from make_control import model, makeProbeData


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

            # Linear / MLP backward overwrites dE_dW and dE_dB, so no zero-grad step needed
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
    # Binary analogue of the margin in Test.py:53-54: score of the true class
    # minus the score of the other class, averaged over samples.
    margin = float((logits[rows, Y] - logits[rows, 1 - Y]).mean())
    return accuracy, margin


# Train two probes with identical input features but different labels:
#   normalProbe  <- data["PROBE_TRAIN_Y"]          (real singular/plural)
#   controlProbe <- data["PROBE_TRAIN_Y_CONTROL"]  (random-but-fixed per noun form)
# `data` is the dict returned by make_control.makeProbeData for the layer
# whose selectivity you want to measure. Both probes must map (B, modelDim) ->
# (B, 2). Returns a dict with per-probe val accuracy, margin, and selectivity.
def trainProbes(normalProbe, controlProbe, data, epochs=10, batchSize=100, lr=1e-3):
    _trainOne(normalProbe, data["PROBE_TRAIN_X"], data["PROBE_TRAIN_Y"],
              epochs, batchSize, lr, name="normal")
    _trainOne(controlProbe, data["PROBE_TRAIN_X"], data["PROBE_TRAIN_Y_CONTROL"],
              epochs, batchSize, lr, name="control")

    normalAcc, normalMargin = _evaluate(normalProbe, data["PROBE_VAL_X"], data["PROBE_VAL_Y"], batchSize)
    controlAcc, controlMargin = _evaluate(controlProbe, data["PROBE_VAL_X"], data["PROBE_VAL_Y_CONTROL"], batchSize)
    selectivity = normalAcc - controlAcc

    print(f"Normal  probe: val acc={normalAcc:.4f}  margin={normalMargin:.4f}")
    print(f"Control probe: val acc={controlAcc:.4f}  margin={controlMargin:.4f}")
    print(f"Selectivity (normal - control): {selectivity:.4f}")

    return {
        "normalAcc": normalAcc,
        "controlAcc": controlAcc,
        "selectivity": selectivity,
        "normalMargin": normalMargin,
        "controlMargin": controlMargin,
    }


def compareProbes():
    probeWidths = [32, 64, 128, 256]

    labels = ["Linear"] + [f"MLP width={w}" for w in probeWidths]
    normal = [Linear(32, 2)] + [MLP(32, [w, 2], GELU) for w in probeWidths]
    control = [Linear(32, 2)] + [MLP(32, [w, 2], GELU) for w in probeWidths]

    # Hidden states for the probe-structure sweep come from the last transformer block.
    probeData = makeProbeData(model, layer=4)

    results = []
    for label, normalProbe, controlProbe in zip(labels, normal, control):
        print(f"\n=== {label} ===")
        result = trainProbes(normalProbe, controlProbe, probeData, epochs=3)
        results.append((label, result))

    print("\n=== Summary ===")
    header = f"{'Probe':<16} {'normalAcc':>10} {'controlAcc':>11} {'selectivity':>12} {'normalMarg':>11} {'controlMarg':>12}"
    print(header)
    print("-" * len(header))
    for label, r in results:
        print(f"{label:<16} {r['normalAcc']:>10.4f} {r['controlAcc']:>11.4f} "
            f"{r['selectivity']:>12.4f} {r['normalMargin']:>11.4f} {r['controlMargin']:>12.4f}")

#compareProbes()

def layerSelectivity():
    results = []
    layerEpochs = [7, 6, 9, 8, 6]

    for epoch, layer in zip(layerEpochs, range(5)):
        print(f"\n=== Layer {layer} ===")
        probeData = makeProbeData(model, layer=layer)

        normalProbe = Linear(32, 2)
        controlProbe = Linear(32, 2)
        result = trainProbes(normalProbe, controlProbe, probeData, epochs=epoch)
        results.append((layer, result))

    print("\n=== Summary ===")
    header = f"{'Layer':<8} {'normalAcc':>10} {'controlAcc':>11} {'selectivity':>12} {'normalMarg':>11} {'controlMarg':>12}"
    print(header)
    print("-" * len(header))
    for layer, r in results:
        print(f"{layer:<8} {r['normalAcc']:>10.4f} {r['controlAcc']:>11.4f} "
            f"{r['selectivity']:>12.4f} {r['normalMargin']:>11.4f} {r['controlMargin']:>12.4f}")

layerSelectivity()