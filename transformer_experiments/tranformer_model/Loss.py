import numpy as np

def crossEntropyLoss(logits, labels, ignoreIndex=None):
    # logits: (..., C), e.g. (B, T, vocabSize) for GPT
    # labels: (...), e.g. (B, T) integer token ids; for next-token prediction these are the inputs shifted left by one
    # ignoreIndex: label value to exclude from the loss, e.g. the padding token id

    numClasses = logits.shape[-1]
    flatLogits = logits.reshape(-1, numClasses)          # (N, C) where N = B * T
    flatLabels = labels.reshape(-1).astype(np.int64)     # (N,)
    rows = np.arange(flatLabels.shape[0])

    # valid : (N,) True at positions that count towards the loss
    if ignoreIndex is None:
        valid = np.ones(flatLabels.shape[0], dtype=bool)
    else:
        valid = flatLabels != ignoreIndex
    # Ignored labels are replaced by 0 so they are always a legal index; their contribution is masked out below
    safeLabels = np.where(valid, flatLabels, 0)
    n = max(valid.sum(), 1)

    # stable log-softmax
    shifted = flatLogits - flatLogits.max(axis=-1, keepdims=True)
    logProbs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    # mean over the valid positions only
    loss = -(logProbs[rows, safeLabels] * valid).sum() / n

    # gradient w.r.t logits: (probs - onehot) / n at valid positions, 0 at ignored positions
    grad = np.exp(logProbs)
    grad[rows, safeLabels] -= 1
    grad *= valid[:, None]
    grad /= n

    return loss, grad.reshape(logits.shape)
