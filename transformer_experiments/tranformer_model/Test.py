import numpy as np

def testNetwork(network, sequences, loss, padId, batchSize = 100):
    sampleSize = sequences.shape[0]
    totalLoss = 0.0
    totalCorrect = 0
    totalTokens = 0

    for batch in range(0, sampleSize, batchSize):
        batchSeq = sequences[batch : batch + batchSize]
        # Next-token prediction: the target at each position is the following token
        batchX = batchSeq[:, :-1]
        batchY = batchSeq[:, 1:]
        # valid : (batch, maxLen - 1) True where the target is a real token
        valid = batchY != padId
        numValid = int(np.sum(valid))

        # logits : (batch, maxLen - 1, vocabSize)
        logits = network.forward(batchX)

        # Positions whose target is padding are excluded from the loss
        # batchLoss is a mean over this batch's valid targets, so weight it by their count
        batchLoss, _ = loss(logits, batchY, ignoreIndex=padId)
        totalLoss += float(batchLoss) * numValid
        totalTokens += numValid

        preds = np.argmax(logits, axis=2)
        totalCorrect += int(np.sum((preds == batchY) & valid))


    # Both are per target token, averaged over every non-padded position in the dataset
    avgLoss = totalLoss / totalTokens
    acc = totalCorrect / totalTokens

    return avgLoss, acc
