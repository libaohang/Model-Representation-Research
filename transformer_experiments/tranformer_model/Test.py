import numpy as np

# data : dict of aligned arrays built by buildDataset in train_gpt.py
#   tokens (N, maxLen), verbPosition (N,), verbCorrect (N,), verbIncorrect (N,), lastMismatch (N,) bool
# Returns the loss per target token, the subject-verb agreement accuracy over all sentences,
# the agreement accuracy over sentences whose last attractor mismatches the subject,
# and the mean margin log P(correct verb) - log P(incorrect verb) over the same two sets
def testNetwork(network, data, loss, padId, batchSize = 100):
    sequences = data["tokens"]
    sampleSize = sequences.shape[0]
    totalLoss = 0.0
    totalTokens = 0
    totalAgree = 0
    mismatchAgree = 0
    mismatchCount = 0
    totalMargin = 0.0
    mismatchMargin = 0.0

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

        # Agreement: at the position that predicts the main verb, does the correct form outscore the incorrect one?
        # Comparing logits is the same as comparing probabilities since both share one softmax; a tie counts as wrong
        rows = np.arange(len(batchSeq))
        verbLogits = logits[rows, data["verbPosition"][batch : batch + batchSize]]  # (batch, vocabSize)
        agree = (verbLogits[rows, data["verbCorrect"][batch : batch + batchSize]] >
                 verbLogits[rows, data["verbIncorrect"][batch : batch + batchSize]])
        totalAgree += int(np.sum(agree))

        mismatch = data["lastMismatch"][batch : batch + batchSize]
        mismatchAgree += int(np.sum(agree & mismatch))
        mismatchCount += int(np.sum(mismatch))

        # Margin: log P(correct verb) - log P(incorrect verb) at the verb position
        # Both log-probabilities share the same softmax normalizer, so this equals the difference of the two logits;
        # unlike a probability difference it is unaffected by how probability is spread over other verbs
        # Positive means the correct form is preferred; 0 means no preference
        margin = (verbLogits[rows, data["verbCorrect"][batch : batch + batchSize]] -
                  verbLogits[rows, data["verbIncorrect"][batch : batch + batchSize]])
        totalMargin += float(np.sum(margin))
        mismatchMargin += float(np.sum(margin[mismatch]))


    # Loss is per target token, averaged over every non-padded position in the dataset
    avgLoss = totalLoss / totalTokens
    # Agreement accuracies are per sentence
    agreementAcc = totalAgree / sampleSize
    lastMismatchAcc = mismatchAgree / mismatchCount if mismatchCount > 0 else float("nan")
    # Margins are averaged per sentence
    avgMargin = totalMargin / sampleSize
    lastMismatchMargin = mismatchMargin / mismatchCount if mismatchCount > 0 else float("nan")

    return avgLoss, agreementAcc, lastMismatchAcc, avgMargin, lastMismatchMargin
