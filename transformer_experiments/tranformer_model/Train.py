import numpy as np

# sequences : (numSentences, maxLen) integer token ids, one sentence per row, right-padded with padId
def trainNetwork(network, loss, optimizer, sequences, padId, epochs = 5, batchSize = 100, lrDecayStart=10):
    sampleSize = sequences.shape[0]

    for epoch in range(epochs):
        epochLoss = 0.0
        numBatches = 0

        indices = np.random.permutation(sampleSize)
        sequences = sequences[indices]

        for batch in range(0, sampleSize, batchSize):
            batchSeq = sequences[batch : batch + batchSize]
            # Next-token prediction: the target at each position is the following token
            batchX = batchSeq[:, :-1]
            batchY = batchSeq[:, 1:]

            # Skip zeroing for the first batch since gradients have not been initialized
            if(epoch > 0 or batch > 0):
                network.zeroGradient()
            # logits : (batch, maxLen - 1, vocabSize)
            logits = network.forward(batchX)

            # Positions whose target is padding are excluded from the loss and get zero gradient
            batchLoss, gradient = loss(logits, batchY, ignoreIndex=padId)
            epochLoss += batchLoss
            numBatches += 1

            network.backward(gradient)
            optimizer.step()

        if (epoch == lrDecayStart - 1):
            print("lr decay")
            optimizer.lrDecay()
        epochLoss /= numBatches
        print(f"Training error on epoch {1 + epoch} is {epochLoss}")


    return network
