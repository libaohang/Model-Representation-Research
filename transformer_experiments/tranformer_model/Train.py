import numpy as np
from Test import testNetwork

# sequences : (numSentences, maxLen) integer token ids, one sentence per row, right-padded with padId
# valData : held-out data evaluated at the end of every epoch; dict of aligned arrays from buildDataset in train_gpt.py
#           (tokens in the same format as sequences, plus the main verb's position and correct/incorrect forms)
# saveDir : existing folder to save the weights to after every epoch; None skips saving
def trainNetwork(network, loss, optimizer, sequences, padId, valData, epochs = 5, batchSize = 100, lrDecayStart=10, saveDir=None):
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

        #dt = time.time() - t0
        #print(f"1 epoch took {dt:.2f}s")
        print(f"Training error on epoch {1 + epoch} is {epochLoss}")

        valLoss, agreementAcc, lastMismatchAcc = testNetwork(network, valData, loss, padId, batchSize)
        print(f"Validation error on epoch {1 + epoch} is {valLoss}")
        print(f"Validation agreement accuracy on epoch {1 + epoch} is {agreementAcc}")
        print(f"Validation agreement accuracy (last attractor mismatched) on epoch {1 + epoch} is {lastMismatchAcc}")

        if saveDir is not None:
            params = [arr for (arr, name) in network.parameters()]
            np.savez(f"{saveDir}/GPT_{1 + epoch}epochs.npz", *params)


    return network
