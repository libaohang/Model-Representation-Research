from Layer import Layer
import numpy as np

# Lookup table mapping token ids to vectors; the same matrix is reused by LMHead (weight tying)
class TokenEmbedding(Layer):
    def __init__(self, vocabSize, modelDim):
        self.weights = (np.random.randn(vocabSize, modelDim) * 0.02).astype(np.float32)
        # The matrix receives gradient from two places: the lookup here and the LMHead projection
        self.dE_dWEmbed = np.zeros_like(self.weights)
        self.dE_dWHead = np.zeros_like(self.weights)

    def forward(self, ids):
        # ids : (batch, seqLen) integers
        self.ids = ids
        # (batch, seqLen, modelDim)
        return self.weights[ids]

    def backward(self, dE_dY):
        # dE_dY : (batch, seqLen, modelDim)
        self.dE_dWEmbed = np.zeros_like(self.weights)
        # np.add.at accumulates over repeated ids; fancy-index += would only keep one of them
        np.add.at(self.dE_dWEmbed, self.ids, dE_dY)
        # ids are not differentiable, so there is nothing to pass back
        return None

    def parameters(self):
        return [(self.weights, "tok emb")]

    def gradients(self):
        return [(self.dE_dWEmbed + self.dE_dWHead, "tok emb")]


# Learned position embedding added to the token embeddings
class PositionEmbedding(Layer):
    def __init__(self, contextLen, modelDim):
        self.contextLen = contextLen
        self.weights = (np.random.randn(contextLen, modelDim) * 0.02).astype(np.float32)

    def forward(self, tokens):
        # tokens : (batch, seqLen, modelDim)
        self.seqLen = tokens.shape[1]
        assert self.seqLen <= self.contextLen
        # Position rows broadcast over the batch dimension
        return tokens + self.weights[np.arange(self.seqLen)]

    def backward(self, dE_dY):
        # Only the first seqLen positions were used
        self.dE_dW = np.zeros_like(self.weights)
        self.dE_dW[:self.seqLen] = np.sum(dE_dY, axis=0)
        return dE_dY

    def parameters(self):
        return [(self.weights, "pos emb")]

    def gradients(self):
        return [(self.dE_dW, "pos emb")]


# Projects tokens to vocabulary logits using the token embedding matrix (weight tying, no bias)
class LMHead(Layer):
    def __init__(self, tokenEmbedding):
        self.tokenEmbedding = tokenEmbedding

    def forward(self, tokens):
        # tokens : (batch, seqLen, modelDim)
        self.input = tokens
        # (batch, seqLen, vocabSize)
        return tokens @ self.tokenEmbedding.weights.T

    def backward(self, dE_dY):
        # dE_dY : (batch, seqLen, vocabSize)
        # The gradient is stored on the embedding, which reports it summed with its own
        self.tokenEmbedding.dE_dWHead = np.sum(dE_dY.transpose(0, 2, 1) @ self.input, axis=0)  # (vocabSize, modelDim)
        # (batch, seqLen, modelDim)
        return dE_dY @ self.tokenEmbedding.weights

    # The weights belong to tokenEmbedding, so they are not listed again here
    def parameters(self):
        return []

    def gradients(self):
        return []
