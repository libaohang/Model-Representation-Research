from Layer import Layer
from LinearLayer import Linear
from Softmax import Softmax
import numpy as np

# Causal Scaled Dot-Product Attention: each token attends only to itself and earlier tokens
class AttentionHead(Layer):
    def __init__(self, modelDim, headSize):
        self.headSize = headSize

        self.query = Linear(modelDim, headSize)
        self.key = Linear(modelDim, headSize)
        self.value = Linear(modelDim, headSize)
        self.softmax = Softmax()
        # Softmax is not needed here but just to be consistent
        self.layers = [self.query, self.key, self.value, self.softmax]

    def forward(self, tokens):
        # tokens : (batch, seqLen, modelDim)
        self.Q = self.query.forward(tokens)
        self.K = self.key.forward(tokens)
        self.V = self.value.forward(tokens)
        # Q, K, V : (batch, seqLen, headSize)

        score = self.Q @ self.K.transpose(0, 2, 1) 
        score /= self.headSize ** 0.5

        # Causal mask: True above the diagonal marks future tokens, which get zero attention after softmax
        T = tokens.shape[1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        score = np.where(mask, -np.inf, score)

        # self.attention : (batch, seqLen, seqLen), lower triangular; row i is how token i attends to tokens 0..i
        self.attention = self.softmax.forward(score, axis=-1)
        attention = self.attention @ self.V
        # attention : (batch, seqLen, headSize)

        return attention
    
    def backward(self, dE_dY):
        # dE_dY : (batch, seqLen, headSize)
        # 2nd dimension of dE_dY correspond 2nd dimension of self.attention
        dE_dV = self.attention.transpose(0, 2, 1) @ dE_dY # dE_dV : (batch, seqLen, headSize)
        dE_dAtt = dE_dY @ self.V.transpose(0, 2, 1) # dE_dAtt : (batch, seqLen, seqLen)
        # Masked positions have zero attention, so softmax backward gives them zero gradient
        dE_dScore = self.softmax.backward(dE_dAtt)
        dE_dScore /= self.headSize ** 0.5

        # 2nd dimension of score correspond to 2nd dimension of Q, 3rd correspond to 2nd dimension of K
        dE_dQ = dE_dScore @ self.K
        dE_dK = dE_dScore.transpose(0, 2, 1) @ self.Q

        dE_dTokensQ = self.query.backward(dE_dQ)
        dE_dTokensK = self.key.backward(dE_dK)
        dE_dTokensV = self.value.backward(dE_dV)
        # (batch, seqLen, modelDim)
        dE_dTokens = dE_dTokensQ + dE_dTokensK + dE_dTokensV

        return dE_dTokens
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            for parameter in layer.parameters():
                parameters.append(parameter)
        return parameters
    
    def gradients(self):
        gradients = []
        for layer in self.layers:
            for gradient in layer.gradients():
                gradients.append(gradient)
        return gradients


class MultiHeadAttention(Layer):
    def __init__(self, modelDim, numHeads):
        assert modelDim % numHeads == 0
        self.headSize = modelDim // numHeads
        self.linear = Linear(modelDim, modelDim)

        self.heads = [AttentionHead(modelDim, self.headSize) for _ in range(numHeads)]
        self.layers = self.heads + [self.linear]

    # Feed tokens to numHeads and concatenate outputs together; output has same shape as input
    def forward(self, tokens):
        outTokens = []
        for head in self.heads:
            outTokens.append(head.forward(tokens))
        outTokens = np.concatenate(outTokens, axis=-1)

        outTokens = self.linear.forward(outTokens)
        return outTokens
    
    def backward(self, dE_dY):
        dE_dOut = self.linear.backward(dE_dY)

        dE_dTokens = np.zeros_like(dE_dY)
        splitGradient = np.split(dE_dOut, len(self.heads), axis=-1)
        for head, tokens in zip(self.heads, splitGradient):
            dE_dTokens += head.backward(tokens)
        
        return dE_dTokens
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            for parameter in layer.parameters():
                parameters.append(parameter)
        return parameters
        
    def gradients(self):
        gradients = []
        for layer in self.layers:
            for gradient in layer.gradients():
                gradients.append(gradient)
        return gradients

