from Layer import Layer
from Embedding import TokenEmbedding, PositionEmbedding, LMHead
from Transformer import TransformerBlock
from LayerNorm import LayerNorm
from Activations import GELU

# Decoder-only transformer:
# token embedding -> position embedding -> numLayers transformer blocks -> layernorm -> LM head
class GPT(Layer):
    def __init__(self, vocabSize, contextLen, modelDim, numHeads, numLayers, mlpWidth=None, activation=GELU):
        if mlpWidth is None:
            mlpWidth = 4 * modelDim
        self.contextLen = contextLen

        self.tokenEmbedding = TokenEmbedding(vocabSize, modelDim)
        self.positionEmbedding = PositionEmbedding(contextLen, modelDim)
        self.blocks = [TransformerBlock(modelDim, numHeads, mlpWidth, activation) for _ in range(numLayers)]
        self.finalNorm = LayerNorm(modelDim)
        # Shares its weight matrix with tokenEmbedding
        self.head = LMHead(self.tokenEmbedding)

        self.layers = [self.tokenEmbedding, self.positionEmbedding] + self.blocks + [self.finalNorm, self.head]

    def forward(self, ids):
        # ids : (batch, seqLen) integers, seqLen <= contextLen
        input = ids
        for layer in self.layers:
            input = layer.forward(input)
        # logits : (batch, seqLen, vocabSize); logits[:, t] predicts token t + 1
        return input

    def backward(self, gradient):
        # gradient : (batch, seqLen, vocabSize)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

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

    def zeroGradient(self):
        for grad, _ in self.gradients():
            grad[...] = 0