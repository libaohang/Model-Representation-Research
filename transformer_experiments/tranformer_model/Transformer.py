from Layer import Layer
from Attention import MultiHeadAttention
from LayerNorm import LayerNorm
from MultiLayerPerceptron import TransformerMLP
from Residual import Residual

# Pre-LN decoder-only block as in GPT-2; input and output are both (batch, seqLen, modelDim)
class TransformerBlock(Layer):
    def __init__(self, modelDim, numHeads, mlpWidth, activation):
        # residual(layernorm -> causal multi head attention)
        ln1 = LayerNorm(modelDim)
        mha = MultiHeadAttention(modelDim, numHeads)
        subnet1 = [ln1, mha]
        self.residual1 = Residual(subnet1)

        # residual(layernorm -> mlp)
        ln2 = LayerNorm(modelDim)
        mlp = TransformerMLP(modelDim, mlpWidth, activation)
        subnet2 = [ln2, mlp]
        self.residual2 = Residual(subnet2)

        self.layers = [self.residual1, self.residual2]
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, gradient):
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
    