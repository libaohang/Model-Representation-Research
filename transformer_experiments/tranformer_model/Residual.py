from Layer import Layer

class Residual(Layer):
    def __init__(self, network):
        # A list of layer objects
        self.subnet = network

    def forward(self, tokens):
        # tokens : (batch, seqLen, modelDim)
        # Don't need to make a copy since no layers modify input
        input = tokens
        for layer in self.subnet:
            tokens = layer.forward(tokens)

        assert tokens.shape == input.shape
        return input + tokens
    
    def backward(self, dE_dY):
        dE_dSubnet = dE_dY
        for layer in reversed(self.subnet):
            dE_dSubnet = layer.backward(dE_dSubnet)
        return dE_dY + dE_dSubnet
    
    def parameters(self):
        parameters = []
        for layer in self.subnet:
            for parameter in layer.parameters():
                parameters.append(parameter)
        return parameters
        
    def gradients(self):
        gradients = []
        for layer in self.subnet:
            for gradient in layer.gradients():
                gradients.append(gradient)
        return gradients
    