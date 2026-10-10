import numpy as np
from Layer import Layer

class Softmax(Layer):
    def forward(self, input, axis=-1):
        self.axis = axis
        ex = np.exp(input - np.max(input, axis=axis, keepdims=True))
        self.output = ex / np.sum(ex, axis=axis, keepdims=True)
        return self.output

    def backward(self, dE_dY):
        output = self.output
        dE_dX = output * (dE_dY - np.sum(output * dE_dY, axis=self.axis, keepdims=True))
        return dE_dX
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []