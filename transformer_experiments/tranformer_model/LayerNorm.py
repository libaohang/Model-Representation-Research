from Layer import Layer
import numpy as np

class LayerNorm(Layer):
    def __init__(self, modelDim, eps=1e-5):
        self.eps = eps
        self.scale = np.ones(modelDim).astype(np.float32)
        self.shift = np.zeros(modelDim).astype(np.float32)

    # normalize over last dimension (modelDim)
    def forward(self, input):
        self.mean = input.mean(axis=-1, keepdims=True)
        self.variance = input.var(axis=-1, keepdims=True)
        self.inputNorm = (input - self.mean) / np.sqrt(self.variance + self.eps)
        return self.inputNorm * self.scale + self.shift
    
    def backward(self, dE_dY):
        self.dE_dShift = np.sum(dE_dY, axis=(0,1))
        self.dE_dScale = np.sum(dE_dY * self.inputNorm, axis=(0,1))

        scaled = dE_dY * self.scale
        dE_dX = ((scaled - scaled.mean(axis=-1, keepdims=True) -
                 self.inputNorm * (scaled * self.inputNorm).mean(axis=-1, keepdims=True))
                 / np.sqrt(self.variance + self.eps))
        
        return dE_dX
    
    def parameters(self):
        return [(self.scale, "ln sc"), (self.shift, "ln sh")]
    
    def gradients(self):
        return [(self.dE_dScale, "ln sc"), (self.dE_dShift, "ln sh")]