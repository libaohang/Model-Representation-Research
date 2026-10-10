from Layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, inputDim, outputDim):
        self.weights = (np.random.randn(outputDim, inputDim)  * np.sqrt(1 / (inputDim))).astype(np.float32)
        # Add axis so bias broadcasts in the batch dimension
        self.bias = np.zeros((1, outputDim), dtype=np.float32)
    
    def forward(self, input):
        self.input = input

        # Weight each input dimension by the corresponding weight dimension and add bias
        return input @ self.weights.T + self.bias
    
    def backward(self, dE_dY):
        x = self.input  # (B, Din) or (B, T, Din)

        if x.ndim == 2:
            # x: (B, Din), dE_dY: (B, Dout)
            self.dE_dW = dE_dY.T @ x                 # (Dout, Din)
            self.dE_dB = np.sum(dE_dY, axis=0, keepdims=True)  # (1, Dout)
            dE_dX = dE_dY @ self.weights             # (B, Din)
            return dE_dX

        elif x.ndim == 3:
            # x: (B, T, Din), dE_dY: (B, T, Dout)
            self.dE_dW = np.sum(dE_dY.transpose(0, 2, 1) @ x, axis=0)  # (Dout, Din)
            self.dE_dB = np.sum(dE_dY, axis=(0, 1)).reshape(1, -1)     # (1, Dout)
            dE_dX = dE_dY @ self.weights                               # (B, T, Din)
            return dE_dX
    
    def parameters(self):
        return [(self.weights, "weight"),(self.bias, "bias")]
    
    def gradients(self):
        return [(self.dE_dW, "weight"), (self.dE_dB, "bias")]