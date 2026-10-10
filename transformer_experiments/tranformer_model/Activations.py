from Layer import Layer
import numpy as np

class ReLU(Layer):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, dE_dY):
        return (self.input > 0) * dE_dY
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []
    
class GELU(Layer):
    def forward(self, x):
        self.input = x
        return 0.5 * x * (1 + np.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))
    
    def backward(self, dE_dY):
        x = self.input
        param = np.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * (x ** 3)))
        dGELU_dX = (0.5 * (1 + param) + 
                    0.5 * x * (1 - param ** 2) * ((2 / np.pi) ** 0.5) * (1 + 3 * 0.044715 * (x ** 2)))
        dE_dX = dGELU_dX * dE_dY
        return dE_dX
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []