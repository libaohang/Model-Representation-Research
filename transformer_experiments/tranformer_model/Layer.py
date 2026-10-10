from abc import ABC, abstractmethod

# Don't necessarily need to have an abstract base class; could simply rely on duck typing; mainly for practice/experimentation
class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dx):
        pass
    
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def gradients(self):
        pass