"""
Activation functions for neural networks.
"""

from typing import Optional

from .modules import Module
from .. import Variable, _C

class ReLU(Module):
    """Applies the rectified linear unit function element-wise."""
    
    def forward(self, input: Variable) -> Variable:
        return _C.ReLU()(input)

class Sigmoid(Module):
    """Applies the sigmoid function element-wise."""
    
    def forward(self, input: Variable) -> Variable:
        return _C.Sigmoid()(input)

class Tanh(Module):
    """Applies the hyperbolic tangent function element-wise."""
    
    def forward(self, input: Variable) -> Variable:
        return _C.Tanh()(input)

class LeakyReLU(Module):
    """Applies the leaky rectified linear unit function element-wise."""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, input: Variable) -> Variable:
        return _C.LeakyReLU(self.negative_slope)(input)

    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}' 