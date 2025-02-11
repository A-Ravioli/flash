"""
Linear layer implementation.
"""

from typing import Optional

from .modules import Module
from .. import Variable, Tensor, _C
from ..init import kaiming_uniform

class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization
        self.weight = Variable(
            kaiming_uniform(out_features, in_features),
            requires_grad=True
        )
        
        if bias:
            self.bias = Variable(
                Tensor.zeros((out_features,)),
                requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: Variable) -> Variable:
        return _C.Linear(self.in_features, self.out_features, self.bias is not None)(
            input, self.weight, self.bias
        )
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}' 