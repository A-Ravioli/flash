"""
Linear layer implementation.
"""

import random
from ...core import Tensor
from ...utils import xavier_uniform, zeros

class Linear:
    """
    Fully connected layer implementing y = Wx + b transformation.
    
    Attributes:
        in_features: Size of each input sample
        out_features: Size of each output sample
        weight: Learnable weights
        bias: Learnable bias
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize a linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        # TODO: Initialize weight with appropriate random initialization
        self.weight = xavier_uniform((in_features, out_features))
        # TODO: Initialize bias if needed
        self.bias = zeros((out_features)) if bias else None

    def __call__(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # TODO: Implement the linear transformation y = Wx + b
        return self.weight @ x + self.bias
    
    def parameters(self):
        """
        Get the layer's learnable parameters.
        
        Returns:
            List of parameters (weight and bias)
        """
        # TODO: Return weight and bias as a list of parameters if bias is True, otherwise return only weight
        return [self.weight] if self.bias is None else [self.weight, self.bias]