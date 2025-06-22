"""
2D convolutional layer implementation.
"""

import random
from ...core import Tensor

class Conv2D:
    """
    2D convolutional layer.
    
    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Padding added to input
        weight: Learnable weights
        bias: Learnable bias
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        Initialize a 2D convolutional layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel (int or tuple)
            stride: Stride of the convolution
            padding: Padding added to input
            bias: If set to False, the layer will not learn an additive bias
        """
        # TODO: Parse kernel_size to tuple if it's an int
        # TODO: Initialize weights with appropriate random initialization
        # TODO: Initialize bias if needed
        
    def __call__(self, x):
        """
        Forward pass of the convolutional layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        # TODO: Implement 2D convolution
        # TODO: Add padding if needed
        # TODO: Perform the convolution operation
        # TODO: Add bias if present
        
    def parameters(self):
        """
        Get the layer's learnable parameters.
        
        Returns:
            List of parameters (weight and bias)
        """
        # TODO: Return weight and bias as a list of parameters 