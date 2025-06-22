"""
Convolutional Neural Network implementation.
"""

from ..layers import Conv2D, Linear
from ..activations import ReLU

class CNN:
    """
    Convolutional Neural Network for image classification.
    
    Architecture:
    - Multiple convolutional layers with ReLU activation
    - Flatten operation
    - Multiple fully connected layers with ReLU activation
    - Output layer (logits)
    
    Attributes:
        conv_layers: List of convolutional layers
        fc_layers: List of fully connected layers
        activations: List of activation functions
    """
    
    def __init__(self, input_channels, input_height, input_width, conv_channels, 
                 conv_kernel_sizes, conv_strides, hidden_sizes, output_size):
        """
        Initialize a CNN.
        
        Args:
            input_channels: Number of input channels
            input_height: Height of input images
            input_width: Width of input images
            conv_channels: List of output channels for each conv layer
            conv_kernel_sizes: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            hidden_sizes: List of hidden layer sizes for FC layers
            output_size: Size of the output (number of classes)
        """
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # TODO: Build convolutional layers
        # TODO: Calculate output shape after convolutions
        # TODO: Build fully connected layers
        
    def flatten(self, x):
        """
        Flatten a tensor from (batch_size, channels, height, width) to (batch_size, channels*height*width).
        
        Args:
            x: Input tensor
            
        Returns:
            Flattened tensor
        """
        # TODO: Implement flatten operation
        
    def __call__(self, x):
        """
        Forward pass of the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, input_height, input_width)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass through conv layers with activations
        # TODO: Flatten the output
        # TODO: Pass through fully connected layers with activations
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers 