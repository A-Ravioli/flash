"""
Multi-layer Perceptron implementation.
"""

from ..layers import Linear
from ..activations import ReLU, Sigmoid

class MLP:
    """
    Multi-layer Perceptron model with configurable hidden layers.
    
    Attributes:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes
        output_size: Size of the output
        layers: List of layer modules
        activations: List of activation functions
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        Initialize an MLP.
        
        Args:
            input_size: Size of the input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of the output
            activation: Activation function to use ('relu' or 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fn = None

        # TODO: Set up activation function based on activation arg
        match activation.lower():
            case 'relu':
                activation_fn = ReLU()
            case 'sigmoid':
                activation_fn = Sigmoid()

        # TODO: Build layers
        self.layers = [
            Linear(3, 5),
            Linear(5, 5),
            Linear(5, 3)
        ]

        # TODO: Create list of layers with appropriate dimensions
        # TODO: Create list of activation functions
        self.activations = [activation_fn]
        
    def __call__(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass through layers and activations
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers 