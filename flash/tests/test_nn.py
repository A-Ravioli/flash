"""
Tests for the Neural Network components.
"""

import unittest
from flash.core import Tensor
from flash.nn.layers import Linear, Conv2D
from flash.nn.activations import ReLU, Sigmoid
from flash.nn.models import MLP

class TestLayers(unittest.TestCase):
    """
    Test cases for the neural network layers.
    """
    
    def test_linear(self):
        """
        Test Linear layer.
        """
        # TODO: Test linear layer forward pass
        # TODO: Test linear layer backward pass
        # TODO: Test linear layer parameters
        
    def test_conv2d(self):
        """
        Test Conv2D layer.
        """
        # TODO: Test conv2d layer forward pass
        # TODO: Test conv2d layer backward pass
        # TODO: Test conv2d layer parameters


class TestActivations(unittest.TestCase):
    """
    Test cases for the activation functions.
    """
    
    def test_relu(self):
        """
        Test ReLU activation.
        """
        # TODO: Test ReLU forward pass
        # TODO: Test ReLU backward pass
        
    def test_sigmoid(self):
        """
        Test Sigmoid activation.
        """
        # TODO: Test Sigmoid forward pass
        # TODO: Test Sigmoid backward pass


class TestModels(unittest.TestCase):
    """
    Test cases for the neural network models.
    """
    
    def test_mlp(self):
        """
        Test MLP model.
        """
        # TODO: Test MLP creation
        # TODO: Test MLP forward pass
        # TODO: Test MLP parameter collection


if __name__ == "__main__":
    unittest.main() 