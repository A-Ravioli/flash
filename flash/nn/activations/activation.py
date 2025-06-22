"""
Activation function implementations.
"""

import math
from ...core import Tensor

class Activation:
    """
    Base class for all activation functions.
    
    Each subclass should implement __call__ and parameters methods.
    """
    
    def __call__(self, x):
        """
        Apply the activation function to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying activation
        """
        raise NotImplementedError
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            List of learnable parameters (empty for most activations)
        """
        return []


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    
    f(x) = max(0, x)
    """
    
    def __call__(self, x):
        """
        Apply ReLU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying ReLU
        """
        return Tensor(data=[max(0, i) for i in x.data], shape=x.shape, requires_grad=x.requires_grad)
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            Empty list (no parameters for ReLU)
        """
        return []


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    f(x) = 1 / (1 + exp(-x))
    """
    
    def __call__(self, x):
        """
        Apply sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying sigmoid
        """
        return Tensor(data=[1 / (1 + math.exp(-i)) for i in x.data], shape=x.shape, requires_grad=x.requires_grad)
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            Empty list (no parameters for Sigmoid)
        """
        return []


class Tanh(Activation):
    """
    Hyperbolic tangent activation function.
    
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def __call__(self, x):
        """
        Apply tanh activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying tanh
        """
        return Tensor(data=[math.tanh(i) for i in x.data], shape=x.shape, requires_grad=x.requires_grad)
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            Empty list (no parameters for Tanh)
        """
        return []


class Softmax:
    """
    Softmax activation function.
    
    f(x)_i = exp(x_i) / sum(exp(x_j)) for j in range(len(x))
    """
    
    def __init__(self, dim=-1):
        """
        Initialize a softmax activation.
        
        Args:
            dim: Dimension along which to apply softmax
        """
        self.dim = dim
        
    def __call__(self, x):
        """
        Apply softmax activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying softmax
        """
        # TODO: Implement softmax activation function
        # TODO: Maintain numerical stability using max subtraction technique
        max_val = x.max(dim=self.dim)
        exp_x = math.exp(x - max_val)
        return Tensor(data=[i / exp_x.sum(dim=self.dim) for i in exp_x.data], shape=x.shape, requires_grad=x.requires_grad)
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            Empty list (no parameters for Softmax)
        """
        return []


class GELU:
    """
    Gaussian Error Linear Unit activation function.
    
    f(x) = x * Phi(x) where Phi is the CDF of the standard normal distribution
    """
    
    def __call__(self, x):
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying GELU
        """
        # TODO: Implement GELU activation function
        return Tensor(data=[0.5 * i * (1 + math.tanh(math.sqrt(2 / math.pi) * (i + 0.044715 * i**3))) for i in x.data], shape=x.shape, requires_grad=x.requires_grad)
        # Note: Can use approximation 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
    def parameters(self):
        """
        Get the activation's learnable parameters.
        
        Returns:
            Empty list (no parameters for GELU)
        """
        return [] 