"""
Weight initialization utility functions.
"""

import random
import math
from ..core import Tensor

def zeros(shape):
    """
    Initialize tensor with zeros.
    
    Args:
        shape: Shape of the tensor
        
    Returns:
        Tensor filled with zeros
    """
    return Tensor(data=[0] * math.prod(shape), shape=shape, requires_grad=False)
    
def ones(shape):
    """
    Initialize tensor with ones.
    
    Args:
        shape: Shape of the tensor
        
    Returns:
        Tensor filled with ones
    """
    return Tensor(data=[1] * math.prod(shape), shape=shape, requires_grad=False)
    
def uniform(shape, a=0, b=1):
    """
    Initialize tensor with values drawn from a uniform distribution U(a, b).
    
    Args:
        shape: Shape of the tensor
        a: Lower bound of the distribution
        b: Upper bound of the distribution
        
    Returns:
        Tensor with uniform random values
    """
    return Tensor(data=[random.uniform(a, b) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)
    
def normal(shape, mean=0, std=1):
    """
    Initialize tensor with values drawn from a normal distribution N(mean, std).
    
    Args:
        shape: Shape of the tensor
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        
    Returns:
        Tensor with normal random values
    """
    return Tensor(data=[random.gauss(mean, std) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)
    
def xavier_uniform(shape, input_dim, output_dim):
    """
    Initialize tensor with values according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks" - Xavier initialization.
    
    Args:
        shape: Shape of the tensor
        input_dim: Dimension of input layer
        output_dim: Dimension of output layer
    Returns:
        Tensor initialized with Xavier uniform initialization
    """
    bound = math.sqrt(6 / (input_dim + output_dim))
    return Tensor(data=[random.uniform(-bound, bound) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)
    
def xavier_normal(shape, input_dim, output_dim):
    """
    Initialize tensor with values according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks" - Xavier initialization.
    
    Args:
        shape: Shape of the tensor
        
    Returns:
        Tensor initialized with Xavier normal initialization
    """
    std = math.sqrt(2 / (input_dim + output_dim))
    return Tensor(data=[random.gauss(0, std) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)
    
def kaiming_uniform(shape, gain=1.0, nonlinearity='relu'):
    """
    Initialize tensor with values according to the method described in
    "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He initialization.
    
    Args:
        shape: Shape of the tensor
        gain: Optional scaling factor
        nonlinearity: The non-linear function
        
    Returns:
        Tensor initialized with Kaiming uniform initialization
    """
    fan_in = shape[0]
    bound = math.sqrt(6 / (fan_in * gain))
    return Tensor(data=[random.uniform(-bound, bound) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)
    
def kaiming_normal(shape, gain=1.0, nonlinearity='relu'):
    """
    Initialize tensor with values according to the method described in
    "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" - He initialization.
    
    Args:
        shape: Shape of the tensor
        gain: Optional scaling factor
        nonlinearity: The non-linear function
        
    Returns:
        Tensor initialized with Kaiming normal initialization
    """
    fan_in = shape[0]
    std = math.sqrt(2 / fan_in)
    return Tensor(data=[random.gauss(0, std) for _ in range(math.prod(shape))], shape=shape, requires_grad=False)