import math

"""
Core tensor operations implemented from scratch.
"""

def dot(a, b):
    """
    Compute the dot product of two vectors.
    
    Args:
        a: First vector (Tensor)
        b: Second vector (Tensor)
        
    Returns:
        A scalar Tensor containing the dot product
    """
    return a.dot(b)

def matmul(a, b):
    """
    Perform matrix multiplication.
    
    Args:
        a: First tensor of shape (m, n)
        b: Second tensor of shape (n, p)
        
    Returns:
        Resulting tensor of shape (m, p)
    """
    return a.matmul(b)

def transpose(x, dim0=0, dim1=1):
    """
    Swap two dimensions of a tensor.
    
    Args:
        x: Input tensor
        dim0, dim1: Dimensions to swap
        
    Returns:
        Transposed tensor
    """
    return x.transpose(dim0, dim1)

def reshape(x, shape):
    """
    Reshape a tensor to a new shape.
    
    Args:
        x: Input tensor
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    return x.reshape(shape)

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with ReLU applied element-wise
    """
    return max(0, x)

def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sigmoid applied element-wise
    """
    return 1 / (1 + math.exp(-x))

def softmax(x, dim=-1):
    """
    Softmax normalization function.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Tensor with softmax applied along specified dimension
    """
    return math.exp(x) / sum(math.exp(x))

def cross_entropy(logits, targets):
    """
    Cross entropy loss function.
    
    Args:
        logits: Predicted logits
        targets: Target labels
        
    Returns:
        Cross entropy loss as a scalar tensor
    """
    return -sum(targets * math.log(logits))
