"""
Tensor implementation from scratch.
"""

class Tensor:
    """
    A multi-dimensional array implementation with automatic differentiation capabilities.
    
    This is the core data structure for the Flash library, similar to numpy.ndarray
    or torch.Tensor, but implemented from scratch.
    
    Attributes:
        data: The actual tensor data (will be a nested list or similar)
        grad: Gradient for this tensor (also a Tensor)
        _op: The operation that created this tensor (for backpropagation)
        _backward_fn: Function to run during backpropagation
        requires_grad: Whether this tensor participates in autodiff
        shape: The shape of the tensor as a tuple
    """
    
    def __init__(self, data, requires_grad=False, _op=None, _backward_fn=None):
        """
        Initialize a new Tensor object.
        
        Args:
            data: The tensor data (list, int, float, etc.)
            requires_grad: Whether this tensor requires gradient computation
            _op: The operation that created this tensor (internal use)
            _backward_fn: Function called during backward pass (internal use)
        """
        self.data = data
        self.grad = None
        self._op = _op
        self._backward_fn = _backward_fn
        self.requires_grad = requires_grad
        self.shape = tuple(data) if isinstance(data, (list, tuple)) else (data,)
        
    def __repr__(self):
        """
        String representation of the Tensor.
        """
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad})"
        
    def backward(self, grad=None):
        """
        Compute gradients via backpropagation.
        
        Args:
            grad: Gradient from upstream operations
        """
        if self._backward_fn is None:
            raise RuntimeError("Backward function not set for this tensor")
        self._backward_fn(self, grad)
        
    def __add__(self, other):
        """
        Element-wise addition.
        
        Args:
            other: Another tensor or scalar
            
        Returns:
            A new Tensor with the result
        """
        return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
    def __mul__(self, other):
        """
        Element-wise multiplication.
        
        Args:
            other: Another tensor or scalar
            
        Returns:
            A new Tensor with the result
        """
        return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
    def __matmul__(self, other):
        """
        Matrix multiplication.
        
        Args:
            other: Another tensor
            
        Returns:
            A new Tensor with the result
        """
        return Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        
    def dot(self, other):
        """
        Dot product of two tensors.
        """
        return Tensor(self.data.dot(other.data), requires_grad=self.requires_grad or other.requires_grad)
    
    def sum(self, dim=None):
        """
        Sum elements of the tensor along a dimension.
        
        Args:
            dim: Dimension to sum along
            
        Returns:
            A new Tensor with the result
        """
        return Tensor(self.data.sum(dim), requires_grad=self.requires_grad)
        
    def reshape(self, shape):
        """
        Reshape the tensor to a new shape.
        
        Args:
            shape: New shape
            
        Returns:
            A new reshaped Tensor
        """
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
    
    def transpose(self, dim0=0, dim1=1):
        """
        Swap two dimensions of the tensor.
        
        Args:
            dim0, dim1: Dimensions to swap
            
        Returns:
            A new transposed Tensor
        """
        return Tensor(self.data.transpose(dim0, dim1), requires_grad=self.requires_grad)
    
    def __getitem__(self, index):
        """
        Get an element or slice of the tensor.
        """
        return Tensor(self.data[index], requires_grad=self.requires_grad)
    
    def __setitem__(self, index, value):
        """
        Set an element or slice of the tensor.
        """
        self.data[index] = value
    
    def __len__(self):
        """
        Get the length of the tensor.
        """
        return len(self.data)
    
    def __iter__(self):
        """
        Iterate over the tensor.
        """
        return iter(self.data)
    
    def __contains__(self, item):
        """
        Check if an item is in the tensor.
        """
        return item in self.data