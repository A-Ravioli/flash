"""
Gradient function implementations for autodiff.
"""

class Function:
    """
    Base class for all autograd functions.
    
    Each subclass implements forward and backward methods.
    """
    
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Performs the forward computation.
        
        Args:
            ctx: Context object to save information for backward
            *args: Input tensors
            **kwargs: Additional arguments
            
        Returns:
            Output tensor(s)
        """
        
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient of the forward function.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradient with respect to each input in forward
        """
        raise NotImplementedError


class AddFunction(Function):
    """
    Function for addition operation.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for addition.
        
        Args:
            ctx: Context object
            a, b: Input tensors
            
        Returns:
            Sum of a and b
        """
        # TODO: Implement addition forward pass
        # TODO: Save any values needed for backward pass
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for addition.
        
        Args:
            ctx: Context object
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradient with respect to a and b
        """
        # TODO: Implement addition backward pass


class MulFunction(Function):
    """
    Function for element-wise multiplication operation.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for multiplication.
        
        Args:
            ctx: Context object
            a, b: Input tensors
            
        Returns:
            Element-wise product of a and b
        """
        # TODO: Implement multiplication forward pass
        # TODO: Save any values needed for backward pass
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for multiplication.
        
        Args:
            ctx: Context object
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradient with respect to a and b
        """
        # TODO: Implement multiplication backward pass


class MatMulFunction(Function):
    """
    Function for matrix multiplication.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for matrix multiplication.
        
        Args:
            ctx: Context object
            a, b: Input tensors
            
        Returns:
            Matrix product of a and b
        """
        # TODO: Implement matrix multiplication forward pass
        # TODO: Save any values needed for backward pass
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for matrix multiplication.
        
        Args:
            ctx: Context object
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradient with respect to a and b
        """
        # TODO: Implement matrix multiplication backward pass


class ReLUFunction(Function):
    """
    Function for ReLU activation.
    """
    
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for ReLU.
        
        Args:
            ctx: Context object
            x: Input tensor
            
        Returns:
            ReLU of x
        """
        # TODO: Implement ReLU forward pass
        # TODO: Save any values needed for backward pass
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for ReLU.
        
        Args:
            ctx: Context object
            grad_output: Gradient from upstream operations
            
        Returns:
            Gradient with respect to x
        """
        # TODO: Implement ReLU backward pass 