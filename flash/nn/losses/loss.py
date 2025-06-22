"""
Loss function implementations.
"""

import math
from ...core import Tensor

class MSELoss:
    """
    Mean Squared Error loss function.
    
    L(y, y_hat) = mean((y - y_hat)^2)
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize MSE loss.
        
        Args:
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.reduction = reduction
        
    def __call__(self, pred, target):
        """
        Compute MSE loss.
        
        Args:
            pred: Predictions tensor
            target: Target tensor
            
        Returns:
            Loss value
        """
        # TODO: Implement mean squared error loss
        # TODO: Apply reduction
        return (pred - target).pow(2).mean() if self.reduction == 'mean' else (pred - target).pow(2).sum() if self.reduction == 'sum' else pred - target
    
    def parameters(self):
        """
        Get the loss's learnable parameters.
        
        Returns:
            Empty list (no parameters for MSELoss)
        """
        return []


class CrossEntropyLoss:
    """
    Cross Entropy loss combining softmax and negative log likelihood loss.
    
    L(y, y_hat) = -sum(y * log(softmax(y_hat)))
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize cross entropy loss.
        
        Args:
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.reduction = reduction
        
    def __call__(self, logits, targets):
        """
        Compute cross entropy loss.
        
        Args:
            logits: Raw logits tensor (not softmax)
            targets: Target tensor (class indices or one-hot)
            
        Returns:
            Loss value
        """
        # TODO: Implement cross entropy loss
        # TODO: Apply softmax to logits
        # TODO: Handle class indices or one-hot targets
        # TODO: Apply reduction
        return -logits[targets].mean() if self.reduction == 'mean' else -logits[targets].sum() if self.reduction == 'sum' else -logits[targets]
    
    def parameters(self):
        """
        Get the loss's learnable parameters.
        
        Returns:
            Empty list (no parameters for CrossEntropyLoss)
        """
        return []


class BCELoss:
    """
    Binary Cross Entropy loss function.
    
    L(y, y_hat) = -mean(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize BCE loss.
        
        Args:
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
        """
        self.reduction = reduction
        
    def __call__(self, pred, target):
        """
        Compute BCE loss.
        
        Args:
            pred: Predictions tensor (values between 0 and 1)
            target: Target tensor (binary values)
            
        Returns:
            Loss value
        """
        # TODO: Implement binary cross entropy loss
        # TODO: Handle numerical stability
        # TODO: Apply reduction
        return -target * math.log(pred) - (1 - target) * math.log(1 - pred)
        
    def parameters(self):
        """
        Get the loss's learnable parameters.
        
        Returns:
            Empty list (no parameters for BCELoss)
        """
        return [] 