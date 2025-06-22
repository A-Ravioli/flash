"""
Optimization algorithm implementations.
"""

import math
from ...utils import zeros

class Optimizer:
    """
    Base class for all optimizers.
    """
    
    def __init__(self, parameters, lr):
        """
        Initialize an optimizer.
        
        Args:
            parameters: Iterable of parameters to optimize
            lr: Learning rate
        """
        self.parameters = list(parameters)
        self.lr = lr
        
    def zero_grad(self):
        """
        Zero out the gradients of all parameters.
        """
        # TODO: Set gradient to zero for all parameters
        
    def step(self):
        """
        Update parameters based on current gradients.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Implements:
        w = w - lr * dw
    """
    
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        """
        Initialize SGD optimizer.
        
        Args:
            parameters: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # TODO: Initialize velocity if momentum > 0
        self.velocity = [zeros(param.shape) for param in self.parameters] if momentum > 0 else None
    def step(self):
        """
        Perform a single optimization step.
        """
        # TODO: Implement SGD update
        for i, param in enumerate(self.parameters):
            if self.velocity is not None:
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
                param.data += self.velocity[i]
            else:
                param.data -= self.lr * param.grad
        # TODO: Include momentum if requested
        # TODO: Include weight decay if requested



class Adam(Optimizer):
    """
    Adam optimizer.
    
    Implements the Adam algorithm as described in "Adam: A Method for Stochastic Optimization".
    """
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Initialize Adam optimizer.
        
        Args:
            parameters: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # TODO: Initialize moment estimates
        self.step_count = 0
        
    def step(self):
        """
        Perform a single optimization step.
        """
        # TODO: Implement Adam update
        # TODO: Update first and second moment estimates
        # TODO: Apply bias correction
        # TODO: Update parameters
        self.step_count += 1
        for i, param in enumerate(self.parameters):
            self.first_moment[i] = self.betas[0] * self.first_moment[i] + (1 - self.betas[0]) * param.grad
            self.second_moment[i] = self.betas[1] * self.second_moment[i] + (1 - self.betas[1]) * param.grad ** 2
            first_moment_corrected = self.first_moment[i] / (1 - self.betas[0] ** self.step_count)
            second_moment_corrected = self.second_moment[i] / (1 - self.betas[1] ** self.step_count)
            param.data -= self.lr * first_moment_corrected / (math.sqrt(second_moment_corrected) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Implements RMSprop as described by Hinton.
    """
    
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0):
        """
        Initialize RMSprop optimizer.
        
        Args:
            parameters: Iterable of parameters to optimize
            lr: Learning rate
            alpha: Smoothing constant
            eps: Term added to denominator for numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        # TODO: Initialize square gradient average
        self.square_gradient_average = [zeros(param.shape) for param in self.parameters]
        
    def step(self):
        """
        Perform a single optimization step.
        """
        # TODO: Implement RMSprop update
        # TODO: Update square gradient running average
        # TODO: Update parameters 
        for i, param in enumerate(self.parameters):
            self.square_gradient_average[i] = self.alpha * self.square_gradient_average[i] + (1 - self.alpha) * param.grad ** 2
            param.data -= self.lr * param.grad / (math.sqrt(self.square_gradient_average[i]) + self.eps)
            param.data -= self.lr * param.grad
            param.data -= self.weight_decay * param.data
