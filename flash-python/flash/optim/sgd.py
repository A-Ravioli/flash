"""
Stochastic Gradient Descent optimizer implementation.
"""

from typing import Dict, Iterable, Optional

from .optimizer import Optimizer
from .. import Variable, Tensor

class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        dampening: dampening for momentum (default: 0)
        nesterov: enables Nesterov momentum (default: False)
    """
    
    def __init__(
        self,
        params: Iterable[Variable],
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super().__init__(params, defaults)
    
    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                grad = param.grad
                
                if weight_decay != 0:
                    grad = grad + weight_decay * param.data
                
                if momentum != 0:
                    param_state = self.state[param]
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = grad.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf = momentum * buf + (1 - dampening) * grad
                        param_state['momentum_buffer'] = buf
                    
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                
                param.data = param.data - group['lr'] * grad 