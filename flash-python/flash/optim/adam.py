"""
Adam and AdamW optimizer implementations.
"""

from typing import Dict, Iterable, Optional
import math

from .optimizer import Optimizer
from .. import Variable, Tensor

class Adam(Optimizer):
    """Implements Adam algorithm.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use the AMSGrad variant (default: False)
    """
    
    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                amsgrad = group['amsgrad']
                
                state = self.state[param]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = Tensor.zeros_like(param.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = Tensor.zeros_like(param.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = Tensor.zeros_like(param.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * param.data
                
                # Decay the first and second moment running average coefficient
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    max_exp_avg_sq = max_exp_avg_sq.maximum(exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq / bias_correction2).sqrt() + group['eps']
                else:
                    denom = (exp_avg_sq / bias_correction2).sqrt() + group['eps']
                
                step_size = group['lr'] / bias_correction1
                param.data = param.data - step_size * exp_avg / denom
                
                # Update state
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                if amsgrad:
                    state['max_exp_avg_sq'] = max_exp_avg_sq

class AdamW(Optimizer):
    """Implements AdamW algorithm.
    
    The original Adam algorithm was proposed in
    `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in
    `Decoupled Weight Decay Regularization`_.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 1e-2)
        amsgrad: whether to use the AMSGrad variant (default: False)
    """
    
    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                amsgrad = group['amsgrad']
                
                state = self.state[param]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = Tensor.zeros_like(param.data)
                    state['exp_avg_sq'] = Tensor.zeros_like(param.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = Tensor.zeros_like(param.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
                
                if amsgrad:
                    max_exp_avg_sq = max_exp_avg_sq.maximum(exp_avg_sq)
                    denom = (max_exp_avg_sq / bias_correction2).sqrt() + group['eps']
                else:
                    denom = (exp_avg_sq / bias_correction2).sqrt() + group['eps']
                
                step_size = group['lr'] / bias_correction1
                
                # AdamW: Perform stepweight decay at each step
                param.data = param.data * (1 - group['lr'] * group['weight_decay'])
                param.data = param.data - step_size * exp_avg / denom
                
                # Update state
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                if amsgrad:
                    state['max_exp_avg_sq'] = max_exp_avg_sq 