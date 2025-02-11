"""
Batch normalization implementations.
"""

from typing import Optional

from .modules import Module
from .. import Variable, Tensor, _C

class BatchNorm2d(Module):
    """Applies Batch Normalization over a 4D input."""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Variable(Tensor.ones((num_features,)), requires_grad=True)
            self.bias = Variable(Tensor.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', Tensor.zeros((num_features,)))
            self.register_buffer('running_var', Tensor.ones((num_features,)))
            self.register_buffer('num_batches_tracked', Tensor.zeros((), dtype='int64'))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, input: Variable) -> Variable:
        if self.training or not self.track_running_stats:
            exponential_average_factor = 0.0
            if self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0
        
        return _C.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum if self.momentum is not None else 0.0,
            self.affine,
            self.track_running_stats
        )(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training or not self.track_running_stats,
            exponential_average_factor
        )
    
    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, '
            f'eps={self.eps}, '
            f'momentum={self.momentum}, '
            f'affine={self.affine}, '
            f'track_running_stats={self.track_running_stats}'
        ) 