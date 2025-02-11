"""
Convolutional layer implementations.
"""

from typing import Optional, Tuple, Union

from .modules import Module
from .. import Variable, Tensor, _C
from ..init import kaiming_uniform

class Conv2d(Module):
    """Applies a 2D convolution over an input signal composed of several input planes."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights using Kaiming initialization
        self.weight = Variable(
            kaiming_uniform(
                out_channels,
                in_channels * self.kernel_size[0] * self.kernel_size[1]
            ).reshape(out_channels, in_channels, *self.kernel_size),
            requires_grad=True
        )
        
        if bias:
            self.bias = Variable(
                Tensor.zeros((out_channels,)),
                requires_grad=True
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: Variable) -> Variable:
        return _C.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            self.bias is not None
        )(input, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        s = (f'in_channels={self.in_channels}, '
             f'out_channels={self.out_channels}, '
             f'kernel_size={self.kernel_size}, '
             f'stride={self.stride}, '
             f'padding={self.padding}')
        if self.bias is None:
            s += ', bias=False'
        return s 