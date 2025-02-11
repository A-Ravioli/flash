"""
Functional interface for neural network operations.
"""

from typing import Optional, Tuple, Union

from .. import Variable, Tensor, _C

def linear(input: Variable, weight: Variable, bias: Optional[Variable] = None) -> Variable:
    """Applies a linear transformation to the incoming data: y = xA^T + b"""
    return _C.Linear(weight.shape[1], weight.shape[0], bias is not None)(input, weight, bias)

def conv2d(
    input: Variable,
    weight: Variable,
    bias: Optional[Variable] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0
) -> Variable:
    """Applies a 2D convolution over an input signal composed of several input planes."""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    return _C.Conv2d(
        weight.shape[1],
        weight.shape[0],
        weight.shape[2],
        stride[0],
        padding[0],
        bias is not None
    )(input, weight, bias)

def batch_norm2d(
    input: Variable,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Variable] = None,
    bias: Optional[Variable] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> Variable:
    """Applies Batch Normalization over a 4D input."""
    if training:
        exponential_average_factor = momentum
    else:
        exponential_average_factor = 0.0
    
    return _C.BatchNorm2d(
        input.shape[1],
        eps,
        momentum,
        weight is not None,
        running_mean is not None
    )(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor
    )

def relu(input: Variable) -> Variable:
    """Applies the rectified linear unit function element-wise."""
    return _C.ReLU()(input)

def sigmoid(input: Variable) -> Variable:
    """Applies the sigmoid function element-wise."""
    return _C.Sigmoid()(input)

def tanh(input: Variable) -> Variable:
    """Applies the hyperbolic tangent function element-wise."""
    return _C.Tanh()(input)

def leaky_relu(input: Variable, negative_slope: float = 0.01) -> Variable:
    """Applies the leaky rectified linear unit function element-wise."""
    return _C.LeakyReLU(negative_slope)(input)

def cross_entropy(input: Variable, target: Variable, reduction: str = 'mean') -> Variable:
    """Computes cross entropy loss between input and target."""
    return _C.CrossEntropy(reduction)(input, target)

def dropout(input: Variable, p: float = 0.5, training: bool = True) -> Variable:
    """Randomly zeroes some of the elements of the input tensor with probability p."""
    if not training or p == 0:
        return input
    return _C.Dropout(p, training)(input) 