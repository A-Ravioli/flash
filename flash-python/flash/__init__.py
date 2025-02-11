"""
flash: A hardware-optimized ML framework with Python bindings
"""

from . import _C
from ._C import (
    Tensor,
    Variable,
    Module,
    Linear,
    Conv2d,
    BatchNorm2d,
    ReLU,
    SGD,
    Adam,
)

# Version information
__version__ = "0.1.0"

def tensor(data, dtype=None, device=None):
    """
    Creates a flash Tensor from a Python object.
    
    Args:
        data: Array-like object (list, tuple, NumPy array, etc.)
        dtype: Data type (optional)
        device: Device to place the tensor on (optional)
    
    Returns:
        flash.Tensor: A new tensor
    """
    import numpy as np
    
    if isinstance(data, Tensor):
        return data.clone()
    
    # Convert to NumPy array
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    elif dtype is None and data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Create tensor
    t = _C.tensor(data)
    
    # Move to device if specified
    if device is not None:
        if device == "cuda":
            t = t.cuda()
        elif device == "cpu":
            t = t.cpu()
        else:
            raise ValueError(f"Unknown device: {device}")
    
    return t

def cuda_is_available():
    """
    Returns True if CUDA is available.
    """
    try:
        t = tensor([1.0]).cuda()
        return True
    except RuntimeError:
        return False

def device(device_str):
    """
    Returns a device object.
    
    Args:
        device_str: String specifying the device ("cpu" or "cuda")
    
    Returns:
        Device object
    """
    if device_str not in ["cpu", "cuda"]:
        raise ValueError(f"Unknown device: {device_str}")
    return _C.device.DeviceType.CPU if device_str == "cpu" else _C.device.DeviceType.CUDA

# Import submodules
from . import nn
from . import optim
from . import distributed

# Clean up namespace
del _C 