"""
Base Module class for all neural network layers.
"""

from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple, Union

from .. import Variable, Tensor

class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._parameters: Dict[str, Variable] = OrderedDict()
        self._buffers: Dict[str, Tensor] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self.training: bool = True
    
    def register_parameter(self, name: str, param: Optional[Variable]) -> None:
        """Add a parameter to the module."""
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param
    
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Add a persistent buffer to the module."""
        if tensor is None:
            self._buffers[name] = None
        else:
            self._buffers[name] = tensor
    
    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Add a child module to the module."""
        if module is not None:
            self._modules[name] = module
    
    def parameters(self) -> Iterator[Variable]:
        """Returns an iterator over module parameters."""
        for param in self._parameters.values():
            if param is not None:
                yield param
        for module in self._modules.values():
            for param in module.parameters():
                yield param
    
    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def zero_grad(self) -> None:
        """Zero out the gradients of all parameters."""
        for param in self.parameters():
            if param.grad is not None:
                param.zero_grad()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass of the module. Should be overridden by subclasses."""
        raise NotImplementedError
    
    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        """Custom setattr to handle parameters and modules."""
        if isinstance(value, Variable):
            self.register_parameter(name, value)
        elif isinstance(value, Tensor):
            self.register_buffer(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") 