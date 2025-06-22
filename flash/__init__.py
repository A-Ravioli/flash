"""
Flash: A pure Python neural network and tensor library built from scratch.
"""

__version__ = "0.1.0"

from .core import Tensor
from .nn import Module, Linear, ReLU, Sigmoid, Softmax, CrossEntropyLoss
from .optim import SGD, Adam

