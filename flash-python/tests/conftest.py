"""
Pytest configuration and common fixtures.
"""

import os
import pytest
import numpy as np
import torch.distributed as dist

from flash import tensor

def is_distributed_test():
    """Check if we're running a distributed test."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

@pytest.fixture
def random_tensor():
    def _random_tensor(shape, requires_grad=False):
        return tensor(
            np.random.randn(*shape).astype(np.float32),
            requires_grad=requires_grad
        )
    return _random_tensor

@pytest.fixture
def cpu_device():
    return 'cpu'

@pytest.fixture
def cuda_device():
    return 'cuda'

@pytest.fixture
def gradient_check():
    def _check_gradient(func, x, eps=1e-6, rtol=1e-5, atol=1e-8):
        """
        Check gradients using finite differences.
        
        Args:
            func: Function that takes x as input and returns a scalar
            x: Input tensor
            eps: Epsilon for finite differences
            rtol: Relative tolerance
            atol: Absolute tolerance
        
        Returns:
            bool: True if gradients match numerical approximation
        """
        x.requires_grad_(True)
        y = func(x)
        y.backward()
        
        analytic_grad = x.grad.clone()
        x.grad.zero_()
        
        it = np.nditer(x.numpy(), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            
            # Compute numerical gradient
            old_value = x.numpy()[idx]
            
            x.numpy()[idx] = old_value + eps
            pos = func(x).item()
            
            x.numpy()[idx] = old_value - eps
            neg = func(x).item()
            
            x.numpy()[idx] = old_value
            
            numeric_grad = (pos - neg) / (2 * eps)
            
            # Compare gradients
            if not np.allclose(
                numeric_grad,
                analytic_grad.numpy()[idx],
                rtol=rtol,
                atol=atol
            ):
                return False
            
            it.iternext()
        
        return True
    
    return _check_gradient

@pytest.fixture(scope='session', autouse=True)
def initialize_distributed():
    """Initialize distributed environment if running distributed tests."""
    if is_distributed_test():
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=int(os.environ['WORLD_SIZE']),
            rank=int(os.environ['RANK'])
        )
        yield
        dist.destroy_process_group()
    else:
        yield

@pytest.fixture
def world_size():
    """Get the world size for distributed tests."""
    if is_distributed_test():
        return int(os.environ['WORLD_SIZE'])
    return 1

@pytest.fixture
def rank():
    """Get the rank for distributed tests."""
    if is_distributed_test():
        return int(os.environ['RANK'])
    return 0

@pytest.fixture
def distributed_tensor():
    """Create a tensor that's different on each process."""
    def _distributed_tensor(shape, requires_grad=False):
        rank = int(os.environ.get('RANK', 0))
        base = np.random.randn(*shape).astype(np.float32)
        return tensor(
            base + rank,
            requires_grad=requires_grad,
            device='cuda'
        )
    return _distributed_tensor

@pytest.fixture
def sync_tensor():
    """Create a tensor that's the same on all processes."""
    def _sync_tensor(shape, requires_grad=False):
        # Use same seed on all processes
        np.random.seed(42)
        t = tensor(
            np.random.randn(*shape).astype(np.float32),
            requires_grad=requires_grad,
            device='cuda'
        )
        np.random.seed()  # Reset seed
        return t
    return _sync_tensor 