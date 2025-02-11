"""
Unit tests for distributed training functionality.
"""

import unittest
import numpy as np
import torch.distributed as dist

from flash import nn, optim, tensor
from flash.distributed import (
    DistributedDataParallel,
    all_reduce,
    broadcast,
    reduce,
    gather,
    scatter
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

class TestDistributedOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:23456',
            world_size=1,
            rank=0
        )
    
    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
    
    def test_all_reduce_sum(self):
        # Create tensor on each GPU
        x = tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device='cuda')
        expected = x * dist.get_world_size()
        
        # Perform all-reduce
        result = all_reduce(x, op='sum')
        
        # Check result
        np.testing.assert_array_almost_equal(
            result.cpu().numpy(),
            expected.cpu().numpy()
        )
    
    def test_broadcast(self):
        if dist.get_rank() == 0:
            x = tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device='cuda')
        else:
            x = tensor(np.zeros(3, dtype=np.float32), device='cuda')
        
        # Broadcast from rank 0
        result = broadcast(x, src=0)
        
        # All ranks should have the same values
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result.cpu().numpy(),
            expected
        )
    
    def test_reduce_sum(self):
        x = tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device='cuda')
        expected = x * dist.get_world_size()
        
        # Reduce to rank 0
        if dist.get_rank() == 0:
            result = reduce(x, dst=0, op='sum')
            np.testing.assert_array_almost_equal(
                result.cpu().numpy(),
                expected.cpu().numpy()
            )
    
    def test_gather(self):
        # Each rank has different data
        rank = dist.get_rank()
        x = tensor(np.array([float(rank)], dtype=np.float32), device='cuda')
        
        # Gather on rank 0
        if dist.get_rank() == 0:
            result = gather(x, dst=0)
            expected = np.arange(dist.get_world_size(), dtype=np.float32)
            np.testing.assert_array_almost_equal(
                result.cpu().numpy(),
                expected
            )
    
    def test_scatter(self):
        if dist.get_rank() == 0:
            x = tensor(np.arange(dist.get_world_size(), dtype=np.float32), device='cuda')
        else:
            x = None
        
        # Scatter from rank 0
        result = scatter(x, src=0)
        
        # Each rank should get its corresponding value
        expected = np.array([float(dist.get_rank())], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            result.cpu().numpy(),
            expected
        )

class TestDistributedDataParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:23456',
            world_size=1,
            rank=0
        )
    
    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
    
    def setUp(self):
        self.model = SimpleModel().cuda()
        self.ddp_model = DistributedDataParallel(self.model)
        self.optimizer = optim.Adam(self.ddp_model.parameters())
    
    def test_forward(self):
        x = tensor(np.random.randn(4, 10).astype(np.float32), device='cuda')
        out = self.ddp_model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_backward(self):
        x = tensor(np.random.randn(4, 10).astype(np.float32), device='cuda')
        y = tensor(np.random.randn(4, 1).astype(np.float32), device='cuda')
        
        # Forward pass
        out = self.ddp_model(x)
        loss = ((out - y) ** 2).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check if gradients are synchronized
        for param in self.ddp_model.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_gradient_synchronization(self):
        # Create same input on all processes
        x = tensor(np.ones((4, 10), dtype=np.float32), device='cuda')
        y = tensor(np.ones((4, 1), dtype=np.float32), device='cuda')
        
        # Forward and backward pass
        out = self.ddp_model(x)
        loss = ((out - y) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check if gradients are same across processes
        for param in self.ddp_model.parameters():
            grad = param.grad.clone()
            grad_list = [tensor(np.zeros_like(grad.cpu().numpy()), device='cuda')
                        for _ in range(dist.get_world_size())]
            dist.all_gather(grad_list, grad)
            
            for other_grad in grad_list[1:]:
                np.testing.assert_array_almost_equal(
                    grad.cpu().numpy(),
                    other_grad.cpu().numpy()
                )
    
    def test_batch_splitting(self):
        # Create large batch
        batch_size = 16
        world_size = dist.get_world_size()
        x = tensor(np.random.randn(batch_size, 10).astype(np.float32), device='cuda')
        y = tensor(np.random.randn(batch_size, 1).astype(np.float32), device='cuda')
        
        # Split batch across processes
        rank = dist.get_rank()
        local_batch_size = batch_size // world_size
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size
        
        local_x = x[start_idx:end_idx]
        local_y = y[start_idx:end_idx]
        
        # Forward and backward pass
        out = self.ddp_model(local_x)
        loss = ((out - local_y) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check output shape
        self.assertEqual(out.shape, (local_batch_size, 1))

if __name__ == '__main__':
    unittest.main() 