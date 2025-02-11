"""
Unit tests for ResNet implementation.
"""

import unittest
import numpy as np

from flash import tensor
from flash.examples.resnet import BasicBlock, ResNet, ResNet18

class TestBasicBlock(unittest.TestCase):
    def test_basic_block_forward(self):
        block = BasicBlock(64, 64)
        x = tensor(np.random.randn(2, 64, 32, 32).astype(np.float32))
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
    
    def test_basic_block_stride(self):
        block = BasicBlock(64, 128, stride=2)
        x = tensor(np.random.randn(2, 64, 32, 32).astype(np.float32))
        out = block(x)
        self.assertEqual(out.shape, (2, 128, 16, 16))
    
    def test_basic_block_backward(self):
        block = BasicBlock(64, 64)
        x = tensor(np.random.randn(2, 64, 32, 32).astype(np.float32), requires_grad=True)
        out = block(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

class TestResNet(unittest.TestCase):
    def test_resnet_forward(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        x = tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
    
    def test_resnet_backward(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        x = tensor(np.random.randn(2, 3, 32, 32).astype(np.float32), requires_grad=True)
        out = model(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
    
    def test_resnet18(self):
        model = ResNet18()
        x = tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
        
        # Test number of layers
        num_layers = sum(1 for m in model.modules() if isinstance(m, BasicBlock))
        self.assertEqual(num_layers, 8)  # ResNet18 has 8 BasicBlocks

if __name__ == '__main__':
    unittest.main() 