"""
Unit tests for neural network modules.
"""

import unittest
import numpy as np

from flash import nn, tensor

class TestLinear(unittest.TestCase):
    def test_linear_forward(self):
        layer = nn.Linear(10, 5)
        x = tensor(np.random.randn(2, 10).astype(np.float32))
        out = layer(x)
        self.assertEqual(out.shape, (2, 5))
    
    def test_linear_backward(self):
        layer = nn.Linear(10, 5)
        x = tensor(np.random.randn(2, 10).astype(np.float32), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

class TestConv2d(unittest.TestCase):
    def test_conv2d_forward(self):
        layer = nn.Conv2d(3, 16, 3, padding=1)
        x = tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))
    
    def test_conv2d_backward(self):
        layer = nn.Conv2d(3, 16, 3, padding=1)
        x = tensor(np.random.randn(2, 3, 32, 32).astype(np.float32), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

class TestBatchNorm2d(unittest.TestCase):
    def test_batchnorm2d_forward(self):
        layer = nn.BatchNorm2d(16)
        x = tensor(np.random.randn(2, 16, 32, 32).astype(np.float32))
        out = layer(x)
        self.assertEqual(out.shape, (2, 16, 32, 32))
    
    def test_batchnorm2d_backward(self):
        layer = nn.BatchNorm2d(16)
        x = tensor(np.random.randn(2, 16, 32, 32).astype(np.float32), requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

class TestActivations(unittest.TestCase):
    def test_relu(self):
        layer = nn.ReLU()
        x = tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        out = layer(x)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
    
    def test_sigmoid(self):
        layer = nn.Sigmoid()
        x = tensor(np.array([0.0], dtype=np.float32))
        out = layer(x)
        self.assertAlmostEqual(out.item(), 0.5, places=6)
    
    def test_tanh(self):
        layer = nn.Tanh()
        x = tensor(np.array([0.0], dtype=np.float32))
        out = layer(x)
        self.assertAlmostEqual(out.item(), 0.0, places=6)
    
    def test_leaky_relu(self):
        layer = nn.LeakyReLU(0.1)
        x = tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        out = layer(x)
        np.testing.assert_array_almost_equal(
            out.numpy(),
            np.array([-0.1, 0.0, 1.0], dtype=np.float32)
        )

if __name__ == '__main__':
    unittest.main() 