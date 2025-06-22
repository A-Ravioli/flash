"""
Tests for the Tensor class.
"""

import unittest
from flash.core import Tensor

class TestTensor(unittest.TestCase):
    """
    Test cases for the Tensor class.
    """
    
    def test_creation(self):
        """
        Test tensor creation.
        """
        # Test tensor creation with different data types
        self.assertEqual(Tensor([1, 2, 3]).data, [1, 2, 3])
        self.assertEqual(Tensor((1, 2, 3)).data, (1, 2, 3))
        self.assertEqual(Tensor(1).data, 1)
        self.assertEqual(Tensor(1.0).data, 1.0)
        self.assertEqual(Tensor(True).data, True)
        # Test tensor shape computation
        self.assertEqual(Tensor([1, 2, 3]).shape, (3,))
        self.assertEqual(Tensor((1, 2, 3)).shape, (3,))
        self.assertEqual(Tensor(1).shape, ())
        self.assertEqual(Tensor(1.0).shape, ())
        self.assertEqual(Tensor(True).shape, ())
        
    def test_operations(self):
        """
        Test basic tensor operations.
        """
        # Test tensor addition
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        self.assertEqual((a + b).data, [5, 7, 9])
        
        # Test tensor multiplication
        self.assertEqual((a * b).data, [4, 10, 18])
        
        # Test matrix multiplication
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual((a @ b).data, [[19, 20], [43, 44]])
        
        # Test sum operation
        self.assertEqual(Tensor([1, 2, 3]).sum(), 6)
        
        # Test reshape
        self.assertEqual(Tensor([1, 2, 3]).reshape((3, 1)).data, [[1], [2], [3]])
        
        # Test transpose
        self.assertEqual(Tensor([[1, 2], [3, 4]]).transpose().data, [[1, 3], [2, 4]])
        
    def test_backward(self):
        """
        Test automatic differentiation with backward.
        """
        # Test gradient computation for simple operations
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        c.backward()
        self.assertEqual(a.grad, [1, 1, 1])
        self.assertEqual(b.grad, [1, 1, 1])
        
        # Test gradient accumulation
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        d = a * b
        e = c + d
        e.backward()
        
    def test_complex_backward(self):
        """
        Test backward on a more complex computation graph.
        """
        # Test a small neural network-like computation
        # Test gradient accumulation
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        d = a * b
        e = c + d
        e.backward()


if __name__ == "__main__":
    unittest.main() 