"""
Unit tests for optimizers.
"""

import unittest
import numpy as np

from flash import nn, optim, tensor

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.x = tensor(np.random.randn(5, 10).astype(np.float32))
        self.y = tensor(np.random.randn(5, 1).astype(np.float32))
    
    def _test_optimizer(self, optimizer_class, **kwargs):
        optimizer = optimizer_class(self.model.parameters(), **kwargs)
        
        # Initial loss
        output = self.model(self.x)
        loss = ((output - self.y) ** 2).mean()
        initial_loss = loss.item()
        
        # Training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if loss decreased
        output = self.model(self.x)
        loss = ((output - self.y) ** 2).mean()
        final_loss = loss.item()
        
        self.assertLess(final_loss, initial_loss)
    
    def test_sgd(self):
        self._test_optimizer(optim.SGD, lr=0.1)
    
    def test_sgd_momentum(self):
        self._test_optimizer(optim.SGD, lr=0.1, momentum=0.9)
    
    def test_sgd_nesterov(self):
        self._test_optimizer(optim.SGD, lr=0.1, momentum=0.9, nesterov=True)
    
    def test_adam(self):
        self._test_optimizer(optim.Adam, lr=0.1)
    
    def test_adam_amsgrad(self):
        self._test_optimizer(optim.Adam, lr=0.1, amsgrad=True)
    
    def test_adamw(self):
        self._test_optimizer(optim.AdamW, lr=0.1)

class TestOptimizerState(unittest.TestCase):
    def test_state_dict(self):
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        
        # Perform an update
        x = tensor(np.random.randn(5, 10).astype(np.float32))
        y = tensor(np.random.randn(5, 1).astype(np.float32))
        output = model(x)
        loss = ((output - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer
        new_optimizer = optim.Adam(model.parameters())
        new_optimizer.load_state_dict(state)
        
        # Check if states match
        for param in model.parameters():
            self.assertTrue(
                np.allclose(
                    optimizer.state[param]['exp_avg'].numpy(),
                    new_optimizer.state[param]['exp_avg'].numpy()
                )
            )
            self.assertTrue(
                np.allclose(
                    optimizer.state[param]['exp_avg_sq'].numpy(),
                    new_optimizer.state[param]['exp_avg_sq'].numpy()
                )
            )

if __name__ == '__main__':
    unittest.main() 