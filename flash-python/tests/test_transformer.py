"""
Unit tests for Transformer implementation.
"""

import unittest
import numpy as np

from flash import tensor
from flash.examples.transformer import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    TransformerEncoderLayer,
    TransformerClassifier
)

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 64
        self.num_heads = 8
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
    
    def test_attention_shape(self):
        x = tensor(np.random.randn(self.batch_size, self.seq_len, self.d_model).astype(np.float32))
        out = self.mha(x, x, x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_attention_mask(self):
        x = tensor(np.random.randn(self.batch_size, self.seq_len, self.d_model).astype(np.float32))
        mask = tensor(np.ones((self.batch_size, self.num_heads, self.seq_len, self.seq_len)))
        out = self.mha(x, x, x, mask)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_attention_backward(self):
        x = tensor(np.random.randn(self.batch_size, self.seq_len, self.d_model).astype(np.float32),
                  requires_grad=True)
        out = self.mha(x, x, x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

class TestPositionwiseFeedForward(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.d_ff = 256
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff)
    
    def test_feedforward_shape(self):
        x = tensor(np.random.randn(2, 10, self.d_model).astype(np.float32))
        out = self.ff(x)
        self.assertEqual(out.shape, (2, 10, self.d_model))
    
    def test_feedforward_backward(self):
        x = tensor(np.random.randn(2, 10, self.d_model).astype(np.float32),
                  requires_grad=True)
        out = self.ff(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.num_heads = 8
        self.d_ff = 256
        self.encoder = TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff)
    
    def test_encoder_shape(self):
        x = tensor(np.random.randn(2, 10, self.d_model).astype(np.float32))
        out = self.encoder(x)
        self.assertEqual(out.shape, (2, 10, self.d_model))
    
    def test_encoder_backward(self):
        x = tensor(np.random.randn(2, 10, self.d_model).astype(np.float32),
                  requires_grad=True)
        out = self.encoder(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

class TestTransformerClassifier(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 64
        self.num_heads = 8
        self.num_layers = 2
        self.d_ff = 256
        self.num_classes = 2
        self.max_seq_len = 20
        
        self.model = TransformerClassifier(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            num_classes=self.num_classes,
            max_seq_len=self.max_seq_len
        )
    
    def test_classifier_shape(self):
        x = tensor(np.random.randint(0, self.vocab_size, size=(2, 15)))
        out = self.model(x)
        self.assertEqual(out.shape, (2, self.num_classes))
    
    def test_classifier_backward(self):
        x = tensor(np.random.randint(0, self.vocab_size, size=(2, 15)))
        out = self.model(x)
        out.sum().backward()
        
        # Check if gradients are computed
        self.assertIsNotNone(self.model.embedding.weight.grad)
        self.assertIsNotNone(self.model.classifier.weight.grad)
    
    def test_positional_encoding(self):
        pos_enc = self.model.pos_encoding
        self.assertEqual(pos_enc.shape, (self.max_seq_len, self.d_model))
        
        # Test if positional encodings are different for different positions
        self.assertFalse(np.allclose(
            pos_enc[0].numpy(),
            pos_enc[1].numpy()
        ))

if __name__ == '__main__':
    unittest.main() 