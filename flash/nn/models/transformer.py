"""
Transformer model implementation.
"""

from ..layers import Linear, MultiHeadAttention, TransformerEncoderLayer
from ..activations import GELU

class PositionalEncoding:
    """
    Positional encoding for transformer input sequences.
    
    Adds positional information to the input embeddings.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        # TODO: Precompute positional encoding values
        # TODO: Use sin/cos functions as in the paper "Attention Is All You Need"
        
    def __call__(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        # TODO: Add precomputed positional encoding to input x
        # TODO: Slice according to sequence length


class Transformer:
    """
    Transformer model for sequence processing.
    
    A simplified implementation following "Attention Is All You Need".
    
    Attributes:
        embedding: Input embedding layer
        positional_encoding: Positional encoding module
        encoder_layers: List of transformer encoder layers
        fc_out: Output fully connected layer
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, output_size, dropout=0.1):
        """
        Initialize a Transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            output_size: Size of the output
            dropout: Dropout probability
        """
        self.d_model = d_model
        
        # TODO: Set up embedding layer
        # TODO: Set up positional encoding
        # TODO: Build encoder layers
        # TODO: Set up output layer
        
    def __call__(self, x, mask=None):
        """
        Forward pass of the Transformer.
        
        Args:
            x: Input tensor of token indices of shape (batch_size, seq_len)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Apply embedding to input
        # TODO: Apply positional encoding
        # TODO: Pass through encoder layers
        # TODO: Pool encoder output (e.g., use first token or average)
        # TODO: Pass through output layer
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers 