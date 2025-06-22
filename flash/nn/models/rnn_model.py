"""
Recurrent Neural Network model implementations.
"""

from ..layers import RNN, LSTM, GRU, Linear
from ..activations import ReLU

class RNNModel:
    """
    Recurrent Neural Network model.
    
    Architecture:
    - RNN layers
    - Optional fully connected layers on top
    - Output layer
    
    Attributes:
        rnn: RNN layer
        fc_layers: List of fully connected layers
        activations: List of activation functions
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 hidden_sizes=None, bidirectional=False):
        """
        Initialize an RNN model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state in RNN
            num_layers: Number of recurrent layers
            output_size: Size of the output
            hidden_sizes: List of hidden layer sizes for FC layers (optional)
            bidirectional: Whether to use bidirectional RNN
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # TODO: Build RNN layer(s)
        # TODO: Build fully connected layers if hidden_sizes is provided
        
    def __call__(self, x, h_0=None):
        """
        Forward pass of the RNN model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass through RNN
        # TODO: Take the last output or all outputs as needed
        # TODO: Pass through fully connected layers if present
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers


class LSTMModel:
    """
    Long Short-Term Memory network model.
    
    Architecture:
    - LSTM layers
    - Optional fully connected layers on top
    - Output layer
    
    Attributes:
        lstm: LSTM layer
        fc_layers: List of fully connected layers
        activations: List of activation functions
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 hidden_sizes=None, bidirectional=False):
        """
        Initialize an LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state in LSTM
            num_layers: Number of recurrent layers
            output_size: Size of the output
            hidden_sizes: List of hidden layer sizes for FC layers (optional)
            bidirectional: Whether to use bidirectional LSTM
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # TODO: Build LSTM layer(s)
        # TODO: Build fully connected layers if hidden_sizes is provided
        
    def __call__(self, x, h_0=None, c_0=None):
        """
        Forward pass of the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state
            c_0: Initial cell state
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass through LSTM
        # TODO: Take the last output or all outputs as needed
        # TODO: Pass through fully connected layers if present
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers


class GRUModel:
    """
    Gated Recurrent Unit network model.
    
    Architecture:
    - GRU layers
    - Optional fully connected layers on top
    - Output layer
    
    Attributes:
        gru: GRU layer
        fc_layers: List of fully connected layers
        activations: List of activation functions
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 hidden_sizes=None, bidirectional=False):
        """
        Initialize a GRU model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state in GRU
            num_layers: Number of recurrent layers
            output_size: Size of the output
            hidden_sizes: List of hidden layer sizes for FC layers (optional)
            bidirectional: Whether to use bidirectional GRU
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # TODO: Build GRU layer(s)
        # TODO: Build fully connected layers if hidden_sizes is provided
        
    def __call__(self, x, h_0=None):
        """
        Forward pass of the GRU model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # TODO: Implement forward pass through GRU
        # TODO: Take the last output or all outputs as needed
        # TODO: Pass through fully connected layers if present
        
    def parameters(self):
        """
        Get all learnable parameters of the model.
        
        Returns:
            List of all parameters
        """
        # TODO: Collect and return parameters from all layers 