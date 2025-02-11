# Flash Neural Network Framework

Flash is a lightweight deep learning framework built for educational purposes and easy experimentation. It provides a PyTorch-like API with clean implementations of common neural network components.

## Features

- **Core Neural Network Components**
  - Linear layers
  - Convolutional layers (Conv2d)
  - Batch normalization
  - Various activation functions (ReLU, Sigmoid, Tanh, LeakyReLU)
  - Dropout

- **Advanced Architectures**
  - ResNet implementation
  - Transformer implementation
  - Support for custom architectures

- **Optimizers**
  - SGD with momentum and Nesterov momentum
  - Adam optimizer
  - AdamW with decoupled weight decay

## Installation

```bash
git clone https://github.com/yourusername/flash.git
cd flash
pip install -e .
```

## Quick Start

Here's a simple example of creating and training a convolutional neural network:

```python
from flash import nn, optim
from flash import tensor

# Define a simple CNN
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x[:, :, ::2, ::2]  # Max pool
        
        x = self.conv2(x)
        x = self.relu(x)
        x = x[:, :, ::2, ::2]  # Max pool
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model and optimizer
model = ConvNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        output = model(batch.data)
        loss = nn.functional.cross_entropy(output, batch.target)
        loss.backward()
        optimizer.step()
```

## Examples

The framework includes several example implementations:

1. **MNIST Training** (`examples/mnist.py`)
   - Basic CNN architecture
   - Complete training pipeline
   - Data loading and preprocessing

2. **ResNet** (`examples/resnet.py`)
   - ResNet-18 architecture
   - Residual blocks
   - Training on CIFAR-10

3. **Transformer** (`examples/transformer.py`)
   - Multi-head attention
   - Positional encoding
   - Text classification example

## Testing

The framework includes a comprehensive test suite:

```bash
python -m pytest tests/
```

Tests cover:
- Core neural network components
- Optimizer implementations
- Example architectures (ResNet, Transformer)
- Gradient checking
- Shape consistency

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 