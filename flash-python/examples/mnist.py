"""
MNIST training example using flash.
"""

import numpy as np
from flash import nn, optim
from flash import tensor

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = x[:, :, ::2, ::2]  # Max pool 2x2
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = x[:, :, ::2, ::2]  # Max pool 2x2
        
        # Fully connected layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def load_mnist():
    """Load MNIST dataset (dummy implementation)."""
    # This is a placeholder. In practice, you would load real MNIST data
    X_train = np.random.randn(60000, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(60000,)).astype(np.int64)
    X_test = np.random.randn(10000, 1, 28, 28).astype(np.float32)
    y_test = np.random.randint(0, 10, size=(10000,)).astype(np.int64)
    return (X_train, y_train), (X_test, y_test)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert numpy arrays to flash tensors
        data = tensor(data, device='cuda')
        target = tensor(target, device='cuda')
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/60000] '
                  f'Loss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        # Convert numpy arrays to flash tensors
        data = tensor(data, device='cuda')
        target = tensor(target, device='cuda')
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
        
        # Get predictions
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

def main():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = load_mnist()
    
    # Create data loaders (simple batching for this example)
    batch_size = 64
    train_loader = [(X_train[i:i+batch_size], y_train[i:i+batch_size])
                    for i in range(0, len(X_train), batch_size)]
    test_loader = [(X_test[i:i+batch_size], y_test[i:i+batch_size])
                   for i in range(0, len(X_test), batch_size)]
    
    # Create model and move to GPU
    model = ConvNet()
    model.cuda()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

if __name__ == '__main__':
    main() 