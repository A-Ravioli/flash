"""
Transformer implementation example using flash.
"""

import numpy as np
from flash import nn, optim
from flash import tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = nn.functional.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = attention @ V
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        out = self.W_o(out)
        
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        self.encoder_layers = nn.Sequential(*[
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def _create_positional_encoding(self, max_seq_len, d_model):
        pos_enc = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        
        return tensor(pos_enc.astype(np.float32))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.shape[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len]
        
        # Transformer encoder layers
        x = self.encoder_layers(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = tensor(data, device='cuda')
        target = tensor(target, device='cuda')
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/50000] '
                  f'Loss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data = tensor(data, device='cuda')
        target = tensor(target, device='cuda')
        
        output = model(data)
        test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

def main():
    # Model parameters
    vocab_size = 10000
    d_model = 256
    num_heads = 8
    num_layers = 6
    d_ff = 1024
    num_classes = 2
    max_seq_len = 512
    batch_size = 32
    
    # Create dummy text classification data
    X_train = np.random.randint(0, vocab_size, size=(50000, max_seq_len))
    y_train = np.random.randint(0, num_classes, size=(50000,))
    X_test = np.random.randint(0, vocab_size, size=(10000, max_seq_len))
    y_test = np.random.randint(0, num_classes, size=(10000,))
    
    # Create data loaders
    train_loader = [(X_train[i:i+batch_size], y_train[i:i+batch_size])
                    for i in range(0, len(X_train), batch_size)]
    test_loader = [(X_test[i:i+batch_size], y_test[i:i+batch_size])
                   for i in range(0, len(X_test), batch_size)]
    
    # Create model and optimizer
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=num_classes,
        max_seq_len=max_seq_len
    ).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98))
    
    # Training loop
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

if __name__ == '__main__':
    main() 