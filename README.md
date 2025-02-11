# flash: A Hardware-Optimized ML Framework

flash is a high-performance machine learning framework written in C++ with CUDA support. It provides an intuitive, imperative API similar to PyTorch while delivering superior performance through hardware-optimized kernels.

## Features

- **Intuitive API**: Imperative programming model with mutable arrays
- **High Performance**: Custom SOTA kernels for critical operations
- **Hardware Optimized**: Efficient CUDA implementations for GPU acceleration
- **Automatic Differentiation**: Dynamic computational graphs
- **Modern Neural Network Layers**: Conv2d, Linear, BatchNorm2d, etc.
- **Optimizers**: SGD, Adam, AdamW with various options
- **Distributed Training**: Support for multi-GPU and distributed setups (coming soon)

## Requirements

- C++17 compiler
- CUDA Toolkit 11.0 or later
- CMake 3.15 or later

## Building from Source

```bash
mkdir build
cd build
cmake ..
make -j
```

## Quick Start

Here's a simple example of creating and training a neural network:

```cpp
#include "framework/tensor.h"
#include "framework/module.h"
#include "framework/optimizer.h"

using namespace flash;

// Define a simple neural network
class Net : public Module {
public:
    Net() {
        fc1_ = make_module<Linear>(784, 128);
        fc2_ = make_module<Linear>(128, 10);
    }
    
    Variable forward(const Variable& x) override {
        auto h = fc1_->forward(x);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_, fc2_;
};

int main() {
    // Create model and optimizer
    auto model = std::make_shared<Net>();
    Adam optimizer(model->parameters(), 0.01);
    
    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        Variable output = model->forward(input);
        // Compute loss
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

## Directory Structure

```
flash/
├── include/              # Public headers
│   ├── framework/        # Core framework headers
│   └── custom_kernels/   # CUDA kernel interfaces
├── src/                  # Implementation files
├── kernels/              # CUDA kernel implementations
│   ├── single_gpu/       # Single GPU kernels
│   └── multi_gpu/        # Multi-GPU kernels
├── examples/             # Example programs
└── tests/                # Unit tests
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by PyTorch and JAX
- Built with modern C++ and CUDA best practices
- Optimized for both single-GPU and multi-GPU environments
