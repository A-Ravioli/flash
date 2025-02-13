# flash: a hardware-optimized ML framework

## Table of Contents

1. [Introduction](#introduction)
2. [Goals and Scope](#goals-and-scope)
3. [Key Design Decisions](#key-design-decisions)
    - 3.1 [Ease of Use – Imperative, Mutable API](#ease-of-use)
    - 3.2 [High Performance – SOTA Kernels and Optimizations](#high-performance)
4. [High-Level Architecture Overview](#high-level-architecture-overview)
    - 4.1 [Core Components](#core-components)
    - 4.2 [Hardware Abstraction and Kernel Integration](#hardware-abstraction-and-kernel-integration)
    - 4.3 [Distributed Training & LLM-Specific Features](#distributed-training-llm-specific-features)
5. [Repository Structure Overview](#repository-structure-overview)
    - 5.1 [C++ Implementation (Primary)](#c++-implementation)
    - 5.2 [Rust Implementation](#rust-implementation)
    - 5.3 [Go Implementation](#go-implementation)
    - 5.4 [Zig Implementation](#zig-implementation)
    - 5.5 [C Implementation](#c-implementation)
    - 5.6 [Python Bindings Implementation](#python-bindings-implementation)
6. [Detailed File and Directory Descriptions](#detailed-file-and-directory-descriptions)
    - 6.1 [Build System Files](#build-system-files)
    - 6.2 [Core Library Files](#core-library-files)
    - 6.3 [Kernel Implementations](#kernel-implementations)
    - 6.4 [Distributed Training Support](#distributed-training-support)
    - 6.5 [Examples, Tests, and Documentation](#examples-tests-and-documentation)
7. [Build System, Dependencies, and Integration](#build-system-dependencies-and-integration)
8. [Testing, Benchmarking, and Documentation](#testing-benchmarking-and-documentation)
9. [Future Work and Extensions](#future-work-and-extensions)
10. [Appendices](#appendices)

---

## 1. Introduction

This document details the design for a new **hardware-optimized ML framework** written in C++. Inspired by PyTorch and JAX, this framework is engineered to be:

- **Easier to use than JAX:** Offering an imperative programming model with mutable arrays that avoids the cognitive overhead of functional programming and immutable data structures.
- **Faster than PyTorch:** Leveraging custom state-of-the-art (SOTA) kernels and optimizations for both single-GPU and multi-GPU environments.
- **Optimized for distributed training and LLM development:** Featuring specialized kernels, efficient communication primitives, and memory management strategies tailored for large language models.

---

## 2. Goals and Scope

- **Usability:**
  Provide a simple, intuitive, imperative API (with mutable arrays) that lowers the learning curve compared to JAX, while maintaining powerful features such as automatic differentiation and dynamic graph execution.
  
- **Performance:**  
  Achieve superior performance relative to PyTorch by integrating custom SOTA kernels for critical operations (e.g., matrix multiplications, convolutions, transformer-specific routines) and advanced optimizations for both single-GPU and multi-GPU configurations.
  
- **Distributed Training:**  
  Incorporate robust support for distributed training with optimized communication routines (such as all-reduce) to easily scale LLMs across multiple GPUs/nodes.
  
- **LLM Optimization:**  
  Embed specialized features for large language model development, including mixed-precision training, memory-efficient data structures, and transformer-specific optimizations.

- **Modular & Extensible:**  
  Maintain a clean, modular codebase that can be extended or integrated with third-party tools and libraries.

---

## 3. Key Design Decisions

### 3.1 Ease of Use
- **Imperative Programming Model:**  
  Unlike JAX's functional and immutable array paradigm, our framework uses a more intuitive, imperative style with mutable arrays. This design minimizes the cognitive overhead and debugging complexity, making it accessible to users who prefer traditional programming paradigms.
  
- **Clear, Minimal API Surface:**  
  Expose a user-friendly API that abstracts away the low-level details while still offering control over performance-critical aspects when needed.

- **Dynamic Graph Execution:**  
  Support dynamic computation graphs that allow for interactive model building and debugging similar to PyTorch, but with simpler memory semantics and less boilerplate code.

### 3.2 High Performance
- **Custom SOTA Kernels:**  
  Implement specialized kernels for common ML operations using CUDA and highly optimized C++ routines. These kernels are designed to outperform generic implementations in PyTorch.

- **Optimized Memory and Device Management:**  
  A dedicated device abstraction layer will handle efficient memory transfers and device-specific optimizations, ensuring minimal overhead and maximal performance across different hardware configurations.

- **Distributed and Multi-GPU Optimizations:**  
  Provide built-in primitives and kernels for distributed training that are tightly integrated into the framework's core, reducing synchronization overhead and ensuring near-linear scaling.

---

## 4. High-Level Architecture Overview

### 4.1 Core Components

- **Tensor Engine:**  
  - **Purpose:** Serve as the core data structure with support for multi-dimensional arrays, various data types, and device-aware memory management.  
  - **Key Feature:** An imperative API allowing in-place operations and mutable arrays, reducing the complexity seen in functional frameworks like JAX.

- **Autograd Engine:**  
  - **Purpose:** Automatically compute gradients through dynamic graph construction.  
  - **Key Feature:** A flexible system that simplifies model training while keeping the interface clean and intuitive.

- **Module/Layer Abstraction:**  
  - **Purpose:** Define a hierarchy of neural network modules with a standard interface (`forward()`, `backward()`, `parameters()`) for building and composing models.
  
- **Optimizers and Schedulers:**  
  - **Purpose:** Implement efficient optimization routines (SGD, Adam, etc.) that work seamlessly with the autograd engine and support additional features like gradient clipping and mixed-precision updates for LLMs.

### 4.2 Hardware Abstraction and Kernel Integration

- **Device Abstraction Layer:**  
  - **Purpose:** Provide a unified API for CPU, single GPU, and multi-GPU environments.
  - **Key Feature:** Transparent device management that automatically dispatches operations to the best available hardware, ensuring both ease of use and high performance.

- **Custom SOTA Kernels:**  
  - **Purpose:** Deliver high-performance implementations of essential ML operations.
  - **Key Kernels Include:**
    - Matrix multiplication
    - Convolution (e.g., conv2d)
    - Transformer-specific operations (e.g., scaled dot-product attention)
  - **Integration:** Kernels reside in dedicated directories and are plugged into the tensor and operator layers via clearly defined interfaces.

### 4.3 Distributed Training & LLM-Specific Features

- **Distributed Training Module:**  
  - **Purpose:** Provide robust support for multi-GPU and multi-node training with optimized communication routines (all-reduce, broadcast, etc.).
  
- **LLM-Specific Optimizations:**  
  - **Purpose:** Offer features such as mixed-precision training, memory-efficient large tensor operations, and specialized transformer kernels, making the development of large language models more straightforward and efficient.
  
- **User Experience:**  
  - **Goal:** Minimize configuration overhead while automatically optimizing distributed operations behind the scenes, making scaling almost transparent to the user.

---

## 5. Repository Structure Overview

### 5.1 C++ Implementation (Primary)
```
flash-cpp/
├── CMakeLists.txt                    # Build configuration
├── README.md                         # Project overview
├── docs/
│   └── design.md                     # Design documentation
├── include/
│   ├── framework/                    # Public API headers
│   │   ├── tensor.h                  # Tensor class
│   │   ├── autograd.h                # Autograd engine
│   │   ├── module.h                  # Neural network modules
│   │   ├── optimizer.h               # Optimizers
│   │   ├── device.h                  # Device abstraction
│   │   └── distributed.h             # Distributed training
│   ├── nn/                           # Neural network layers
│   │   ├── linear.h                  # Linear layer
│   │   ├── conv.h                    # Convolution layers
│   │   ├── activation.h              # Activation functions
│   │   └── batchnorm.h              # Batch normalization
│   └── custom_kernels/               # CUDA kernel interfaces
├── src/
│   ├── tensor.cpp                    # Tensor implementation
│   ├── autograd.cpp                  # Autograd implementation
│   ├── module.cpp                    # Module base class
│   ├── optimizer.cpp                 # Optimizers
│   └── distributed.cpp               # Distributed training
├── kernels/
│   ├── single_gpu/                   # Single GPU kernels
│   │   ├── elementwise.cu           # Element-wise operations
│   │   ├── matmul.cu               # Matrix multiplication
│   │   └── conv.cu                 # Convolution operations
│   └── multi_gpu/                    # Multi-GPU kernels
├── examples/
│   ├── mnist.cpp                     # MNIST example
│   └── distributed.cpp               # Distributed training
└── tests/                            # Unit tests
```

### 5.2 Rust Implementation
```
flash-rs/
├── Cargo.toml                        # Rust package manifest
├── README.md                         # Project overview
├── src/
│   ├── lib.rs                        # Library root
│   ├── tensor/
│   │   ├── mod.rs                    # Tensor module
│   │   └── ops.rs                    # Tensor operations
│   ├── autograd/
│   │   ├── mod.rs                    # Autograd module
│   │   └── engine.rs                 # Autograd engine
│   ├── nn/
│   │   ├── mod.rs                    # Neural network module
│   │   ├── linear.rs                 # Linear layer
│   │   ├── conv.rs                   # Convolution layers
│   │   └── activation.rs             # Activation functions
│   ├── cuda/
│   │   ├── mod.rs                    # CUDA module
│   │   ├── kernels/                  # CUDA kernels
│   │   └── device.rs                 # Device management
│   └── distributed/
│       ├── mod.rs                    # Distributed module
│       └── comm.rs                   # Communication primitives
├── examples/
│   ├── mnist.rs                      # MNIST example
│   └── distributed.rs                # Distributed example
└── tests/                            # Integration tests
```

### 5.3 Go Implementation
```
flash-go/
├── go.mod                            # Go module file
├── README.md                         # Project overview
├── pkg/
│   ├── tensor/
│   │   ├── tensor.go                 # Tensor implementation
│   │   └── ops.go                    # Tensor operations
│   ├── autograd/
│   │   ├── engine.go                 # Autograd engine
│   │   └── node.go                   # Computation nodes
│   ├── nn/
│   │   ├── module.go                 # Base module
│   │   ├── linear.go                 # Linear layer
│   │   └── conv.go                   # Convolution layer
│   ├── cuda/
│   │   ├── device.go                 # Device management
│   │   └── kernels/                  # CUDA kernels
│   └── distributed/
│       └── comm.go                   # Communication
├── cmd/
│   ├── mnist/                        # MNIST example
│   └── distributed/                  # Distributed example
├── internal/                         # Internal packages
└── test/                            # Tests
```

### 5.4 Zig Implementation
```
flash-zig/
├── build.zig                         # Build script
├── README.md                         # Project overview
├── src/
│   ├── main.zig                      # Library root
│   ├── tensor/
│   │   ├── tensor.zig                # Tensor implementation
│   │   └── ops.zig                   # Tensor operations
│   ├── autograd/
│   │   ├── engine.zig                # Autograd engine
│   │   └── node.zig                  # Computation nodes
│   ├── nn/
│   │   ├── module.zig                # Base module
│   │   ├── linear.zig                # Linear layer
│   │   └── conv.zig                  # Convolution layer
│   ├── cuda/
│   │   ├── device.zig                # Device management
│   │   └── kernels/                  # CUDA kernels
│   └── distributed/
│       └── comm.zig                  # Communication
├── examples/
│   ├── mnist.zig                     # MNIST example
│   └── distributed.zig               # Distributed example
└── test/                            # Tests
```

### 5.5 C Implementation
```
flash-c/
├── Makefile                          # Build configuration
├── README.md                         # Project overview
├── include/
│   ├── flash/
│   │   ├── tensor.h                  # Tensor interface
│   │   ├── autograd.h                # Autograd interface
│   │   ├── nn.h                      # Neural network interface
│   │   ├── cuda.h                    # CUDA interface
│   │   └── distributed.h             # Distributed interface
│   └── flash_internal.h              # Internal headers
├── src/
│   ├── tensor/
│   │   ├── tensor.c                  # Tensor implementation
│   │   └── ops.c                     # Tensor operations
│   ├── autograd/
│   │   ├── engine.c                  # Autograd engine
│   │   └── node.c                    # Computation nodes
│   ├── nn/
│   │   ├── linear.c                  # Linear layer
│   │   ├── conv.c                    # Convolution layer
│   │   └── activation.c              # Activation functions
│   ├── cuda/
│   │   ├── device.c                  # Device management
│   │   └── kernels/                  # CUDA kernels
│   └── distributed/
│       └── comm.c                    # Communication
├── examples/
│   ├── mnist.c                       # MNIST example
│   └── distributed.c                 # Distributed example
└── tests/                           # Unit tests
```

### 5.6 Python Bindings Implementation
```
flash-python/
├── setup.py                          # Python package setup
├── pyproject.toml                    # Build system requirements
├── CMakeLists.txt                    # CMake config for C++ bindings
├── README.md                         # Python package documentation
├── flash/
│   ├── __init__.py                  # Package initialization
│   ├── _C/                          # C++ extension module
│   │   ├── bindings/
│   │   │   ├── tensor.cpp           # Tensor bindings
│   │   │   ├── autograd.cpp         # Autograd bindings
│   │   │   ├── nn.cpp              # Neural network bindings
│   │   │   └── cuda.cpp            # CUDA operation bindings
│   │   └── module.cpp               # Main binding module
│   ├── tensor.py                    # Python Tensor wrapper
│   ├── autograd.py                  # Autograd interface
│   ├── nn/
│   │   ├── __init__.py             # Neural network package
│   │   ├── modules.py              # Base modules
│   │   ├── linear.py               # Linear layers
│   │   ├── conv.py                 # Convolution layers
│   │   └── functional.py           # Functional interface
│   ├── optim/
│   │   ├── __init__.py             # Optimizer package
│   │   ├── optimizer.py            # Base optimizer
│   │   └── adam.py                 # Adam optimizer
│   └── distributed/
│       ├── __init__.py             # Distributed package
│       └── parallel.py             # Parallel processing
├── examples/
│   ├── mnist.py                     # MNIST training example
│   └── distributed_training.py      # Distributed training
└── tests/
    ├── test_tensor.py               # Tensor tests
    ├── test_autograd.py             # Autograd tests
    └── test_nn.py                   # Neural network tests
```

The Python bindings follow these design principles:

1. **Seamless Integration**:
   - Native Python types convert automatically to flash types
   - NumPy-like interface for tensor operations
   - PyTorch-like API for neural networks

2. **Performance**:
   - Zero-copy tensor operations where possible
   - Direct access to C++ CUDA kernels
   - Efficient memory management

3. **Type Safety**:
   - Strong type checking via pybind11
   - Python type hints for better IDE support
   - Clear error messages

4. **Example Python Interface**:
```python
import flash
import flash.nn as nn
import flash.optim as optim

# Create a simple neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * 26 * 26)
        x = self.fc1(x)
        return x

# Create model, optimizer, and move to GPU
model = Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for data, target in dataloader:
        data = flash.tensor(data).cuda()
        target = flash.tensor(target).cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
```

5. **Key Components**:

   a. **Core Bindings** (`flash/_C/`):
      - Direct bindings to C++ classes and functions
      - Memory management and type conversion
      - CUDA operation exposure

   b. **Python Wrappers** (`flash/`):
      - High-level Python interfaces
      - NumPy/PyTorch-like API
      - Additional Python-specific features

   c. **Neural Network API** (`flash.nn`):
      - Module system matching PyTorch's design
      - Layer implementations
      - Functional interface

   d. **Optimizers** (`flash.optim`):
      - Standard optimization algorithms
      - Learning rate scheduling
      - Gradient clipping

   e. **Distributed Training** (`flash.distributed`):
      - Multi-GPU training support
      - Process groups
      - Collective operations

6. **Build System Integration**:
   ```python
   # setup.py
   from setuptools import setup, Extension
   from setuptools.command.build_ext import build_ext
   import pybind11

   ext_modules = [
       Extension(
           "flash._C",
           ["flash/_C/module.cpp"],
           include_dirs=[pybind11.get_include()],
           extra_compile_args=["-std=c++17"],
           language='c++'
       )
   ]

   setup(
       name="flash",
       version="0.1.0",
       ext_modules=ext_modules,
       cmdclass={"build_ext": build_ext},
       install_requires=[
           "numpy>=1.20.0",
           "pybind11>=2.6.0",
       ],
   )
   ```

7. **Testing and Documentation**:
   - Unit tests using pytest
   - Integration tests with real models
   - Comprehensive API documentation
   - Jupyter notebook tutorials

Each implementation follows the idioms and best practices of its respective language while maintaining similar functionality:
- Tensor operations
- Automatic differentiation
- Neural network layers
- CUDA acceleration
- Distributed training support

The core functionality remains consistent across implementations while leveraging each language's strengths:
- Tensor operations
- Automatic differentiation
- Neural network layers
- CUDA acceleration
- Distributed training support

---

## 6. Detailed File and Directory Descriptions

### 6.1 Build System Files

- **CMakeLists.txt**  
  - **Purpose:** Configure the build system, detect available hardware (e.g., CUDA devices), set compiler flags (optimizations, C++17/20 standard), and include external libraries (e.g., NCCL, CUDA, BLAS).
  - **Key Points:**  
    - Separate build targets for CPU-only, single-GPU, and multi-GPU builds.
    - Options for debugging and performance benchmarking.

- **README.md**  
  - **Purpose:** Provide a high-level overview, build instructions, usage examples, and a comparison with other frameworks (highlighting simplicity and speed).

- **docs/design.md**  
  - **Purpose:** Maintain this design document and update design decisions as the project evolves.

### 6.2 Core Library Files

- **include/framework/tensor.h**  
  - **Purpose:** Declaration of the `Tensor` class.
  - **Key Features:**  
    - Support for mutable arrays and in-place operations.
    - Overloaded operators for arithmetic, indexing, and device transfer (e.g., `toGPU()`, `toCPU()`).
    - A user-friendly API that hides low-level details.

- **src/tensor.cpp**  
  - **Purpose:** Implements tensor operations.
  - **Key Features:**  
    - Efficient memory allocation with custom allocators.
    - Integration with custom SOTA kernels for performance-critical paths.
    - In-place operations to allow a simpler imperative programming model.

- **include/framework/autograd.h** and **src/autograd.cpp**  
  - **Purpose:** Define and implement the automatic differentiation engine.
  - **Key Features:**  
    - Dynamic graph construction that supports imperative coding.
    - Simplified gradient propagation logic that abstracts complexity from the user.

- **include/framework/module.h** and **src/module.cpp**  
  - **Purpose:** Provide an abstract base class for neural network layers/modules.
  - **Key Features:**  
    - Clean interface (`forward()`, `backward()`, `parameters()`) that facilitates easy model building.
    - Support for both simple and composite layers, ensuring ease of use.

- **include/framework/optimizer.h** and **src/optimizer.cpp**  
  - **Purpose:** Define and implement optimizers.
  - **Key Features:**  
    - Standard algorithms (SGD, Adam, etc.) with LLM-specific extensions (e.g., gradient clipping, mixed-precision).
    - Seamless integration with the autograd engine to simplify training loops.

- **include/framework/device.h** and **src/device.cpp**  
  - **Purpose:** Provide a unified abstraction over hardware devices (CPU, GPU).
  - **Key Features:**  
    - Automatic detection and dispatch to available hardware.
    - Transparent memory management and device scheduling.

- **include/framework/distributed.h** and **src/distributed.cpp**  
  - **Purpose:** Define and implement APIs for distributed training.
  - **Key Features:**  
    - High-performance collective operations (all-reduce, broadcast) integrated with custom multi-GPU kernels.
    - Fault tolerance and synchronization primitives that reduce complexity for end users.

### 6.3 Kernel Implementations

- **include/custom_kernels/matmul_kernel.h**  
  - **Purpose:** Define the interface for matrix multiplication kernels.
  - **Key Features:**  
    - Abstracts the differences between single-GPU and multi-GPU implementations.
    - Clearly documented function signatures.

- **kernels/single_gpu/matmul.cu**  
  - **Purpose:** Implement a highly optimized matrix multiplication routine for single GPU.
  - **Key Features:**  
    - Leverages shared memory tiling and other CUDA best practices.
    - Provides a clear integration point with the tensor engine.

- **kernels/single_gpu/conv2d.cu**  
  - **Purpose:** Implement a fast convolution kernel.
  - **Key Features:**  
    - Optimized for various input sizes and batch dimensions.
    - Fallbacks to cuBLAS/cuDNN where beneficial, with custom routines for specialized cases.

- **kernels/single_gpu/transformer_attention.cu**  
  - **Purpose:** Provide a custom kernel optimized for transformer attention mechanisms.
  - **Key Features:**  
    - Tailored for LLM training scenarios, ensuring both speed and memory efficiency.

- **kernels/multi_gpu/all_reduce.cu**  
  - **Purpose:** Implement an all-reduce operation optimized for distributed training.
  - **Key Features:**  
    - Uses efficient communication patterns (e.g., ring-allreduce).
    - Integrates with network libraries (e.g., NCCL) for high performance.

- **kernels/multi_gpu/ring_reduce.cu & pipeline_parallel.cu**  
  - **Purpose:** Provide alternative methods for gradient reduction and support for pipeline parallelism in large-scale models.

### 6.4 Distributed Training Support

- **include/framework/distributed.h** and **src/distributed.cpp**  
  - **Purpose:** Provide APIs for initializing and managing multi-GPU/multi-node environments.
  - **Key Features:**  
    - Built-in fault tolerance and synchronization.
    - Transparent handling of both synchronous and asynchronous distributed updates.

### 6.5 Examples, Tests, and Documentation

- **examples/mnist.cpp**  
  - **Purpose:** Demonstrate building, training, and evaluating a simple neural network on MNIST with an imperative API.
  
- **examples/transformer_llm.cpp**  
  - **Purpose:** Showcase how to set up and train a transformer model for LLM development using specialized kernels and distributed routines.
  
- **examples/distributed_training.cpp**  
  - **Purpose:** Illustrate multi-GPU training with minimal user configuration, emphasizing ease of use.
  
- **tests/**  
  - **Purpose:** Contain unit and integration tests for every component.
  - **Files Include:**  
    - `test_tensor.cpp`: Validates tensor operations and memory management.
    - `test_autograd.cpp`: Verifies gradient computation and graph consistency.
    - `test_kernels.cu`: Checks both correctness and performance of custom kernels.
    - `test_distributed.cpp`: Tests distributed routines under simulated conditions.
  
- **third_party/**  
  - **Purpose:** Store or reference external libraries (e.g., NCCL, MPI) as needed.
  - **Documentation:** A README with setup instructions for third-party dependencies.

---

## 7. Build System, Dependencies, and Integration

- **Build System:**  
  - Use **CMake** for cross-platform support and integration of CUDA code.
  
- **Dependencies:**  
  - **CUDA Toolkit:** For compiling GPU kernels.
  - **NCCL/MPI:** For distributed training.
  - **BLAS/LAPACK:** For CPU-based linear algebra operations.
  - **Optional Tools:** Doxygen for documentation and Catch2/GoogleTest for testing.
  
- **Integration:**  
  - The CMake configuration automatically detects hardware, sets proper compiler flags, and creates separate build targets for CPU-only, single-GPU, and distributed builds.

---

## 8. Testing, Benchmarking, and Documentation

- **Unit and Integration Tests:**  
  - Use a testing framework (e.g., GoogleTest or Catch2) to ensure correctness of each module.
  - Dedicated tests for GPU kernels with both synthetic and real-world data.
  
- **Benchmarking:**  
  - Provide benchmarks comparing performance across single-GPU and multi-GPU setups.
  - Specific benchmarks for gradient computations, kernel execution times, and training throughput.
  
- **Documentation:**  
  - Generate API documentation using Doxygen.
  - Maintain the `docs/` directory with user guides, developer guides, and design notes.
  - Include tutorials demonstrating the imperative, easy-to-use API in contrast with JAX.

---

## 9. Future Work and Extensions

- **JIT Compilation and Graph Optimization:**  
  - Explore integrating a JIT compiler for further optimizations without sacrificing the simplicity of the imperative interface.
  
- **Extended Hardware Support:**  
  - Investigate support for emerging accelerators (e.g., TPUs, specialized AI chips) while preserving ease of use.
  
- **Advanced Distributed Strategies:**  
  - Implement dynamic load balancing and further reduce synchronization overhead.
  
- **Community Contributions:**  
  - Open the repository for community contributions with clear guidelines for adding new kernels and modules.

---

## 10. Appendices

### Appendix A: Coding Standards and Conventions

- **C++ Standard:**  
  Use C++17 (or C++20 where beneficial).
  
- **Style Guidelines:**  
  Follow modern C++ best practices (RAII, smart pointers, etc.) with thorough inline documentation, especially in performance-critical sections.
  
- **Documentation:**  
  Provide detailed comments and usage examples, ensuring that the imperative API is easy to understand and adopt.

### Appendix B: Performance Goals

- **Single-GPU:**  
  Target at least a 20–30% speed improvement over existing frameworks (e.g., PyTorch) on key operations.
  
- **Multi-GPU:**  
  Achieve near-linear scaling up to 8–16 GPUs for distributed training.
  
- **LLM Support:**  
  Efficiently train transformer models with low memory overhead and fast gradient propagation.
