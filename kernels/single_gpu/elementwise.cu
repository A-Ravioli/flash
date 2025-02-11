#include "custom_kernels/elementwise.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace flash {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

// Element-wise operation kernels
__global__ void add_forward_kernel(const float* a, const float* b, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void add_backward_kernel(const float* grad_output, float* grad_a, float* grad_b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (grad_a) grad_a[idx] = grad_output[idx];
        if (grad_b) grad_b[idx] = grad_output[idx];
    }
}

__global__ void sub_forward_kernel(const float* a, const float* b, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_backward_kernel(const float* grad_output, float* grad_a, float* grad_b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (grad_a) grad_a[idx] = grad_output[idx];
        if (grad_b) grad_b[idx] = -grad_output[idx];  // Note the negative sign
    }
}

__global__ void mul_forward_kernel(const float* a, const float* b, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_backward_kernel(const float* grad_output, const float* a, const float* b,
                                  float* grad_a, float* grad_b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (grad_a) grad_a[idx] = grad_output[idx] * b[idx];
        if (grad_b) grad_b[idx] = grad_output[idx] * a[idx];
    }
}

__global__ void div_forward_kernel(const float* a, const float* b, float* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void div_backward_kernel(const float* grad_output, const float* a, const float* b,
                                  float* grad_a, float* grad_b, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float b_val = b[idx];
        float b_squared = b_val * b_val;
        if (grad_a) grad_a[idx] = grad_output[idx] / b_val;
        if (grad_b) grad_b[idx] = -grad_output[idx] * a[idx] / b_squared;
    }
}

// ReLU kernels
__global__ void relu_forward_kernel(const float* input, float* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

__global__ void relu_backward_kernel(const float* grad_output, const float* input,
                                   float* grad_input, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

// Utility kernels
__global__ void fill_kernel(float* data, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void scale_kernel(float* data, float scale, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Launch functions
void add_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void add_backward(const float* grad_output, float* grad_a, float* grad_b, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_output, grad_a, grad_b, n);
}

void sub_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sub_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void sub_backward(const float* grad_output, float* grad_a, float* grad_b, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sub_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_output, grad_a, grad_b, n);
}

void mul_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void mul_backward(const float* grad_output, const float* a, const float* b,
                 float* grad_a, float* grad_b, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_output, a, b, grad_a, grad_b, n);
}

void div_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    div_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

void div_backward(const float* grad_output, const float* a, const float* b,
                 float* grad_a, float* grad_b, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    div_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_output, a, b, grad_a, grad_b, n);
}

void relu_forward(const float* input, float* output, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

void relu_backward(const float* grad_output, const float* input, float* grad_input,
                  size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(grad_output, input, grad_input, n);
}

void fill(float* data, float value, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, value, n);
}

void scale(float* data, float scale, size_t n, cudaStream_t stream) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, scale, n);
}

} // namespace cuda
} // namespace flash 