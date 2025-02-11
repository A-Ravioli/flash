#pragma once

#include <cuda_runtime.h>

namespace flash {
namespace cuda {

// Matrix multiplication
void matmul_forward(const float* A, const float* B,
                   float* C,
                   int M, int N, int K,
                   cudaStream_t stream);

void matmul_backward(const float* grad_output,
                    const float* A, const float* B,
                    float* grad_A, float* grad_B,
                    int M, int N, int K,
                    cudaStream_t stream);

// Matrix transpose
void transpose_2d(const float* input, float* output,
                 int rows, int cols,
                 cudaStream_t stream);

} // namespace cuda
} // namespace flash 