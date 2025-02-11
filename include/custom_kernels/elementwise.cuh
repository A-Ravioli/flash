#pragma once

#include <cuda_runtime.h>

namespace flash {
namespace cuda {

// Element-wise operations
void add_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);
void add_backward(const float* grad_output, float* grad_a, float* grad_b, size_t n, cudaStream_t stream);

void sub_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);
void sub_backward(const float* grad_output, float* grad_a, float* grad_b, size_t n, cudaStream_t stream);

void mul_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);
void mul_backward(const float* grad_output, const float* a, const float* b,
                 float* grad_a, float* grad_b, size_t n, cudaStream_t stream);

void div_forward(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);
void div_backward(const float* grad_output, const float* a, const float* b,
                 float* grad_a, float* grad_b, size_t n, cudaStream_t stream);

// Activation functions
void relu_forward(const float* input, float* output, size_t n, cudaStream_t stream);
void relu_backward(const float* grad_output, const float* input, float* grad_input,
                  size_t n, cudaStream_t stream);

void sigmoid_forward(const float* input, float* output, size_t n, cudaStream_t stream);
void sigmoid_backward(const float* grad_output, const float* output, float* grad_input,
                     size_t n, cudaStream_t stream);

void tanh_forward(const float* input, float* output, size_t n, cudaStream_t stream);
void tanh_backward(const float* grad_output, const float* output, float* grad_input,
                  size_t n, cudaStream_t stream);

void leaky_relu_forward(const float* input, float* output, float negative_slope,
                       size_t n, cudaStream_t stream);
void leaky_relu_backward(const float* grad_output, const float* input,
                        float* grad_input, float negative_slope,
                        size_t n, cudaStream_t stream);

// Reduction operations
void sum_forward(const float* input, float* output, size_t n, cudaStream_t stream);
void mean_forward(const float* input, float* output, size_t n, cudaStream_t stream);

// Utility functions
void fill(float* data, float value, size_t n, cudaStream_t stream);
void scale(float* data, float scale, size_t n, cudaStream_t stream);

} // namespace cuda
} // namespace flash 