#pragma once

#include <cuda_runtime.h>

namespace flash {
namespace cuda {

// Reduction operations along a dimension
void sum_forward(const float* input, float* output,
                const int* input_shape, const int* output_shape,
                int num_dims, int reduce_dim,
                bool keepdim,
                cudaStream_t stream);

void sum_backward(const float* grad_output, float* grad_input,
                 const int* input_shape, const int* output_shape,
                 int num_dims, int reduce_dim,
                 bool keepdim,
                 cudaStream_t stream);

void mean_forward(const float* input, float* output,
                 const int* input_shape, const int* output_shape,
                 int num_dims, int reduce_dim,
                 bool keepdim,
                 cudaStream_t stream);

void mean_backward(const float* grad_output, float* grad_input,
                  const int* input_shape, const int* output_shape,
                  int num_dims, int reduce_dim,
                  bool keepdim,
                  cudaStream_t stream);

// Helper functions
void compute_strides(const int* shape, int* strides, int num_dims);
int compute_size(const int* shape, int num_dims);

} // namespace cuda
} // namespace flash 