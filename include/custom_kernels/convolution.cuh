#pragma once

#include <cuda_runtime.h>

namespace flash {
namespace cuda {

// Forward convolution operation
void conv2d_forward(const float* input, const float* weight, float* output,
                   int batch_size, int in_channels, int out_channels,
                   int in_height, int in_width,
                   int kernel_height, int kernel_width,
                   int stride_height, int stride_width,
                   int padding_height, int padding_width,
                   cudaStream_t stream);

// Backward convolution operations
void conv2d_backward_input(const float* grad_output, const float* weight, float* grad_input,
                          int batch_size, int in_channels, int out_channels,
                          int in_height, int in_width,
                          int kernel_height, int kernel_width,
                          int stride_height, int stride_width,
                          int padding_height, int padding_width,
                          cudaStream_t stream);

void conv2d_backward_weight(const float* grad_output, const float* input, float* grad_weight,
                           int batch_size, int in_channels, int out_channels,
                           int in_height, int in_width,
                           int kernel_height, int kernel_width,
                           int stride_height, int stride_width,
                           int padding_height, int padding_width,
                           cudaStream_t stream);

// Helper functions
void im2col(const float* input, float* col,
            int batch_size, int channels,
            int height, int width,
            int kernel_height, int kernel_width,
            int stride_height, int stride_width,
            int padding_height, int padding_width,
            cudaStream_t stream);

void col2im(const float* col, float* input,
            int batch_size, int channels,
            int height, int width,
            int kernel_height, int kernel_width,
            int stride_height, int stride_width,
            int padding_height, int padding_width,
            cudaStream_t stream);

} // namespace cuda
} // namespace flash 