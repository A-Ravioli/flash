#include "custom_kernels/matrix.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace flash {
namespace cuda {

constexpr int BLOCK_SIZE = 16;

// Matrix multiplication kernels
__global__ void matmul_forward_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        if (row < M && i * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + i * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (i * BLOCK_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmul_backward_A_kernel(const float* grad_output, const float* B,
                                       float* grad_A, int M, int N, int K) {
    __shared__ float grad_outputs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        if (row < M && i * BLOCK_SIZE + tx < N) {
            grad_outputs[ty][tx] = grad_output[row * N + i * BLOCK_SIZE + tx];
        } else {
            grad_outputs[ty][tx] = 0.0f;
        }
        
        if (col < K && i * BLOCK_SIZE + ty < N) {
            Bs[ty][tx] = B[col * N + i * BLOCK_SIZE + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += grad_outputs[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < K) {
        grad_A[row * K + col] = sum;
    }
}

__global__ void matmul_backward_B_kernel(const float* grad_output, const float* A,
                                       float* grad_B, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float grad_outputs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < (M + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        if (i * BLOCK_SIZE + ty < M && row < K) {
            As[ty][tx] = A[(i * BLOCK_SIZE + ty) * K + row];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (i * BLOCK_SIZE + ty < M && col < N) {
            grad_outputs[ty][tx] = grad_output[(i * BLOCK_SIZE + ty) * N + col];
        } else {
            grad_outputs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[k][tx] * grad_outputs[k][ty];
        }
        
        __syncthreads();
    }
    
    if (row < K && col < N) {
        grad_B[row * N + col] = sum;
    }
}

__global__ void transpose_2d_kernel(const float* input, float* output,
                                  int rows, int cols) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Launch functions
void matmul_forward(const float* A, const float* B, float* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_forward_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void matmul_backward(const float* grad_output, const float* A, const float* B,
                    float* grad_A, float* grad_B,
                    int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    if (grad_A) {
        dim3 grid_A((K + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_backward_A_kernel<<<grid_A, block, 0, stream>>>(
            grad_output, B, grad_A, M, N, K);
    }
    
    if (grad_B) {
        dim3 grid_B((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_backward_B_kernel<<<grid_B, block, 0, stream>>>(
            grad_output, A, grad_B, M, N, K);
    }
}

void transpose_2d(const float* input, float* output,
                 int rows, int cols, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    transpose_2d_kernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
}

} // namespace cuda
} // namespace flash 