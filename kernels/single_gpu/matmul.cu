#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace flash {
namespace cuda {

constexpr int BLOCK_SIZE = 16;

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
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

void launch_matmul(const float* A, const float* B, float* C,
                  int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if (M <= 512 && N <= 512 && K <= 512) {
        matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    } else {
        matmul_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
}

} // namespace cuda
} // namespace flash 