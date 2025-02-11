#include "custom_kernels/reduction.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace flash {
namespace cuda {

constexpr int BLOCK_SIZE = 256;

// Helper functions
__device__ __host__ void compute_strides(const int* shape, int* strides, int num_dims) {
    strides[num_dims - 1] = 1;
    for (int i = num_dims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

__device__ __host__ int compute_size(const int* shape, int num_dims) {
    int size = 1;
    for (int i = 0; i < num_dims; ++i) {
        size *= shape[i];
    }
    return size;
}

__device__ int compute_index(const int* strides, const int* indices, int num_dims) {
    int index = 0;
    for (int i = 0; i < num_dims; ++i) {
        index += indices[i] * strides[i];
    }
    return index;
}

// Reduction kernels
template<typename ReduceOp>
__global__ void reduce_forward_kernel(const float* input, float* output,
                                    const int* input_shape, const int* output_shape,
                                    const int* input_strides, const int* output_strides,
                                    int num_dims, int reduce_dim,
                                    int reduce_size, ReduceOp op) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Calculate output index
    int output_size = compute_size(output_shape, num_dims);
    int output_idx = bid;
    
    if (output_idx >= output_size) return;
    
    // Convert linear output index to multidimensional indices
    int output_indices[8];  // Assuming max 8 dimensions
    int temp_idx = output_idx;
    for (int i = 0; i < num_dims; ++i) {
        if (i != reduce_dim || output_shape[i] != 1) {
            output_indices[i] = temp_idx / output_strides[i];
            temp_idx %= output_strides[i];
        } else {
            output_indices[i] = 0;
        }
    }
    
    // Perform reduction
    float acc = op.init();
    for (int i = tid; i < reduce_size; i += blockDim.x) {
        output_indices[reduce_dim] = i;
        int input_idx = compute_index(input_strides, output_indices, num_dims);
        acc = op(acc, input[input_idx]);
    }
    
    // Store in shared memory
    shared_mem[tid] = acc;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = op(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[output_idx] = op.finalize(shared_mem[0], reduce_size);
    }
}

// Reduction operators
struct SumOp {
    __device__ float init() const { return 0.0f; }
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ float finalize(float val, int size) const { return val; }
};

struct MeanOp {
    __device__ float init() const { return 0.0f; }
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ float finalize(float val, int size) const { return val / size; }
};

// Backward kernels
template<typename BackwardOp>
__global__ void reduce_backward_kernel(const float* grad_output, float* grad_input,
                                     const int* input_shape, const int* output_shape,
                                     const int* input_strides, const int* output_strides,
                                     int num_dims, int reduce_dim,
                                     int reduce_size, BackwardOp op) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int input_size = compute_size(input_shape, num_dims);
    
    if (tid >= input_size) return;
    
    // Convert linear input index to multidimensional indices
    int indices[8];  // Assuming max 8 dimensions
    int temp_idx = tid;
    for (int i = 0; i < num_dims; ++i) {
        indices[i] = temp_idx / input_strides[i];
        temp_idx %= input_strides[i];
    }
    
    // Calculate output index
    indices[reduce_dim] = 0;  // Collapse reduced dimension
    int output_idx = compute_index(output_strides, indices, num_dims);
    
    // Compute gradient
    grad_input[tid] = op(grad_output[output_idx], reduce_size);
}

// Backward operators
struct SumBackwardOp {
    __device__ float operator()(float grad_output, int size) const {
        return grad_output;
    }
};

struct MeanBackwardOp {
    __device__ float operator()(float grad_output, int size) const {
        return grad_output / size;
    }
};

// Launch functions
void sum_forward(const float* input, float* output,
                const int* input_shape, const int* output_shape,
                int num_dims, int reduce_dim,
                bool keepdim,
                cudaStream_t stream) {
    int input_strides[8], output_strides[8];
    compute_strides(input_shape, input_strides, num_dims);
    compute_strides(output_shape, output_strides, num_dims);
    
    int reduce_size = input_shape[reduce_dim];
    int output_size = compute_size(output_shape, num_dims);
    
    int block_size = std::min(BLOCK_SIZE, reduce_size);
    int num_blocks = output_size;
    
    reduce_forward_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        input, output,
        input_shape, output_shape,
        input_strides, output_strides,
        num_dims, reduce_dim,
        reduce_size, SumOp()
    );
}

void sum_backward(const float* grad_output, float* grad_input,
                 const int* input_shape, const int* output_shape,
                 int num_dims, int reduce_dim,
                 bool keepdim,
                 cudaStream_t stream) {
    int input_strides[8], output_strides[8];
    compute_strides(input_shape, input_strides, num_dims);
    compute_strides(output_shape, output_strides, num_dims);
    
    int reduce_size = input_shape[reduce_dim];
    int input_size = compute_size(input_shape, num_dims);
    
    int block_size = BLOCK_SIZE;
    int num_blocks = (input_size + block_size - 1) / block_size;
    
    reduce_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_output, grad_input,
        input_shape, output_shape,
        input_strides, output_strides,
        num_dims, reduce_dim,
        reduce_size, SumBackwardOp()
    );
}

void mean_forward(const float* input, float* output,
                 const int* input_shape, const int* output_shape,
                 int num_dims, int reduce_dim,
                 bool keepdim,
                 cudaStream_t stream) {
    int input_strides[8], output_strides[8];
    compute_strides(input_shape, input_strides, num_dims);
    compute_strides(output_shape, output_strides, num_dims);
    
    int reduce_size = input_shape[reduce_dim];
    int output_size = compute_size(output_shape, num_dims);
    
    int block_size = std::min(BLOCK_SIZE, reduce_size);
    int num_blocks = output_size;
    
    reduce_forward_kernel<<<num_blocks, block_size, block_size * sizeof(float), stream>>>(
        input, output,
        input_shape, output_shape,
        input_strides, output_strides,
        num_dims, reduce_dim,
        reduce_size, MeanOp()
    );
}

void mean_backward(const float* grad_output, float* grad_input,
                  const int* input_shape, const int* output_shape,
                  int num_dims, int reduce_dim,
                  bool keepdim,
                  cudaStream_t stream) {
    int input_strides[8], output_strides[8];
    compute_strides(input_shape, input_strides, num_dims);
    compute_strides(output_shape, output_strides, num_dims);
    
    int reduce_size = input_shape[reduce_dim];
    int input_size = compute_size(input_shape, num_dims);
    
    int block_size = BLOCK_SIZE;
    int num_blocks = (input_size + block_size - 1) / block_size;
    
    reduce_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        grad_output, grad_input,
        input_shape, output_shape,
        input_strides, output_strides,
        num_dims, reduce_dim,
        reduce_size, MeanBackwardOp()
    );
}

} // namespace cuda
} // namespace flash 