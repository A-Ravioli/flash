#include "framework/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>
#include <functional>

namespace flash {

class TensorImpl {
public:
    TensorImpl() : device_(DeviceType::CPU), dtype_(DataType::FLOAT32) {}
    
    TensorImpl(const std::vector<int64_t>& shape, DataType dtype, DeviceType device)
        : shape_(shape), device_(device), dtype_(dtype) {
        allocateMemory();
    }
    
    ~TensorImpl() {
        freeMemory();
    }

    void allocateMemory() {
        size_t size = std::accumulate(shape_.begin(), shape_.end(), 
                                    static_cast<size_t>(1), std::multiplies<size_t>());
        size_t bytes = size * elementSize();
        
        if (device_ == DeviceType::CPU) {
            data_ = malloc(bytes);
            if (!data_) throw std::runtime_error("CPU memory allocation failed");
        } else {
            cudaError_t err = cudaMalloc(&data_, bytes);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memory allocation failed");
            }
        }
    }

    void freeMemory() {
        if (data_) {
            if (device_ == DeviceType::CPU) {
                free(data_);
            } else {
                cudaFree(data_);
            }
            data_ = nullptr;
        }
    }

    size_t elementSize() const {
        switch (dtype_) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT16: return 2;
            case DataType::INT32: return 4;
            case DataType::INT64: return 8;
            default: throw std::runtime_error("Unknown data type");
        }
    }

    std::vector<int64_t> shape_;
    DeviceType device_;
    DataType dtype_;
    void* data_ = nullptr;
};

// Tensor implementation

Tensor::Tensor() : impl_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype, DeviceType device)
    : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}

Tensor::Tensor(const Tensor& other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    impl_ = other.impl_;
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}

Tensor::~Tensor() = default;

const std::vector<int64_t>& Tensor::shape() const {
    return impl_->shape_;
}

int64_t Tensor::dim() const {
    return impl_->shape_.size();
}

int64_t Tensor::numel() const {
    return std::accumulate(impl_->shape_.begin(), impl_->shape_.end(), 
                         static_cast<int64_t>(1), std::multiplies<int64_t>());
}

DataType Tensor::dtype() const {
    return impl_->dtype_;
}

DeviceType Tensor::device() const {
    return impl_->device_;
}

void* Tensor::data() const {
    return impl_->data_;
}

template<typename T>
T* Tensor::data_ptr() const {
    return static_cast<T*>(impl_->data_);
}

Tensor Tensor::to(DeviceType target_device) const {
    if (device() == target_device) return *this;
    
    Tensor result(shape(), dtype(), target_device);
    size_t bytes = numel() * impl_->elementSize();
    
    if (device() == DeviceType::CPU && target_device == DeviceType::CUDA) {
        cudaError_t err = cudaMemcpy(result.data(), data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Host to device transfer failed");
        }
    } else if (device() == DeviceType::CUDA && target_device == DeviceType::CPU) {
        cudaError_t err = cudaMemcpy(result.data(), data(), bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Device to host transfer failed");
        }
    }
    
    return result;
}

Tensor Tensor::cuda() const {
    return to(DeviceType::CUDA);
}

Tensor Tensor::cpu() const {
    return to(DeviceType::CPU);
}

// Note: Basic operations will be implemented later with CUDA kernels
// For now, we'll just throw "not implemented" exceptions

Tensor Tensor::operator+(const Tensor& other) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::operator-(const Tensor& other) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::operator*(const Tensor& other) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::operator/(const Tensor& other) const {
    throw std::runtime_error("Not implemented");
}

Tensor& Tensor::operator+=(const Tensor& other) {
    throw std::runtime_error("Not implemented");
}

Tensor& Tensor::operator-=(const Tensor& other) {
    throw std::runtime_error("Not implemented");
}

Tensor& Tensor::operator*=(const Tensor& other) {
    throw std::runtime_error("Not implemented");
}

Tensor& Tensor::operator/=(const Tensor& other) {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::matmul(const Tensor& other) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    throw std::runtime_error("Not implemented");
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    throw std::runtime_error("Not implemented");
}

} // namespace flash 