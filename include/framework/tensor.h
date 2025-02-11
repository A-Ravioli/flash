#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace flash {

enum class DeviceType {
    CPU,
    CUDA
};

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64
};

class TensorImpl;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32, DeviceType device = DeviceType::CPU);
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor();

    // Basic info
    const std::vector<int64_t>& shape() const;
    int64_t dim() const;
    int64_t numel() const;
    DataType dtype() const;
    DeviceType device() const;

    // Device transfer
    Tensor to(DeviceType device) const;
    Tensor cuda() const;
    Tensor cpu() const;

    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    // In-place operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim0, int dim1) const;

    // Reduction operations
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;

    // Memory access
    void* data() const;
    template<typename T>
    T* data_ptr() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace flash 