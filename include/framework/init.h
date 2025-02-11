#pragma once

#include "framework/tensor.h"
#include <random>

namespace flash {

class Init {
public:
    // Kaiming initialization (He initialization)
    static Tensor kaiming_uniform(const std::vector<int64_t>& shape, float gain = 1.0f) {
        float fan_in = compute_fan_in(shape);
        float bound = gain * std::sqrt(1.0f / fan_in);
        return uniform(shape, -bound, bound);
    }

    static Tensor kaiming_normal(const std::vector<int64_t>& shape, float gain = 1.0f) {
        float fan_in = compute_fan_in(shape);
        float std = gain * std::sqrt(1.0f / fan_in);
        return normal(shape, 0.0f, std);
    }

    // Xavier initialization (Glorot initialization)
    static Tensor xavier_uniform(const std::vector<int64_t>& shape, float gain = 1.0f) {
        float fan_in = compute_fan_in(shape);
        float fan_out = compute_fan_out(shape);
        float bound = gain * std::sqrt(6.0f / (fan_in + fan_out));
        return uniform(shape, -bound, bound);
    }

    static Tensor xavier_normal(const std::vector<int64_t>& shape, float gain = 1.0f) {
        float fan_in = compute_fan_in(shape);
        float fan_out = compute_fan_out(shape);
        float std = gain * std::sqrt(2.0f / (fan_in + fan_out));
        return normal(shape, 0.0f, std);
    }

    // Basic initialization methods
    static Tensor uniform(const std::vector<int64_t>& shape, float a = 0.0f, float b = 1.0f) {
        Tensor tensor(shape, DType::kFloat32, Device::CUDA);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(a, b);
        
        int size = tensor.numel();
        std::vector<float> host_data(size);
        for (int i = 0; i < size; ++i) {
            host_data[i] = dis(gen);
        }
        
        tensor.copy_from_host(host_data.data());
        return tensor;
    }

    static Tensor normal(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f) {
        Tensor tensor(shape, DType::kFloat32, Device::CUDA);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(mean, std);
        
        int size = tensor.numel();
        std::vector<float> host_data(size);
        for (int i = 0; i < size; ++i) {
            host_data[i] = dis(gen);
        }
        
        tensor.copy_from_host(host_data.data());
        return tensor;
    }

    static Tensor constant(const std::vector<int64_t>& shape, float value) {
        Tensor tensor(shape, DType::kFloat32, Device::CUDA);
        int size = tensor.numel();
        std::vector<float> host_data(size, value);
        tensor.copy_from_host(host_data.data());
        return tensor;
    }

    static Tensor zeros(const std::vector<int64_t>& shape) {
        return constant(shape, 0.0f);
    }

    static Tensor ones(const std::vector<int64_t>& shape) {
        return constant(shape, 1.0f);
    }

private:
    static float compute_fan_in(const std::vector<int64_t>& shape) {
        if (shape.size() <= 1) return shape[0];
        if (shape.size() == 2) return shape[1];  // Linear layer
        // Conv2d layer
        int64_t fan_in = shape[1];  // in_channels
        for (size_t i = 2; i < shape.size(); ++i) {
            fan_in *= shape[i];  // kernel dimensions
        }
        return fan_in;
    }

    static float compute_fan_out(const std::vector<int64_t>& shape) {
        if (shape.size() <= 1) return shape[0];
        if (shape.size() == 2) return shape[0];  // Linear layer
        // Conv2d layer
        int64_t fan_out = shape[0];  // out_channels
        for (size_t i = 2; i < shape.size(); ++i) {
            fan_out *= shape[i];  // kernel dimensions
        }
        return fan_out;
    }
};

} // namespace flash 