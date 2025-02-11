#pragma once

#include "framework/module.h"
#include "framework/init.h"
#include "framework/variable.h"

namespace flash {
namespace nn {

class BatchNorm2d : public Module {
public:
    BatchNorm2d(int num_features, float eps = 1e-5, float momentum = 0.1)
        : num_features_(num_features), eps_(eps), momentum_(momentum) {
        
        // Create parameters
        weight_ = register_parameter("weight", Init::ones({num_features}));
        bias_ = register_parameter("bias", Init::zeros({num_features}));
        
        // Create buffers for running statistics
        running_mean_ = register_buffer("running_mean", Init::zeros({num_features}));
        running_var_ = register_buffer("running_var", Init::ones({num_features}));
    }

    Variable forward(const Variable& input) override {
        if (input.data().shape().size() != 4) {
            throw std::runtime_error("BatchNorm2d expects 4D input");
        }
        if (input.data().shape()[1] != num_features_) {
            throw std::runtime_error("Number of features doesn't match");
        }

        int batch_size = input.data().shape()[0];
        int channels = input.data().shape()[1];
        int height = input.data().shape()[2];
        int width = input.data().shape()[3];

        if (training()) {
            // Calculate batch statistics
            auto dims = std::vector<int>{0, 2, 3};  // Average over batch, height, width
            auto batch_mean = input.mean(dims, true);
            auto centered = input - batch_mean;
            auto batch_var = (centered * centered).mean(dims, true);
            
            // Update running statistics
            running_mean_ = running_mean_ * momentum_ + batch_mean.data() * (1 - momentum_);
            running_var_ = running_var_ * momentum_ + batch_var.data() * (1 - momentum_);
            
            // Normalize
            auto std = (batch_var + eps_).sqrt();
            auto normalized = centered / std;
            
            // Scale and shift
            return normalized * weight_ + bias_;
        } else {
            // Use running statistics for inference
            auto normalized = (input - running_mean_) / (running_var_ + eps_).sqrt();
            return normalized * weight_ + bias_;
        }
    }

private:
    int num_features_;
    float eps_;
    float momentum_;
    
    Variable weight_;
    Variable bias_;
    Tensor running_mean_;
    Tensor running_var_;
};

} // namespace nn
} // namespace flash 