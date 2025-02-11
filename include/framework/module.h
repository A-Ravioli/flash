#pragma once

#include "tensor.h"
#include "autograd.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace flash {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    // Forward pass
    virtual Variable forward(const Variable& input) = 0;
    Variable operator()(const Variable& input);

    // Parameter management
    virtual std::vector<Variable> parameters() const;
    void zero_grad();

    // Training mode
    void train();
    void eval();
    bool is_training() const;

protected:
    bool training_ = true;
    std::vector<Variable> parameters_;
};

class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, bool bias = true);
    
    Variable forward(const Variable& input) override;
    std::vector<Variable> parameters() const override;

private:
    Variable weight_;
    Variable bias_;
    bool use_bias_;
};

class ReLU : public Module {
public:
    ReLU() = default;
    
    Variable forward(const Variable& input) override;
};

class Conv2d : public Module {
public:
    Conv2d(int64_t in_channels, int64_t out_channels, 
           int64_t kernel_size, int64_t stride = 1, 
           int64_t padding = 0, bool bias = true);
    
    Variable forward(const Variable& input) override;
    std::vector<Variable> parameters() const override;

private:
    Variable weight_;
    Variable bias_;
    bool use_bias_;
    int64_t stride_;
    int64_t padding_;
};

class BatchNorm2d : public Module {
public:
    BatchNorm2d(int64_t num_features, double eps = 1e-5, double momentum = 0.1);
    
    Variable forward(const Variable& input) override;
    std::vector<Variable> parameters() const override;

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    Variable weight_;
    Variable bias_;
    Variable running_mean_;
    Variable running_var_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    
    template<typename... Modules>
    Sequential(Modules&&... modules);
    
    void add(std::shared_ptr<Module> module);
    Variable forward(const Variable& input) override;
    std::vector<Variable> parameters() const override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

// Helper function to create modules
template<typename M, typename... Args>
std::shared_ptr<M> make_module(Args&&... args) {
    return std::make_shared<M>(std::forward<Args>(args)...);
}

} // namespace flash 