#pragma once

#include "tensor.h"
#include "autograd.h"
#include <vector>
#include <memory>

namespace flash {

class Optimizer {
public:
    Optimizer(const std::vector<Variable>& parameters, double learning_rate = 0.01);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();

protected:
    std::vector<Variable> parameters_;
    double learning_rate_;
};

class SGD : public Optimizer {
public:
    SGD(const std::vector<Variable>& parameters, 
        double learning_rate = 0.01,
        double momentum = 0.0,
        double weight_decay = 0.0);

    void step() override;

private:
    double momentum_;
    double weight_decay_;
    std::vector<Tensor> momentum_buffers_;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<Variable>& parameters,
         double learning_rate = 0.001,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0);

    void step() override;

private:
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    int64_t step_count_;
    std::vector<Tensor> moment1_buffers_;
    std::vector<Tensor> moment2_buffers_;
};

class AdamW : public Optimizer {
public:
    AdamW(const std::vector<Variable>& parameters,
          double learning_rate = 0.001,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.01);

    void step() override;

private:
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    int64_t step_count_;
    std::vector<Tensor> moment1_buffers_;
    std::vector<Tensor> moment2_buffers_;
};

} // namespace flash 