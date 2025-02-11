#include "framework/optimizer.h"
#include <cmath>

namespace flash {

// Base Optimizer implementation
Optimizer::Optimizer(const std::vector<Variable>& parameters, double learning_rate)
    : parameters_(parameters), learning_rate_(learning_rate) {}

void Optimizer::zero_grad() {
    for (auto& param : parameters_) {
        param.zero_grad();
    }
}

// SGD implementation
SGD::SGD(const std::vector<Variable>& parameters,
         double learning_rate,
         double momentum,
         double weight_decay)
    : Optimizer(parameters, learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay) {
    if (momentum_ > 0) {
        momentum_buffers_.resize(parameters_.size());
        for (size_t i = 0; i < parameters_.size(); ++i) {
            momentum_buffers_[i] = Tensor(parameters_[i].data().shape(),
                                        parameters_[i].data().dtype(),
                                        parameters_[i].data().device());
        }
    }
}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i].requires_grad()) continue;

        const Tensor& grad = parameters_[i].grad();
        Tensor& param_data = const_cast<Tensor&>(parameters_[i].data());

        if (weight_decay_ > 0) {
            // TODO: Add weight decay to gradient
        }

        if (momentum_ > 0) {
            // TODO: Update with momentum
            // v = momentum * v + grad
            // param -= learning_rate * v
        } else {
            // TODO: Simple SGD update
            // param -= learning_rate * grad
        }
    }
}

// Adam implementation
Adam::Adam(const std::vector<Variable>& parameters,
          double learning_rate,
          double beta1,
          double beta2,
          double eps,
          double weight_decay)
    : Optimizer(parameters, learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      step_count_(0) {
    moment1_buffers_.resize(parameters_.size());
    moment2_buffers_.resize(parameters_.size());
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        moment1_buffers_[i] = Tensor(parameters_[i].data().shape(),
                                   parameters_[i].data().dtype(),
                                   parameters_[i].data().device());
        moment2_buffers_[i] = Tensor(parameters_[i].data().shape(),
                                   parameters_[i].data().dtype(),
                                   parameters_[i].data().device());
    }
}

void Adam::step() {
    ++step_count_;
    double step_size = learning_rate_ * std::sqrt(1.0 - std::pow(beta2_, step_count_)) /
                      (1.0 - std::pow(beta1_, step_count_));

    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i].requires_grad()) continue;

        const Tensor& grad = parameters_[i].grad();
        Tensor& param_data = const_cast<Tensor&>(parameters_[i].data());

        if (weight_decay_ > 0) {
            // TODO: Add weight decay to gradient
        }

        // TODO: Update moments and parameters
        // m = beta1 * m + (1 - beta1) * grad
        // v = beta2 * v + (1 - beta2) * grad * grad
        // param -= step_size * m / (sqrt(v) + eps)
    }
}

// AdamW implementation
AdamW::AdamW(const std::vector<Variable>& parameters,
             double learning_rate,
             double beta1,
             double beta2,
             double eps,
             double weight_decay)
    : Optimizer(parameters, learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      step_count_(0) {
    moment1_buffers_.resize(parameters_.size());
    moment2_buffers_.resize(parameters_.size());
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        moment1_buffers_[i] = Tensor(parameters_[i].data().shape(),
                                   parameters_[i].data().dtype(),
                                   parameters_[i].data().device());
        moment2_buffers_[i] = Tensor(parameters_[i].data().shape(),
                                   parameters_[i].data().dtype(),
                                   parameters_[i].data().device());
    }
}

void AdamW::step() {
    ++step_count_;
    double step_size = learning_rate_ * std::sqrt(1.0 - std::pow(beta2_, step_count_)) /
                      (1.0 - std::pow(beta1_, step_count_));

    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i].requires_grad()) continue;

        const Tensor& grad = parameters_[i].grad();
        Tensor& param_data = const_cast<Tensor&>(parameters_[i].data());

        // TODO: AdamW weight decay (different from Adam)
        // param -= learning_rate * weight_decay * param

        // TODO: Update moments and parameters
        // m = beta1 * m + (1 - beta1) * grad
        // v = beta2 * v + (1 - beta2) * grad * grad
        // param -= step_size * m / (sqrt(v) + eps)
    }
}

} // namespace flash 