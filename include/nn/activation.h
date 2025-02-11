#pragma once

#include "framework/module.h"
#include "framework/variable.h"
#include "custom_kernels/elementwise.cuh"

namespace flash {
namespace nn {

class ReLU : public Module {
public:
    Variable forward(const Variable& input) override {
        auto node = std::make_shared<ReLUNode>(input.data());
        node->set_requires_grad(input.requires_grad());
        if (input.requires_grad()) node->add_child(input.node());
        return Variable(node);
    }
};

class Sigmoid : public Module {
public:
    Variable forward(const Variable& input) override {
        auto node = std::make_shared<SigmoidNode>(input.data());
        node->set_requires_grad(input.requires_grad());
        if (input.requires_grad()) node->add_child(input.node());
        return Variable(node);
    }
};

class Tanh : public Module {
public:
    Variable forward(const Variable& input) override {
        auto node = std::make_shared<TanhNode>(input.data());
        node->set_requires_grad(input.requires_grad());
        if (input.requires_grad()) node->add_child(input.node());
        return Variable(node);
    }
};

class LeakyReLU : public Module {
public:
    LeakyReLU(float negative_slope = 0.01) : negative_slope_(negative_slope) {}

    Variable forward(const Variable& input) override {
        auto node = std::make_shared<LeakyReLUNode>(input.data(), negative_slope_);
        node->set_requires_grad(input.requires_grad());
        if (input.requires_grad()) node->add_child(input.node());
        return Variable(node);
    }

private:
    float negative_slope_;
};

} // namespace nn
} // namespace flash 