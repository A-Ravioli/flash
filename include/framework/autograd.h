#pragma once

#include "tensor.h"
#include <memory>
#include <vector>
#include <functional>

namespace flash {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Node {
public:
    Node(const Tensor& data);
    virtual ~Node() = default;

    // Forward and backward operations
    virtual Tensor forward() = 0;
    virtual void backward(const Tensor& grad) = 0;

    // Graph building
    void add_child(const NodePtr& child);
    const std::vector<NodePtr>& children() const;

    // Gradient access
    const Tensor& grad() const;
    void set_grad(const Tensor& grad);

    // Data access
    const Tensor& data() const;
    void set_data(const Tensor& data);

    // Requires grad flag
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);

protected:
    Tensor data_;
    Tensor grad_;
    std::vector<NodePtr> children_;
    bool requires_grad_ = false;
};

class Variable {
public:
    Variable(const Tensor& data, bool requires_grad = false);
    Variable(const Variable& other);
    Variable(Variable&& other) noexcept;

    Variable& operator=(const Variable& other);
    Variable& operator=(Variable&& other) noexcept;

    // Basic arithmetic operations
    Variable operator+(const Variable& other) const;
    Variable operator-(const Variable& other) const;
    Variable operator*(const Variable& other) const;
    Variable operator/(const Variable& other) const;

    // Matrix operations
    Variable matmul(const Variable& other) const;
    Variable transpose(int dim0, int dim1) const;

    // Reduction operations
    Variable sum(int dim = -1, bool keepdim = false) const;
    Variable mean(int dim = -1, bool keepdim = false) const;

    // Gradient computation
    void backward();
    void zero_grad();

    // Access underlying data
    const Tensor& data() const;
    const Tensor& grad() const;
    bool requires_grad() const;

private:
    NodePtr node_;
};

} // namespace flash 