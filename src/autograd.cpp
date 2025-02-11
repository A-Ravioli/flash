#include "framework/autograd.h"
#include "custom_kernels/elementwise.cuh"
#include "custom_kernels/matrix.cuh"
#include "custom_kernels/reduction.cuh"
#include <queue>
#include <unordered_set>

namespace flash {

// Node implementation
Node::Node(const Tensor& data) : data_(data) {}

void Node::add_child(const NodePtr& child) {
    children_.push_back(child);
}

const std::vector<NodePtr>& Node::children() const {
    return children_;
}

const Tensor& Node::grad() const {
    return grad_;
}

void Node::set_grad(const Tensor& grad) {
    grad_ = grad;
}

const Tensor& Node::data() const {
    return data_;
}

void Node::set_data(const Tensor& data) {
    data_ = data;
}

bool Node::requires_grad() const {
    return requires_grad_;
}

void Node::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

// Operation nodes
class AddNode : public Node {
public:
    AddNode(const Tensor& a, const Tensor& b) : Node(a), other_(b) {}

    Tensor forward() override {
        Tensor output(data_.shape(), data_.dtype(), data_.device());
        cuda::add_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(output.data()),
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());
        cuda::add_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(grad_.data()),
            nullptr,  // Other gradient will be handled by its own node
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    Tensor other_;
};

class SubNode : public Node {
public:
    SubNode(const Tensor& a, const Tensor& b) : Node(a), other_(b) {}

    Tensor forward() override {
        Tensor output(data_.shape(), data_.dtype(), data_.device());
        cuda::sub_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(output.data()),
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());
        cuda::sub_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(grad_.data()),
            nullptr,  // Other gradient will be handled by its own node
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    Tensor other_;
};

class MulNode : public Node {
public:
    MulNode(const Tensor& a, const Tensor& b) : Node(a), other_(b) {}

    Tensor forward() override {
        Tensor output(data_.shape(), data_.dtype(), data_.device());
        cuda::mul_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(output.data()),
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());
        cuda::mul_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(grad_.data()),
            nullptr,  // Other gradient will be handled by its own node
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    Tensor other_;
};

class DivNode : public Node {
public:
    DivNode(const Tensor& a, const Tensor& b) : Node(a), other_(b) {}

    Tensor forward() override {
        Tensor output(data_.shape(), data_.dtype(), data_.device());
        cuda::div_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(output.data()),
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());
        cuda::div_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(grad_.data()),
            nullptr,  // Other gradient will be handled by its own node
            data_.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    Tensor other_;
};

class MatMulNode : public Node {
public:
    MatMulNode(const Tensor& a, const Tensor& b) : Node(a), other_(b) {
        // Check dimensions
        if (data_.shape().size() != 2 || other_.shape().size() != 2) {
            throw std::runtime_error("MatMul requires 2D tensors");
        }
        if (data_.shape()[1] != other_.shape()[0]) {
            throw std::runtime_error("Incompatible dimensions for matrix multiplication");
        }
    }

    Tensor forward() override {
        int M = data_.shape()[0];
        int K = data_.shape()[1];
        int N = other_.shape()[1];

        std::vector<int64_t> output_shape = {M, N};
        Tensor output(output_shape, data_.dtype(), data_.device());

        cuda::matmul_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(output.data()),
            M, N, K,
            nullptr  // TODO: Add CUDA stream support
        );

        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        int M = data_.shape()[0];
        int K = data_.shape()[1];
        int N = other_.shape()[1];

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());
        cuda::matmul_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(data_.data()),
            static_cast<float*>(other_.data()),
            static_cast<float*>(grad_.data()),
            nullptr,  // Other gradient will be handled by its own node
            M, N, K,
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    Tensor other_;
};

class TransposeNode : public Node {
public:
    TransposeNode(const Tensor& input, int dim0, int dim1)
        : Node(input), dim0_(dim0), dim1_(dim1) {
        if (dim0 >= input.shape().size() || dim1 >= input.shape().size()) {
            throw std::runtime_error("Dimension out of range");
        }
    }

    Tensor forward() override {
        if (dim0_ == dim1_) return data_;  // No-op if dimensions are the same

        // For now, only support 2D transpose
        if (data_.shape().size() != 2) {
            throw std::runtime_error("Only 2D transpose is currently supported");
        }

        int rows = data_.shape()[0];
        int cols = data_.shape()[1];
        std::vector<int64_t> output_shape = {cols, rows};  // Swap dimensions
        Tensor output(output_shape, data_.dtype(), data_.device());

        cuda::transpose_2d(
            static_cast<float*>(data_.data()),
            static_cast<float*>(output.data()),
            rows, cols,
            nullptr  // TODO: Add CUDA stream support
        );

        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        // Transpose the gradient
        int rows = grad_output.shape()[0];
        int cols = grad_output.shape()[1];
        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());

        cuda::transpose_2d(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(grad_.data()),
            rows, cols,
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    int dim0_;
    int dim1_;
};

class SumNode : public Node {
public:
    SumNode(const Tensor& input, int dim, bool keepdim)
        : Node(input), dim_(dim), keepdim_(keepdim) {
        if (dim >= input.shape().size()) {
            throw std::runtime_error("Dimension out of range");
        }
    }

    Tensor forward() override {
        std::vector<int64_t> output_shape = data_.shape();
        if (keepdim_) {
            output_shape[dim_] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim_);
        }

        Tensor output(output_shape, data_.dtype(), data_.device());

        // Convert shapes to int arrays for CUDA kernel
        std::vector<int> input_shape_int(data_.shape().begin(), data_.shape().end());
        std::vector<int> output_shape_int(output_shape.begin(), output_shape.end());

        cuda::sum_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(output.data()),
            input_shape_int.data(),
            output_shape_int.data(),
            data_.shape().size(),
            dim_,
            keepdim_,
            nullptr  // TODO: Add CUDA stream support
        );

        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());

        // Convert shapes to int arrays for CUDA kernel
        std::vector<int> input_shape_int(data_.shape().begin(), data_.shape().end());
        std::vector<int> output_shape_int(grad_output.shape().begin(), grad_output.shape().end());

        cuda::sum_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(grad_.data()),
            input_shape_int.data(),
            output_shape_int.data(),
            data_.shape().size(),
            dim_,
            keepdim_,
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    int dim_;
    bool keepdim_;
};

class MeanNode : public Node {
public:
    MeanNode(const Tensor& input, int dim, bool keepdim)
        : Node(input), dim_(dim), keepdim_(keepdim) {
        if (dim >= input.shape().size()) {
            throw std::runtime_error("Dimension out of range");
        }
    }

    Tensor forward() override {
        std::vector<int64_t> output_shape = data_.shape();
        if (keepdim_) {
            output_shape[dim_] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim_);
        }

        Tensor output(output_shape, data_.dtype(), data_.device());

        // Convert shapes to int arrays for CUDA kernel
        std::vector<int> input_shape_int(data_.shape().begin(), data_.shape().end());
        std::vector<int> output_shape_int(output_shape.begin(), output_shape.end());

        cuda::mean_forward(
            static_cast<float*>(data_.data()),
            static_cast<float*>(output.data()),
            input_shape_int.data(),
            output_shape_int.data(),
            data_.shape().size(),
            dim_,
            keepdim_,
            nullptr  // TODO: Add CUDA stream support
        );

        return output;
    }

    void backward(const Tensor& grad_output) override {
        if (!requires_grad_) return;

        grad_ = Tensor(data_.shape(), data_.dtype(), data_.device());

        // Convert shapes to int arrays for CUDA kernel
        std::vector<int> input_shape_int(data_.shape().begin(), data_.shape().end());
        std::vector<int> output_shape_int(grad_output.shape().begin(), grad_output.shape().end());

        cuda::mean_backward(
            static_cast<float*>(grad_output.data()),
            static_cast<float*>(grad_.data()),
            input_shape_int.data(),
            output_shape_int.data(),
            data_.shape().size(),
            dim_,
            keepdim_,
            nullptr  // TODO: Add CUDA stream support
        );
    }

private:
    int dim_;
    bool keepdim_;
};

// Variable implementation
Variable::Variable(const Tensor& data, bool requires_grad)
    : node_(std::make_shared<Node>(data)) {
    node_->set_requires_grad(requires_grad);
}

Variable::Variable(const Variable& other) : node_(other.node_) {}

Variable::Variable(Variable&& other) noexcept : node_(std::move(other.node_)) {}

Variable& Variable::operator=(const Variable& other) {
    node_ = other.node_;
    return *this;
}

Variable& Variable::operator=(Variable&& other) noexcept {
    node_ = std::move(other.node_);
    return *this;
}

const Tensor& Variable::data() const {
    return node_->data();
}

const Tensor& Variable::grad() const {
    return node_->grad();
}

bool Variable::requires_grad() const {
    return node_->requires_grad();
}

void Variable::zero_grad() {
    if (requires_grad()) {
        Tensor zero_grad(node_->data().shape(), node_->data().dtype(), node_->data().device());
        cuda::fill(
            static_cast<float*>(zero_grad.data()),
            0.0f,
            zero_grad.numel(),
            nullptr  // TODO: Add CUDA stream support
        );
        node_->set_grad(zero_grad);
    }
}

void Variable::backward() {
    if (!requires_grad()) {
        throw std::runtime_error("Variable does not require gradients");
    }

    // Start with gradient of 1 for scalar variables
    if (data().numel() != 1) {
        throw std::runtime_error("backward() can only be called on scalar variables");
    }

    // Initialize gradient to 1
    Tensor grad(data().shape(), data().dtype(), data().device());
    cuda::fill(
        static_cast<float*>(grad.data()),
        1.0f,
        grad.numel(),
        nullptr  // TODO: Add CUDA stream support
    );

    node_->set_grad(grad);

    // Perform backward pass using topological sort
    std::queue<NodePtr> queue;
    std::unordered_set<NodePtr> visited;
    queue.push(node_);

    while (!queue.empty()) {
        NodePtr current = queue.front();
        queue.pop();

        if (visited.find(current) != visited.end()) {
            continue;
        }

        visited.insert(current);

        // Propagate gradients to children
        current->backward(current->grad());

        // Add children to queue
        for (const auto& child : current->children()) {
            if (child->requires_grad() && visited.find(child) == visited.end()) {
                queue.push(child);
            }
        }
    }
}

Variable Variable::operator+(const Variable& other) const {
    auto node = std::make_shared<AddNode>(data(), other.data());
    node->set_requires_grad(requires_grad() || other.requires_grad());
    if (requires_grad()) node->add_child(node_);
    if (other.requires_grad()) node->add_child(other.node_);
    return Variable(node);
}

Variable Variable::operator-(const Variable& other) const {
    auto node = std::make_shared<SubNode>(data(), other.data());
    node->set_requires_grad(requires_grad() || other.requires_grad());
    if (requires_grad()) node->add_child(node_);
    if (other.requires_grad()) node->add_child(other.node_);
    return Variable(node);
}

Variable Variable::operator*(const Variable& other) const {
    auto node = std::make_shared<MulNode>(data(), other.data());
    node->set_requires_grad(requires_grad() || other.requires_grad());
    if (requires_grad()) node->add_child(node_);
    if (other.requires_grad()) node->add_child(other.node_);
    return Variable(node);
}

Variable Variable::operator/(const Variable& other) const {
    auto node = std::make_shared<DivNode>(data(), other.data());
    node->set_requires_grad(requires_grad() || other.requires_grad());
    if (requires_grad()) node->add_child(node_);
    if (other.requires_grad()) node->add_child(other.node_);
    return Variable(node);
}

Variable Variable::matmul(const Variable& other) const {
    auto node = std::make_shared<MatMulNode>(data(), other.data());
    node->set_requires_grad(requires_grad() || other.requires_grad());
    if (requires_grad()) node->add_child(node_);
    if (other.requires_grad()) node->add_child(other.node_);
    return Variable(node);
}

Variable Variable::transpose(int dim0, int dim1) const {
    auto node = std::make_shared<TransposeNode>(data(), dim0, dim1);
    node->set_requires_grad(requires_grad());
    if (requires_grad()) node->add_child(node_);
    return Variable(node);
}

Variable Variable::sum(int dim, bool keepdim) const {
    auto node = std::make_shared<SumNode>(data(), dim, keepdim);
    node->set_requires_grad(requires_grad());
    if (requires_grad()) node->add_child(node_);
    return Variable(node);
}

Variable Variable::mean(int dim, bool keepdim) const {
    auto node = std::make_shared<MeanNode>(data(), dim, keepdim);
    node->set_requires_grad(requires_grad());
    if (requires_grad()) node->add_child(node_);
    return Variable(node);
}

} // namespace flash 