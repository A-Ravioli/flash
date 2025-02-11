#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "framework/tensor.h"
#include "framework/autograd.h"
#include "framework/module.h"
#include "framework/optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "flash: A hardware-optimized ML framework";

    // Device submodule
    py::module device = m.def_submodule("device", "Device management");
    py::enum_<flash::DeviceType>(device, "DeviceType")
        .value("CPU", flash::DeviceType::CPU)
        .value("CUDA", flash::DeviceType::CUDA);

    // Tensor class
    py::class_<flash::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int64_t>&, flash::DataType, flash::DeviceType>())
        .def("shape", &flash::Tensor::shape)
        .def("dim", &flash::Tensor::dim)
        .def("numel", &flash::Tensor::numel)
        .def("dtype", &flash::Tensor::dtype)
        .def("device", &flash::Tensor::device)
        .def("cuda", &flash::Tensor::cuda)
        .def("cpu", &flash::Tensor::cpu)
        // Arithmetic operations
        .def("__add__", [](const flash::Tensor& a, const flash::Tensor& b) { return a + b; })
        .def("__sub__", [](const flash::Tensor& a, const flash::Tensor& b) { return a - b; })
        .def("__mul__", [](const flash::Tensor& a, const flash::Tensor& b) { return a * b; })
        .def("__truediv__", [](const flash::Tensor& a, const flash::Tensor& b) { return a / b; })
        // In-place operations
        .def("__iadd__", [](flash::Tensor& a, const flash::Tensor& b) { return a += b; })
        .def("__isub__", [](flash::Tensor& a, const flash::Tensor& b) { return a -= b; })
        .def("__imul__", [](flash::Tensor& a, const flash::Tensor& b) { return a *= b; })
        .def("__itruediv__", [](flash::Tensor& a, const flash::Tensor& b) { return a /= b; })
        // Matrix operations
        .def("matmul", &flash::Tensor::matmul)
        .def("transpose", &flash::Tensor::transpose)
        // Reduction operations
        .def("sum", &flash::Tensor::sum)
        .def("mean", &flash::Tensor::mean);

    // Variable class (autograd)
    py::class_<flash::Variable>(m, "Variable")
        .def(py::init<const flash::Tensor&, bool>())
        .def("backward", &flash::Variable::backward)
        .def("zero_grad", &flash::Variable::zero_grad)
        .def("data", &flash::Variable::data)
        .def("grad", &flash::Variable::grad)
        .def("requires_grad", &flash::Variable::requires_grad);

    // Module base class
    py::class_<flash::Module, std::shared_ptr<flash::Module>>(m, "Module")
        .def("forward", &flash::Module::forward)
        .def("__call__", &flash::Module::operator())
        .def("parameters", &flash::Module::parameters)
        .def("zero_grad", &flash::Module::zero_grad)
        .def("train", &flash::Module::train)
        .def("eval", &flash::Module::eval)
        .def("is_training", &flash::Module::is_training);

    // Linear layer
    py::class_<flash::Linear, flash::Module, std::shared_ptr<flash::Linear>>(m, "Linear")
        .def(py::init<int64_t, int64_t, bool>())
        .def("forward", &flash::Linear::forward);

    // ReLU activation
    py::class_<flash::ReLU, flash::Module, std::shared_ptr<flash::ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &flash::ReLU::forward);

    // Conv2d layer
    py::class_<flash::Conv2d, flash::Module, std::shared_ptr<flash::Conv2d>>(m, "Conv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, bool>())
        .def("forward", &flash::Conv2d::forward);

    // BatchNorm2d layer
    py::class_<flash::BatchNorm2d, flash::Module, std::shared_ptr<flash::BatchNorm2d>>(m, "BatchNorm2d")
        .def(py::init<int64_t, double, double>())
        .def("forward", &flash::BatchNorm2d::forward);

    // Optimizer base class
    py::class_<flash::Optimizer>(m, "Optimizer")
        .def("step", &flash::Optimizer::step)
        .def("zero_grad", &flash::Optimizer::zero_grad);

    // SGD optimizer
    py::class_<flash::SGD, flash::Optimizer>(m, "SGD")
        .def(py::init<const std::vector<flash::Variable>&, double, double, double>());

    // Adam optimizer
    py::class_<flash::Adam, flash::Optimizer>(m, "Adam")
        .def(py::init<const std::vector<flash::Variable>&, double, double, double, double, double>());

    // Utility functions
    m.def("tensor", [](py::array_t<float> array) {
        py::buffer_info buf = array.request();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < buf.ndim; ++i) {
            shape.push_back(buf.shape[i]);
        }
        flash::Tensor tensor(shape);
        std::memcpy(tensor.data(), buf.ptr, buf.size * sizeof(float));
        return tensor;
    });
} 