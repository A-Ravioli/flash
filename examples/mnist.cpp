#include "framework/tensor.h"
#include "framework/autograd.h"
#include "framework/module.h"
#include "framework/optimizer.h"
#include <iostream>
#include <vector>

using namespace flash;

class MNISTNet : public Module {
public:
    MNISTNet() {
        // Input: 1x28x28
        conv1_ = make_module<Conv2d>(1, 32, 3, 1, 1);  // Output: 32x28x28
        conv2_ = make_module<Conv2d>(32, 64, 3, 1, 1); // Output: 64x28x28
        fc1_ = make_module<Linear>(64 * 7 * 7, 128);   // After 2 max pools: 64x7x7
        fc2_ = make_module<Linear>(128, 10);
        
        parameters_.insert(parameters_.end(), conv1_->parameters().begin(), conv1_->parameters().end());
        parameters_.insert(parameters_.end(), conv2_->parameters().begin(), conv2_->parameters().end());
        parameters_.insert(parameters_.end(), fc1_->parameters().begin(), fc1_->parameters().end());
        parameters_.insert(parameters_.end(), fc2_->parameters().begin(), fc2_->parameters().end());
    }
    
    Variable forward(const Variable& x) override {
        // First conv block
        auto out = conv1_->forward(x);
        // TODO: Add ReLU and MaxPool2d
        
        // Second conv block
        out = conv2_->forward(out);
        // TODO: Add ReLU and MaxPool2d
        
        // Flatten
        std::vector<int64_t> shape = out.data().shape();
        // TODO: Reshape to (batch_size, 64 * 7 * 7)
        
        // Fully connected layers
        out = fc1_->forward(out);
        // TODO: Add ReLU
        out = fc2_->forward(out);
        
        return out;
    }

private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

int main() {
    try {
        // Create model
        auto model = std::make_shared<MNISTNet>();
        model->train();
        
        // Create optimizer
        Adam optimizer(model->parameters(), 0.001);
        
        // TODO: Load MNIST dataset
        
        // Training loop
        int num_epochs = 10;
        int batch_size = 32;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // TODO: Iterate over batches
            {
                // Forward pass
                // TODO: Get batch data
                Variable images(Tensor({batch_size, 1, 28, 28}));  // Placeholder
                Variable targets(Tensor({batch_size}));            // Placeholder
                
                Variable outputs = model->forward(images);
                // TODO: Compute loss
                
                // Backward pass
                optimizer.zero_grad();
                // loss.backward();
                optimizer.step();
                
                // TODO: Print statistics
            }
            
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << " completed\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 