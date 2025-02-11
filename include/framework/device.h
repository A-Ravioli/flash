#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace flash {

class StreamManager {
public:
    static StreamManager& instance() {
        static StreamManager instance;
        return instance;
    }

    cudaStream_t get_stream(int device_id = 0) {
        if (device_id >= streams_.size()) {
            resize(device_id + 1);
        }
        return streams_[device_id];
    }

    void synchronize(int device_id = 0) {
        if (device_id < streams_.size()) {
            cudaStreamSynchronize(streams_[device_id]);
        }
    }

    void synchronize_all() {
        for (size_t i = 0; i < streams_.size(); ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
    }

private:
    StreamManager() {
        resize(1);  // Create stream for device 0 by default
    }

    ~StreamManager() {
        for (auto stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }

    void resize(size_t new_size) {
        size_t old_size = streams_.size();
        streams_.resize(new_size);
        for (size_t i = old_size; i < new_size; ++i) {
            cudaStreamCreate(&streams_[i]);
        }
    }

    std::vector<cudaStream_t> streams_;
};

enum class DeviceType {
    CPU,
    CUDA
};

class Device {
public:
    static Device& instance() {
        static Device instance;
        return instance;
    }

    void set_device(int device_id) {
        current_device_ = device_id;
        cudaSetDevice(device_id);
    }

    int current_device() const { return current_device_; }

    cudaStream_t current_stream() {
        return StreamManager::instance().get_stream(current_device_);
    }

    void synchronize() {
        StreamManager::instance().synchronize(current_device_);
    }

private:
    Device() : current_device_(0) {
        cudaSetDevice(0);
    }

    int current_device_;
};

} // namespace flash 