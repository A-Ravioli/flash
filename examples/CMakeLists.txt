add_executable(mnist_example mnist.cpp)
target_link_libraries(mnist_example flash flash_core flash_kernels ${CUDA_LIBRARIES})
target_include_directories(mnist_example PUBLIC ${PROJECT_SOURCE_DIR}/include) 