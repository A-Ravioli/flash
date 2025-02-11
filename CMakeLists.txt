cmake_minimum_required(VERSION 3.15)
project(flash VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/framework
    ${PROJECT_SOURCE_DIR}/include/custom_kernels
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(kernels)
add_subdirectory(examples)
add_subdirectory(tests)

# Set compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
endif()

# Create the main library target
add_library(flash
    src/tensor.cpp
    src/autograd.cpp
    src/module.cpp
    src/optimizer.cpp
    src/device.cpp
    src/distributed.cpp
)

# Link against CUDA
target_link_libraries(flash ${CUDA_LIBRARIES})

# Installation rules
install(TARGETS flash
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
) 