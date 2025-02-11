set(SRC_FILES
    tensor.cpp
    autograd.cpp
    module.cpp
    optimizer.cpp
    device.cpp
    distributed.cpp
)

add_library(flash_core STATIC ${SRC_FILES})
target_include_directories(flash_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(flash_core ${CUDA_LIBRARIES}) 