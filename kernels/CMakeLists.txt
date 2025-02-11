set(CUDA_SOURCES
    single_gpu/matmul.cu
    single_gpu/elementwise.cu
    single_gpu/matrix.cu
    single_gpu/reduction.cu
)

cuda_add_library(flash_kernels STATIC ${CUDA_SOURCES})
target_include_directories(flash_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(flash_kernels ${CUDA_LIBRARIES}) 