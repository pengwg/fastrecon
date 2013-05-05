set(GRID_GPU_SRC_FILES
    GridGpu.cpp
    GridGpu.cu
    FFTGpu.cpp
)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)

set(CUDA_NVCC_FLAGS -gencode;arch=compute_20,code=sm_20;--ptxas-options=-v)

cuda_add_library(gridGPU SHARED ${GRID_GPU_SRC_FILES})
cuda_add_cufft_to_target(gridGPU)
