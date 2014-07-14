set(GRID_GPU_SRC_FILES
#    GridGpu.cpp
#    GridGpu.cu
#    FFTGpu.cpp
    cuReconData.cpp
)

include_directories(
    #${CUDA_INCLUDE_DIRS}
)

#cuda_include_directories(gridGpu ${Qt5Core_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS
    -gencode;arch=compute_20,code=sm_21;
    -gencode;arch=compute_30,code=sm_30;
    --ptxas-options=-v
)

#add_library(gridGPU SHARED ${GRID_GPU_SRC_FILES})


cuda_add_library(gridGPU SHARED ${GRID_GPU_SRC_FILES})
#cuda_add_cufft_to_target(gridGPU)
qt5_use_modules(gridGPU Core)
