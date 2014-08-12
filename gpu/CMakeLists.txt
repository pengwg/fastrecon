set(GRID_GPU_SRC_FILES
    basicReconData.cu
    GridLut.cu
)

include_directories(
    #${CUDA_INCLUDE_DIRS}
)

cuda_include_directories(gridGpu ${Qt5Core_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS
    -std=c++11; # Experimental c++11 support for nvcc
    -gencode;arch=compute_20,code=sm_21;
    -gencode;arch=compute_30,code=sm_30;
    --ptxas-options=-v;--compiler-options;-Wno-attributes,-Wall
)

#add_library(gridGPU SHARED ${GRID_GPU_SRC_FILES})
cuda_add_library(gridGPU SHARED ${GRID_GPU_SRC_FILES})
#cuda_add_cufft_to_target(gridGPU)
qt5_use_modules(gridGPU Core)
