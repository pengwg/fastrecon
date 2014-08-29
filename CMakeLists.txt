project(FastRecon)

cmake_minimum_required(VERSION 2.8.9)

add_definitions(-std=c++11 -Wall -Wno-sign-compare)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(OpenMP)
find_package(CUDA)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

include_directories(
    ImageData/
    ReconData/
    Recon/
    ${CUDA_INCLUDE_DIRS}
)

add_subdirectory(ImageData)
add_subdirectory(ReconData)
add_subdirectory(Recon)

set(CUDA_SRC_FILES
    ReconData/cuReconData.cu
    ImageData/cuImageData.cu
    Recon/GridLut.cu
)

cuda_include_directories(${Qt5Core_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS
    -std=c++11; # Experimental c++11 support for nvcc
#    -gencode;arch=compute_20,code=sm_21;
#    -gencode;arch=compute_30,code=sm_30;
    --ptxas-options=-v;--compiler-options;-Wno-attributes,-Wall
)

cuda_add_library(cuRecon SHARED ${CUDA_SRC_FILES})

set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon recon imageData reconData cuRecon)
qt5_use_modules(fastrecon Core Widgets)
