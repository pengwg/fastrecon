project(FastRecon)

cmake_minimum_required(VERSION 2.8.9)

add_definitions(-std=c++11 -Wall)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(OpenMP)
find_package(CUDA)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

set(CMAKE_INSTALL_PREFIX /usr/local/fastrecon)

if(CUDA_FOUND)
    set(BUILD_CUDA ON CACHE BOOL "")
else(CUDA_FOUND)
    unset(BUILD_CUDA CACHE)
endif(CUDA_FOUND)

include_directories(
    ImageData/
    ReconData/
    Recon/
)

add_subdirectory(ImageData)
add_subdirectory(ReconData)
add_subdirectory(Recon)

set(RECON_LIB Recon)

if(BUILD_CUDA)
    cuda_include_directories(${Qt5Core_INCLUDE_DIRS})
    add_definitions(-DBUILD_CUDA)
    set(CUDA_NVCC_FLAGS
        -std=c++11; # Experimental c++11 support for nvcc
        -DBUILD_CUDA
#       -gencode;arch=compute_20,code=sm_21;
#       -gencode;arch=compute_30,code=sm_30;
        --ptxas-options=-v;--compiler-options;-Wno-attributes,-Wall
    )

    set(CUDA_SRC_FILES
        ReconData/cuReconData.cu
        ImageData/cuImageData.cu
        Recon/cuGridLut.cu
        Recon/cuFFT.cpp
    )
    cuda_add_library(cuRecon SHARED ${CUDA_SRC_FILES})
    cuda_add_cufft_to_target(cuRecon)
    target_link_libraries(cuRecon Recon)
    set_target_properties(cuRecon PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
    qt5_use_modules(cuRecon Core)

    install(TARGETS cuRecon LIBRARY DESTINATION lib)
    set(RECON_LIB cuRecon)
endif(BUILD_CUDA)

set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon ${RECON_LIB})

qt5_use_modules(fastrecon Core Widgets)

