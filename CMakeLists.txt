project(FastRecon)

cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_BUILD_TYPE Release)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(OpenMP)
find_package(CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(FFTW3 COMPONENTS single REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

if(CUDA_FOUND)
    set(BUILD_CUDA ON CACHE BOOL "")
else(CUDA_FOUND)
    unset(BUILD_CUDA CACHE)
endif(CUDA_FOUND)

include_directories(
    ImageData/
    ReconData/
    ImageProcessor/
    ${FFTW3_INCLUDE_DIR}
)

add_subdirectory(ImageData)
add_subdirectory(ReconData)
add_subdirectory(ImageProcessor)

add_library(Recon SHARED ${IMAGE_SRC_FILES} ${RECON_SRC_FILES} ${IMAGEPROCESSOR_SRC_FILES})
target_link_libraries(Recon ${FFTW3_LIBRARIES} gsl blas)
qt5_use_modules(Recon Core)

set(CMAKE_INSTALL_PREFIX /usr/local/fastrecon)
install(TARGETS Recon LIBRARY DESTINATION lib)

set(RECON_LIB Recon)
if(BUILD_CUDA)
    add_definitions(-DBUILD_CUDA)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS
        -std=c++11;
#        -gencode;arch=compute_50,code=sm_50;
        --ptxas-options=-v;--compiler-options;-Wno-attributes,-Wall
    )
    cuda_include_directories(${Qt5Core_INCLUDE_DIRS})

    cuda_add_library(cuRecon SHARED ${CU_IMAGE_SRC_FILES} ${CU_RECON_SRC_FILES} ${CU_IMAGEPROCESSOR_SRC_FILES})
    cuda_add_cufft_to_target(cuRecon)
    target_link_libraries(cuRecon Recon)
    set_target_properties(cuRecon PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
    qt5_use_modules(cuRecon Core)

    install(TARGETS cuRecon LIBRARY DESTINATION lib)
    set(RECON_LIB cuRecon)
endif(BUILD_CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_subdirectory(Program)

