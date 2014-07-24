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
    common/
    cpu/
    gpu/
    ${CUDA_INCLUDE_DIRS}
)

set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
# Include the following files to show them in QtCreator project
    cpu/hostImageData.h
    cpu/hostImageData.inl
    cpu/hostReconData.h
    cpu/hostReconData.inl
    gpu/cuImageData.h
    gpu/cuImageData.inl
    gpu/cuReconData.h
    gpu/cuReconData.inl
)

add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(gpu)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon common recon gridGPU)
qt5_use_modules(fastrecon Core Widgets)
