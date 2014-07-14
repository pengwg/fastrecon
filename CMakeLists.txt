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
)

add_subdirectory(cpu)
add_subdirectory(gpu)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon recon gridGPU)
qt5_use_modules(fastrecon Core Widgets)
