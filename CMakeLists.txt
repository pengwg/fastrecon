project(FastRecon)

cmake_minimum_required(VERSION 2.8.9)

add_definitions( -std=c++11 -Wall -Wno-sign-compare)

set(CMAKE_BUILD_TYPE Release)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
include_directories(
    ${CMAKE_SOURCE_DIR}/common/
    cpu/
)

set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
)

find_package(CUDA)
if(CUDA_FOUND)
    set(CUDA_HOST_COMPILER /usr/bin/gcc-4.6)
#    add_definitions(-DCUDA_CAPABLE)
#    add_subdirectory(gpu)
endif(CUDA_FOUND)

add_subdirectory(cpu)

add_executable(fastrecon ${MAIN_SRC_FILES})

target_link_libraries(fastrecon gridCPU)
qt5_use_modules(fastrecon Core Widgets)
