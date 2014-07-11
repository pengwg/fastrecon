project(FastRecon)

cmake_minimum_required(VERSION 2.8.9)

add_definitions(-std=c++11 -Wall -Wno-sign-compare)

set(CMAKE_BUILD_TYPE Release)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

include_directories(
    common/
    cpu/
)

set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
)

find_package(CUDA)

add_subdirectory(cpu)
add_subdirectory(gpu)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon recon)
qt5_use_modules(fastrecon Core Widgets)
