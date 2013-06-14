set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    #add_definitions(${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

find_package(FFTW3 COMPONENTS single REQUIRED)

set(GRID_CPU_SRC_FILES
    GridLut.cpp
    FFT.cpp
    ../common/ConvKernel.cpp
    ../common/ReconData.cpp
)

include_directories(
    ${FFTW3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/common/
)

add_library(gridCPU SHARED ${GRID_CPU_SRC_FILES})

target_link_libraries(gridCPU gsl blas ${FFTW3_LIBRARIES})
qt5_use_modules(gridCPU Core)
