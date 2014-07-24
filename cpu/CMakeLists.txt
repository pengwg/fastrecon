set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(FFTW3 COMPONENTS single REQUIRED)

set(CPU_SRC_FILES
    GridLut.cpp
    FFT.cpp
    ImageRecon.cpp
    hostImageData.h
    hostImageData.inl
    hostReconData.h
    hostReconData.inl
    ../common/ConvKernel.cpp
    ../common/basicImageData.cpp
    ../common/ImageData.cpp
    ../common/basicReconData.cpp
    ../common/ReconData.cpp
    ../common/common.h
)

include_directories(
    ${FFTW3_INCLUDE_DIR}
)

add_library(recon SHARED ${CPU_SRC_FILES})

target_link_libraries(recon gsl blas ${FFTW3_LIBRARIES})
qt5_use_modules(recon Core)
