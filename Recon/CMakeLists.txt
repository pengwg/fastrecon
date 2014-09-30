set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(FFTW3 COMPONENTS single REQUIRED)

set(SRC_FILES
    GridLut.cpp
    FFT.cpp
    ImageRecon.cpp
    ConvKernel.cpp
    cuFFT.cpp

    cuGridLut.h
)

include_directories(
    ${FFTW3_INCLUDE_DIR}
)

add_library(recon STATIC ${SRC_FILES})

target_link_libraries(recon ${FFTW3_LIBRARIES} gsl blas)
qt5_use_modules(recon Core)
