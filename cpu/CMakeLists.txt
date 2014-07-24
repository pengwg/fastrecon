set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(FFTW3 COMPONENTS single REQUIRED)

set(CPU_SRC_FILES
    GridLut.cpp
    FFT.cpp
    ImageRecon.cpp
)

include_directories(
    ${FFTW3_INCLUDE_DIR}
)

add_library(recon SHARED ${CPU_SRC_FILES})

target_link_libraries(recon ${FFTW3_LIBRARIES})
qt5_use_modules(recon Core)
