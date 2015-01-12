set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(FFTW3 COMPONENTS single REQUIRED)

set(SRC_FILES
    GridLut.cpp
    FFT.cpp
    ImageRecon.cpp
    ConvKernel.cpp

    cuGridLut.h
)

file(GLOB HDR_FILES *.h)

include_directories(
    ${FFTW3_INCLUDE_DIR}
)

add_library(Recon SHARED ${SRC_FILES})

target_link_libraries(Recon ${FFTW3_LIBRARIES} gsl blas imageData reconData)
qt5_use_modules(Recon Core)

install(TARGETS Recon LIBRARY DESTINATION lib)
install(FILES ${HDR_FILES} DESTINATION include)
