set(COMMON_SRC_FILES
    ConvKernel.cpp
    basicImageData.cpp
    ImageData.cpp
    basicReconData.cpp
    ReconData.cpp
    common.h
)

add_library(common STATIC ${COMMON_SRC_FILES})
target_link_libraries(common gsl blas)
qt5_use_modules(common Core)
