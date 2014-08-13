set(SRC_FILES
    basicImageData.cpp
    ImageData.cpp
# Include the following to show them in QtCreator project
    common.h
    hostImageData.h
    hostImageData.inl
    cuImageData.h
    cuImageData.inl
)

add_library(imageData STATIC ${SRC_FILES})

