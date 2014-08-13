set(SRC_FILES
    basicReconData.cpp
    ReconData.cpp
# Include the following to show them in QtCreator project
    hostReconData.h
    hostReconData.inl
    cuReconData.h
    cuReconData.inl
)

add_library(reconData STATIC ${SRC_FILES})
qt5_use_modules(reconData Core)
