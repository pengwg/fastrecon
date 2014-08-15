set(SRC_FILES
    ReconData.cpp
# Include the following to show them in QtCreator project
    cuReconData.h
)

add_library(reconData STATIC ${SRC_FILES})
qt5_use_modules(reconData Core)
