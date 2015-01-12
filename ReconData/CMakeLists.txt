set(SRC_FILES
    ReconData.cpp
# Include the following to show them in QtCreator project
    cuReconData.h
)

file(GLOB HDR_FILES *.h)

add_library(reconData STATIC ${SRC_FILES})
qt5_use_modules(reconData Core)

install(FILES ${HDR_FILES} DESTINATION include)
