set(SRC_FILES
    ImageData.cpp
)

file(GLOB HDR_FILES *.h)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(imageData STATIC ${SRC_FILES})

install(FILES ${HDR_FILES} DESTINATION include)

