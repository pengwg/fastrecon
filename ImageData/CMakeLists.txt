set(IMAGE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ImageData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.h        # Include header files to show them in qtcreator
    CACHE INTERNAL ""
)

set(CU_IMAGE_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cuImageData.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuImageData.h   # Include header files to show them in qtcreator
    CACHE INTERNAL ""
)

install(FILES common.h ImageData.h DESTINATION include)

