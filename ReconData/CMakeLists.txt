set(RECON_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ReconData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GridLut.cpp   
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvKernel.cpp 
    CACHE INTERNAL ""  
)

set(CU_RECON_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cuReconData.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuGridLut.cu   
    ${CMAKE_CURRENT_SOURCE_DIR}/cuReconData.h   # Include the header files to show them in QtCreator project
    ${CMAKE_CURRENT_SOURCE_DIR}/cuGridLut.h
    CACHE INTERNAL ""
)

install(FILES ReconData.h GridLut.h DESTINATION include)
