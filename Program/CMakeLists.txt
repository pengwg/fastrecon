set(MAIN_SRC_FILES
    Main.cpp
    ProgramOptions.cpp
)

add_executable(fastrecon ${MAIN_SRC_FILES})
target_link_libraries(fastrecon Recon)
qt5_use_modules(fastrecon Core Widgets)

