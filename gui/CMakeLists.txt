cmake_minimum_required(VERSION 3.16)
project(CuSAXS_GUI)

# Find Qt6 or Qt5
find_package(Qt6 COMPONENTS Core Widgets QUIET)
if (NOT Qt6_FOUND)
    find_package(Qt5 5.15 REQUIRED COMPONENTS Core Widgets)
    set(QT_VERSION_MAJOR 5)
else()
    set(QT_VERSION_MAJOR 6)
endif()

# Set up Qt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Tell AUTOUIC where to find .ui files
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/ui)

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # For UI generated headers

# UI files
set(UI_FILES
    ui/MainWindow.ui
    ui/InputForm.ui
    ui/AdvancedOptionsDialog.ui
)

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/InputForm.cpp
    src/AdvancedOptionsDialog.cpp
    src/ProcessRunner.cpp
    src/OptionsData.cpp
    src/InputValidator.cpp
)

# Header files
set(HEADERS
    include/MainWindow.h
    include/InputForm.h
    include/AdvancedOptionsDialog.h
    include/ProcessRunner.h
    include/OptionsData.h
    include/InputValidator.h
)

# Create executable
add_executable(CuSAXS_GUI ${SOURCES} ${HEADERS} ${UI_FILES})

# Link Qt libraries
if(QT_VERSION_MAJOR EQUAL 6)
    target_link_libraries(CuSAXS_GUI Qt6::Core Qt6::Widgets)
else()
    target_link_libraries(CuSAXS_GUI Qt5::Core Qt5::Widgets)
endif()

# Installation
install(TARGETS CuSAXS_GUI DESTINATION bin)

# On Windows, deploy Qt libraries
if(WIN32)
    if(QT_VERSION_MAJOR EQUAL 6)
        add_custom_command(TARGET CuSAXS_GUI POST_BUILD
            COMMAND ${CMAKE_PREFIX_PATH}/bin/windeployqt.exe $<TARGET_FILE:CuSAXS_GUI>
        )
    else()
        add_custom_command(TARGET CuSAXS_GUI POST_BUILD
            COMMAND ${CMAKE_PREFIX_PATH}/bin/windeployqt.exe $<TARGET_FILE:CuSAXS_GUI>
        )
    endif()
endif()