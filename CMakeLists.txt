#[[
 * @file CMakeLists.txt
 * @brief Modern CMake configuration file for the CuSAXS project.
 *
 * This CMake file uses modern CMake practices including:
 * - Target-based approach with PRIVATE/PUBLIC/INTERFACE keywords
 * - Proper dependency management with imported targets
 * - Feature-based configuration
 * - Comprehensive error checking
 * - Version and compatibility requirements
]]

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project declaration with version
project(CuSAXS  
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated Small-Angle X-ray Scattering calculation"
    LANGUAGES CXX C CUDA
)


# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Options
option(BUILD_GUI "Build the Qt GUI for CuSAXS" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_TESTING "Enable testing" ON)

# Include useful CMake modules
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
include(CMakePackageConfigHelpers)

# Include our custom dependency finder
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDependencies.cmake)

# Find and configure all dependencies
configure_all_dependencies()

# CUDA architectures - set explicitly to avoid auto-detection issues
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures to build for" FORCE)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
endif()

# CUDA compiler flags - Enhanced optimization flags for better performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math --maxrregcount=32 -Xptxas -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Suppress GCC extension and other warnings from NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -diag-suppress 20012,2464,2417,128")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler  ")

# Add fmt subdirectory only if not using system version
option(USE_SYSTEM_FMT "Use system-installed fmt library instead of bundled version" OFF)
if(NOT USE_SYSTEM_FMT)
    # Disable fmt tests to avoid target conflicts
    set(FMT_TEST OFF CACHE BOOL "Disable fmt tests")
    set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation")
    set(FMT_INSTALL ON CACHE BOOL "Enable fmt installation")
    add_subdirectory(fmt)
endif()
add_subdirectory(Utilities)
add_subdirectory(System)  
add_subdirectory(Saxs)
add_subdirectory(Exec)
add_subdirectory(docs)

if(BUILD_GUI)
    find_package(Qt6 QUIET COMPONENTS Core Widgets)
    if(Qt6_FOUND)
        message(STATUS "Found Qt6, building GUI")
        add_subdirectory(gui)
    else()
        find_package(Qt5 5.15 COMPONENTS Core Widgets)
        if(Qt5_FOUND)
            message(STATUS "Found Qt5, building GUI")
            add_subdirectory(gui)
        else()
            message(WARNING "Qt not found, disabling GUI build")
            set(BUILD_GUI OFF)
        endif()
    endif()
endif()

# Main executable
add_executable(CuSAXS CuSAXS.cu)

# Set target properties
set_target_properties(CuSAXS PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Target-based include directories
target_include_directories(CuSAXS 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/Exec
        ${CMAKE_CURRENT_SOURCE_DIR}/System
)

# Target-based linking with proper visibility
target_link_libraries(CuSAXS 
    PRIVATE 
        CuSAXS::exec
        CuSAXS::saxs
        CuSAXS::system
        CuSAXS::utils
        CUDA::cufft
        CUDA::cublas
        CUDA::cudart
        CUDA::nvtx3
        fmt::fmt
        Python3::Python
        pybind11::module
        OpenMP::OpenMP_CXX
)

# Target-specific compile definitions
target_compile_definitions(CuSAXS 
    PRIVATE 
        PY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/pysrc"
)

# Installation
install(TARGETS CuSAXS
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)

# Install Python scripts
install(DIRECTORY pysrc/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/CuSAXS/python
    FILES_MATCHING PATTERN "*.py"
)

# Package configuration
if(BUILD_GUI AND TARGET CuSAXS_GUI)
    install(TARGETS CuSAXS_GUI
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Runtime
    )
endif()

# Package configuration files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CuSAXSConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CuSAXSConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuSAXS
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CuSAXSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CuSAXSConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CuSAXSConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuSAXS
    COMPONENT Development
)

# Export targets
install(EXPORT CuSAXSTargets
    FILE CuSAXSTargets.cmake
    NAMESPACE CuSAXS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CuSAXS
    COMPONENT Development
)

# Export from build tree
export(EXPORT CuSAXSTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/CuSAXSTargets.cmake"
    NAMESPACE CuSAXS::
)

# Register package in user's package registry
export(PACKAGE CuSAXS)

# CPack configuration for packaging
set(CPACK_PACKAGE_NAME "CuSAXS")
set(CPACK_PACKAGE_VENDOR "CuSAXS Project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU-accelerated Small-Angle X-ray Scattering calculation")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
else()
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
endif()

include(CPack)

# Development information
message(STATUS "")
message(STATUS "CuSAXS Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build GUI: ${BUILD_GUI}")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")