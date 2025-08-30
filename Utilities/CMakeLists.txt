#[[
 * @file Utilities/CMakeLists.txt
 * @brief Modern CMake configuration for the CuSAXS utilities library.
 *
 * This library provides utility functions and classes including:
 * - B-Spline operations
 * - Scattering form factors
 * - CUDA error handling utilities
]]

# Sources for the utilities library
set(UTILS_SOURCES
    BSpline.cpp
    BSpmod.cpp
    Scattering.cpp
)

set(UTILS_HEADERS
    BSpline.h
    BSpmod.h
    Scattering.h
    MyErrors.h
    CudaErrors.h
)

# Create the utilities library
add_library(utils ${UTILS_SOURCES} ${UTILS_HEADERS})

# Create an alias for consistent naming
add_library(CuSAXS::utils ALIAS utils)

# Set target properties
set_target_properties(utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Target-specific include directories
target_include_directories(utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(utils
    PUBLIC
        CUDA::cudart
        CUDA::cufft
    PRIVATE
        fmt::fmt
)

# Compiler-specific options
target_compile_features(utils PUBLIC cxx_std_17)

# Install targets
install(TARGETS utils
    EXPORT CuSAXSTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${UTILS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CuSAXS/utilities
    COMPONENT Development
)