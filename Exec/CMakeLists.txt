#[[
 * @file Exec/CMakeLists.txt
 * @brief Modern CMake configuration for the CuSAXS execution library.
 *
 * This library provides the main execution functionality including:
 * - RunSaxs main execution class
 * - Command-line options parsing
 * - Integration with Python analysis scripts
]]

# Sources for the exec library
set(EXEC_SOURCES
    RunSaxs.cu
    Options.cpp
)

set(EXEC_HEADERS
    RunSaxs.h
    Options.h
)

# Create the exec library
add_library(exec ${EXEC_SOURCES} ${EXEC_HEADERS})

# Create an alias for consistent naming
add_library(CuSAXS::exec ALIAS exec)

# Set target properties
set_target_properties(exec PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Target-specific include directories
target_include_directories(exec
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/Saxs
        ${PROJECT_SOURCE_DIR}/System
        ${PROJECT_SOURCE_DIR}/Utilities
)

# Link dependencies
target_link_libraries(exec
    PUBLIC
        CuSAXS::saxs
        CuSAXS::system
        CuSAXS::utils
        Python3::Python
        pybind11::module
    PRIVATE
        fmt::fmt
        CUDA::cudart
)

# Target-specific compile definitions
target_compile_definitions(exec 
    PRIVATE 
        PY_SOURCE_DIR="${PROJECT_SOURCE_DIR}/pysrc"
)

# CUDA-specific compile options
target_compile_options(exec PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# Compiler-specific options
target_compile_features(exec PUBLIC cxx_std_17 cuda_std_17)

# Install targets
install(TARGETS exec
    EXPORT CuSAXSTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(FILES ${EXEC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CuSAXS/exec
    COMPONENT Development
)