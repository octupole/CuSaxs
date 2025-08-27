#ifndef CUDAERRORS_H
#define CUDAERRORS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "MyErrors.h"

#pragma once

/**
 * @brief CUDA error checking utilities
 * 
 * This header provides macros and functions for comprehensive CUDA error checking
 * throughout the cudaSAXS application. It includes error checking for CUDA runtime,
 * CUFFT operations, and device synchronization.
 */

/**
 * @brief Macro for checking CUDA runtime errors
 * 
 * This macro checks the return value of CUDA runtime API calls and throws
 * an exception if an error is detected.
 * 
 * @param call The CUDA runtime API call to check
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw MyErrors("CUDA error at " + std::string(__FILE__) + ":" + \
                          std::to_string(__LINE__) + " - " + \
                          cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * @brief Macro for checking CUDA kernel launches
 * 
 * This macro should be called after kernel launches to check for errors.
 * It checks both the launch error and synchronizes the device to catch
 * any runtime errors in the kernel.
 */
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            throw MyErrors("CUDA kernel launch error at " + std::string(__FILE__) + ":" + \
                          std::to_string(__LINE__) + " - " + \
                          cudaGetErrorString(error)); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

/**
 * @brief Macro for checking CUFFT errors
 * 
 * This macro checks the return value of CUFFT API calls and throws
 * an exception if an error is detected.
 * 
 * @param call The CUFFT API call to check
 */
#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            throw MyErrors("CUFFT error at " + std::string(__FILE__) + ":" + \
                          std::to_string(__LINE__) + " - " + \
                          cufftErrorToString(error)); \
        } \
    } while(0)

/**
 * @brief Convert CUFFT error code to string
 * 
 * @param error CUFFT error code
 * @return String description of the error
 */
inline const char* cufftErrorToString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
        default:
            return "Unknown CUFFT error";
    }
}

/**
 * @brief Check CUDA device properties and capabilities
 * 
 * This function checks if the current CUDA device supports the required
 * features for the SAXS calculations.
 * 
 * @param device_id CUDA device ID to check (default: 0)
 * @throws MyErrors if device doesn't meet requirements
 */
inline void checkCudaDevice(int device_id = 0) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        throw MyErrors("No CUDA devices found");
    }
    
    if (device_id >= device_count) {
        throw MyErrors("Invalid CUDA device ID: " + std::to_string(device_id) + 
                      " (available devices: " + std::to_string(device_count) + ")");
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Check compute capability (require at least 3.0 for double precision)
    if (prop.major < 3) {
        throw MyErrors("CUDA device compute capability too low: " + 
                      std::to_string(prop.major) + "." + std::to_string(prop.minor) + 
                      " (minimum required: 3.0)");
    }
    
    // Check available memory (warn if less than 1GB)
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    const size_t min_memory = 1024 * 1024 * 1024; // 1GB
    if (free_mem < min_memory) {
        std::cerr << "Warning: Low GPU memory available: " 
                  << free_mem / (1024 * 1024) << " MB" << std::endl;
    }
    
    std::cout << "Using CUDA device " << device_id << ": " << prop.name 
              << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
}

/**
 * @brief Validate array dimensions for CUDA operations
 * 
 * @param nx X dimension
 * @param ny Y dimension  
 * @param nz Z dimension
 * @param name Array name for error messages
 * @throws MyErrors if dimensions are invalid
 */
inline void validateArrayDimensions(int nx, int ny, int nz, const std::string& name) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw MyErrors("Invalid " + name + " dimensions: " + 
                      std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + 
                      " (all dimensions must be positive)");
    }
    
    // Check for reasonable maximum sizes (prevent excessive memory allocation)
    const int max_dim = 4096;
    if (nx > max_dim || ny > max_dim || nz > max_dim) {
        throw MyErrors("Grid dimension too large for " + name + ": " + 
                      std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + 
                      " (maximum per dimension: " + std::to_string(max_dim) + ")");
    }
}

#endif