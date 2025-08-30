/**
 * @class saxsKernel
 * @brief Provides functionality for performing small-angle X-ray scattering (SAXS) calculations.
 *
 * The `saxsKernel` class is responsible for managing the memory and computation required for SAXS
 * calculations. It provides methods for setting the number of particles and grid dimensions, as well
 * as running the main SAXS kernel. The class also manages the allocation and deallocation of
 * various device memory buffers used in the SAXS computations.
 */
#ifndef SAXSKERNEL_H
#define SAXSKERNEL_H
#include "Splines.h"
#include "Options.h"
#include <vector>
#include <cufft.h>
#include <cuComplex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <map>
#include "Ftypedefs.h"
#include <fmt/core.h>

#pragma once

class saxsKernel
{
public:
    saxsKernel(int _nx, int _ny, int _nz, int _order) : nx{_nx}, ny{_ny}, nz{_nx}, order(_order) {};
    void setnpx(int _npx, int _npy, int _npz)
    {
        npx = _npx;
        npy = _npy;
        npz = _npz;
    }
    void setnpx(int _npx)
    {
        npx = _npx;
        npy = _npx;
        npz = _npx;
    }
    void runPKernel(int, float, std::vector<std::vector<float>> &, std::map<std::string, std::vector<int>> &, std::vector<std::vector<float>> &);
    double getCudaTime() { return cudaTime / cudaCalls; }
    void scaledCell();
    void zeroIq();
    void getHistogram(std::vector<std::vector<float>> &);
    std::vector<std::vector<double>> getSaxs();
    void createMemory();
    void resetHistogramParameters(std::vector<std::vector<float>> &);
    void writeBanner();
    void setupPinnedMemory();
    void optimizeKernelLaunchParams();
    void setupUnifiedMemory();
    void cleanupUnifiedMemory();
    void initializeMemoryPools();
    void cleanupMemoryPools();
    float* getPooledMemory(size_t size);
    void returnPooledMemory(float* ptr, size_t size);
    void initializeCudaStreams();
    void cleanupCudaStreams();
    void setcufftPlan(int nnx, int nny, int nnz)
    {
        // Create optimized FFT plan with better performance
        cufftPlan3d(&cufftPlan, nnx, nny, nnz, CUFFT_R2C);
        
        // Set auto-allocation to off for better memory management
        cufftSetAutoAllocation(cufftPlan, 0);
        
        // Allocate work area manually for better control
        size_t workSize;
        cufftGetSize(cufftPlan, &workSize);
        cudaMalloc(&fftWorkArea, workSize);
        cufftSetWorkArea(cufftPlan, fftWorkArea);
    }
    cufftHandle &getPlan() { return cufftPlan; }

    ~saxsKernel();

private:
    int size;
    int order;
    int npx, npy, npz;
    
    // Optimized kernel launch parameters
    int optimal_block_size_1d{256};
    dim3 optimal_block_size_3d{8, 8, 8};
    int max_occupancy{0};
    int nx, ny, nz, nnx, nny, nnz;
    int numParticles;
    float sigma;
    float bin_size;
    float kcut;
    float dk;
    int num_bins;
    double cudaTime{0};
    double cudaCalls{0};
    static int frame_count;
    cufftHandle cufftPlan;
    void *fftWorkArea{nullptr};

    thrust::device_vector<float> d_moduleX;
    thrust::device_vector<float> d_moduleY;
    thrust::device_vector<float> d_moduleZ;
    thrust::device_vector<float> d_grid;
    thrust::device_vector<float> d_gridSup;
    thrust::device_vector<cuFloatComplex> d_gridSupAcc;
    thrust::device_vector<cuFloatComplex> d_Iq;
    thrust::device_vector<cuFloatComplex> d_gridSupC;
    thrust::device_vector<double> d_histogram;
    thrust::device_vector<double> d_nhist;

    thrust::host_vector<float> h_moduleX;
    thrust::host_vector<float> h_moduleY;
    thrust::host_vector<float> h_moduleZ;
    thrust::host_vector<double> h_histogram;
    thrust::host_vector<double> h_nhist;
    
    // Pinned memory pointers for faster host-device transfers
    float *h_particles_pinned{nullptr};
    float *h_scatter_pinned{nullptr};
    size_t h_particles_size{0};
    size_t h_scatter_size{0};
    
    // Unified memory pointers for seamless CPU-GPU access
    float *unified_particles{nullptr};
    float *unified_scatter{nullptr};
    double *unified_histogram{nullptr};
    double *unified_nhist{nullptr};
    size_t unified_particles_size{0};
    size_t unified_histogram_size{0};
    
    // Memory pools for efficient allocation/deallocation
    struct MemoryPool {
        std::vector<float*> available_blocks;
        std::vector<float*> allocated_blocks;
        size_t block_size;
        size_t total_blocks;
        size_t allocated_count;
    };
    
    std::map<size_t, MemoryPool> memory_pools;
    std::vector<size_t> common_sizes{1024, 4096, 16384, 65536, 262144, 1048576}; // Common allocation sizes
    
    // CUDA streams for overlapping computation and memory transfers
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    cudaStream_t fft_stream;
    cudaEvent_t compute_done;
    cudaEvent_t transfer_done;
    cudaEvent_t fft_done;

    float *d_grid_ptr{nullptr};
    float *d_gridSup_ptr{nullptr};
    cuFloatComplex *d_gridSupC_ptr{nullptr};
    cuFloatComplex *d_gridSupAcc_ptr{nullptr};
    cuFloatComplex *d_Iq_ptr{nullptr};
    // Do bspmod
    float *d_moduleX_ptr{nullptr};
    float *d_moduleY_ptr{nullptr};
    float *d_moduleZ_ptr{nullptr};
    double *d_histogram_ptr{nullptr};
    double *d_nhist_ptr{nullptr};
    std::function<int(int, float)> borderBins = [](int nx, float shell) -> int
    {
        return static_cast<int>(shell * nx / 2);
    };

    std::vector<long long> generateMultiples(long long limit);
    long long findClosestProduct(int n, float sigma);
};

#endif