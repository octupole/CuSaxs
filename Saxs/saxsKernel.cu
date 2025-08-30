#include "saxsKernel.h"
#include "BSpmod.h"
#include "Scattering.h"
#include "opsfact.h"
#include <cuda_runtime.h> // Include CUDA runtime header
#include <cuComplex.h>
#include <nvtx3/nvToolsExt.h>  // NVIDIA Nsight profiling

#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "Splines.h"
#include "Ftypedefs.h"
#include "opsfact.h"
#include "saxsDeviceKernels.cuh"
int saxsKernel::frame_count = 0;

void saxsKernel::getHistogram(std::vector<std::vector<float>> &oc)
{
    nvtxRangePush("saxsKernel::getHistogram");
    
    auto nnpz = nnz / 2 + 1;
    dim3 blockDim(npx, npy, npz);
    dim3 gridDim((nnx + blockDim.x - 1) / blockDim.x,
                 (nny + blockDim.y - 1) / blockDim.y,
                 (nnpz + blockDim.z - 1) / blockDim.z);

    float mySigma = (float)Options::nx / (float)Options::nnx;

    nvtxRangePush("Histogram Data Preparation");
    thrust::host_vector<float> h_oc(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
        {
            h_oc[i * DIM + j] = mySigma * oc[i][j];
        }
    thrust::device_vector<float> d_oc = h_oc;
    float *d_oc_ptr = thrust::raw_pointer_cast(d_oc.data());
    float frames_fact = 1.0 / (float)frame_count;
    std::cout << "frames_fact: " << bin_size << " " << kcut << " " << num_bins << std::endl;
    nvtxRangePop();
    
    nvtxRangePush("Histogram Calculation Kernel");
    calculate_histogram<<<gridDim, blockDim>>>(d_Iq_ptr, d_histogram_ptr, d_nhist_ptr, d_oc_ptr, nnx, nny, nnz,
                                               bin_size, kcut, num_bins, frames_fact);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePop();
}
void saxsKernel::zeroIq()
{
    nvtxRangePush("saxsKernel::zeroIq");
    // Use cudaMemset for better performance than custom kernel
    cudaMemset(d_Iq_ptr, 0, d_Iq.size() * sizeof(cuFloatComplex));
    nvtxRangePop();
}
// Kernel to calculate |K| values and populate the histogram

/**
 * Processes a set of particles and computes their contribution to the SAXS intensity.
 *
 * This function iterates over a set of particles, transforms their coordinates based on the orientation matrix,
 * and computes their contribution to the SAXS intensity. It then performs padding, supersampling, and Fourier
 * transform operations on the density grid to compute the final SAXS intensity.
 *
 * @param coords A vector of particle coordinates.
 * @param index_map A map of particle indices, where the keys are particle types and the values are vectors of indices.
 * @param oc The orientation matrix.
 */
void saxsKernel::runPKernel(int frame, float Time, std::vector<std::vector<float>> &coords, std::map<std::string, std::vector<int>> &index_map, std::vector<std::vector<float>> &oc)
{
    nvtxRangePush("saxsKernel::runPKernel");
    
    // Cudaevents
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int mx = borderBins(nx, SHELL);
    int my = borderBins(ny, SHELL);
    int mz = borderBins(nz, SHELL);
    float mySigma = (float)Options::nx / (float)Options::nnx;

    thrust::host_vector<float> h_oc(DIM * DIM);
    thrust::host_vector<float> h_oc_or(DIM * DIM);
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
        {
            h_oc[i * DIM + j] = mySigma * oc[i][j];
            h_oc_or[i * DIM + j] = oc[i][j];
        }
    thrust::device_vector<float> d_oc = h_oc;
    thrust::device_vector<float> d_oc_or = h_oc_or;
    float *d_oc_ptr = thrust::raw_pointer_cast(d_oc.data());
    float *d_oc_or_ptr = thrust::raw_pointer_cast(d_oc_or.data());
    auto nnpz = nnz / 2 + 1;

    // Use optimized block dimensions
    dim3 blockDim = optimal_block_size_3d;

    dim3 gridDim((nnx + blockDim.x - 1) / blockDim.x,
                 (nny + blockDim.y - 1) / blockDim.y,
                 (nnpz + blockDim.z - 1) / blockDim.z);
    dim3 gridDimR((nnx + blockDim.x - 1) / blockDim.x,
                  (nny + blockDim.y - 1) / blockDim.y,
                  (nnz + blockDim.z - 1) / blockDim.z);
    dim3 gridDim0((nx + blockDim.x - 1) / blockDim.x,
                  (ny + blockDim.y - 1) / blockDim.y,
                  (nz + blockDim.z - 1) / blockDim.z);
    // Use optimized 1D block size
    const int THREADS_PER_BLOCK = optimal_block_size_1d;
    int numBlocksGrid = (d_grid.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuperC = (d_gridSupC.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuperAcc = (d_gridSupAcc.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridSuper = (d_gridSup.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBlocksGridIq = (d_Iq.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // zeroes the Sup density grid - using cudaMemset for better performance
    nvtxRangePush("Grid Initialization");
    cudaMemset(d_gridSupAcc_ptr, 0, d_gridSupAcc.size() * sizeof(cuFloatComplex));
    nvtxRangePop();

    float totParticles{0};
    std::string formatted_string = fmt::format("--> Frame: {:<7}  Time Step: {:.2f} fs", frame, Time);

    // Print the formatted string
    std::cout << formatted_string << std::endl;
    float N1 = (float)nnx / (float)nx;
    float N2 = (float)nny / (float)ny;
    float N3 = (float)nnz / (float)nz;
    auto plan = this->getPlan();
    for (const auto &pair : index_map)
    {
        nvtxRangePush(("Processing particle type: " + pair.first).c_str());
        
        std::string type = pair.first;
        std::vector<int> value = pair.second;
        
        nvtxRangePush("Particle Data Preparation");
        // Create a host vector to hold the particles
        thrust::host_vector<float> h_particles;
        h_particles.reserve(value.size() * 3);
        // Fill the host vector with the particles according to the indices
        std::for_each(value.begin(), value.end(), [&h_particles, &coords](int i)
                      { h_particles.insert(h_particles.end(), coords[i].begin(), coords[i].end()); });
        // define the number of particles
        this->numParticles = value.size();

        // Allocate and copy particles to the device
        thrust::device_vector<float> d_particles = h_particles;
        // Copy the host vector to the device
        thrust::host_vector<float> h_scatter = Scattering::getScattering(type);
        thrust::device_vector<float> d_scatter = h_scatter;
        nvtxRangePop();

        float *d_particles_ptr = thrust::raw_pointer_cast(d_particles.data());
        float *d_scatter_ptr = thrust::raw_pointer_cast(d_scatter.data());

        int numBlocks = (numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        //    Initialize grid with cudaMemset for better performance
        cudaMemset(d_grid_ptr, 0, d_grid.size() * sizeof(float));
        // Check for errors
        nvtxRangePush("Density Grid Calculation");
        rhoCartKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_particles_ptr, d_oc_or_ptr, d_grid_ptr, order,
                                                        numParticles, nx, ny, nz);
        // Synchronize the device
        cudaDeviceSynchronize();
        nvtxRangePop();
        // picking the padding
        float myDens = 0.0f;
        if (Options::myPadding == padding::avg)
        {
            thrust::host_vector<float> h_Dens = {0.0f};
            thrust::host_vector<int> h_count = {0};
            thrust::device_vector<float> d_Dens = h_Dens;
            thrust::device_vector<int> d_count = h_count;
            paddingKernel<<<gridDim0, blockDim>>>(d_grid_ptr, nx, ny, nz, mx, my, mz,
                                                  thrust::raw_pointer_cast(d_Dens.data()),
                                                  thrust::raw_pointer_cast(d_count.data()));
            // Synchronize the device
            cudaDeviceSynchronize();
            h_Dens = d_Dens;
            h_count = d_count;
            myDens = h_Dens[0] / (float)h_count[0];
        }
        else
        {
            if (Options::myWmodel.find(type) != Options::myWmodel.end())
            {
                myDens = Options::myWmodel[type];
            }
        }

        // zeroes the Sup density grid - using cudaMemset for better performance
        cudaMemset(d_gridSupC_ptr, 0, d_gridSupC.size() * sizeof(cuFloatComplex));

        nvtxRangePush("Super Density Kernel");
        superDensityKernel<<<gridDimR, blockDim>>>(d_grid_ptr, d_gridSup_ptr, myDens, nx, ny, nz, nnx, nny, nnz);
        // Synchronize the device
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePush("FFT Transform");
        cufftExecR2C(plan, d_gridSup_ptr, d_gridSupC_ptr);
        // Synchronize the device
        cudaDeviceSynchronize();
        nvtxRangePop();

        thrust::host_vector<float> h_nato = {0.0f};
        thrust::device_vector<float> d_nato = h_nato;
        nvtxRangePush("Scatter Kernel");
        scatterKernel<<<gridDim, blockDim>>>(d_gridSupC_ptr, d_gridSupAcc_ptr, d_oc_ptr, d_scatter_ptr, nnx, nny, nnz, kcut, thrust::raw_pointer_cast(d_nato.data()));
        cudaDeviceSynchronize();
        nvtxRangePop();
        
        h_nato = d_nato;
        totParticles += h_nato[0];
        
        nvtxRangePop(); // End particle type processing
    }
    nvtxRangePush("Modulus Calculation");
    modulusKernel<<<gridDim, blockDim>>>(d_gridSupAcc_ptr, d_moduleX_ptr, d_moduleY_ptr, d_moduleZ_ptr, nnx, nny, nnz);
    // // Synchronize the device
    cudaDeviceSynchronize();
    nvtxRangePop();
    if (Options::Simulation == "nvt")
    {
        nvtxRangePush("Grid Addition (NVT)");
        gridAddKernel<<<numBlocksGridIq, THREADS_PER_BLOCK>>>(d_gridSupAcc_ptr, d_Iq_ptr, d_Iq.size());
        cudaDeviceSynchronize();
        frame_count++;
        nvtxRangePop();
    }
    else if (Options::Simulation == "npt")
    {
        nvtxRangePush("Histogram Calculation (NPT)");
        calculate_histogram<<<gridDim, blockDim>>>(d_gridSupAcc_ptr, d_histogram_ptr, d_nhist_ptr, d_oc_ptr, nnx, nny, nnz,
                                                   bin_size, kcut, num_bins);
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
    cudaTime += gpuElapsedTime;
    cudaCalls += 1.0;

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    nvtxRangePop(); // End runPKernel
}
std::vector<std::vector<double>> saxsKernel::getSaxs()
{
    std::vector<std::vector<double>> saxs;
    h_histogram = d_histogram;
    h_nhist = d_nhist;

    for (size_t o{1}; o < h_histogram.size(); o++)
    {
        if (h_nhist[o] != 0.0f)
        {
            vector<double> val = {o * this->bin_size, h_histogram[o] / h_nhist[o]};
            saxs.push_back(val);
        }
    }
    return saxs;
}

/**
 * @brief Creates the necessary memory for the SAXS computation.
 *
 * This function sets up the memory buffers and allocates memory for the SAXS computation.
 * It calculates the optimal grid sizes (nnx, nny, nnz) based on the original grid sizes (nx, ny, nz)
 * and the given sigma value. It then creates the necessary host and device memory buffers for the
 * grid, super-grid, and module data.
 *
 * @param[in,out] nnx The optimal x-dimension of the super-grid.
 * @param[in,out] nny The optimal y-dimension of the super-grid.
 * @param[in,out] nnz The optimal z-dimension of the super-grid.
 * @param[in] sigma The sigma value used to calculate the optimal grid sizes.
 */
void saxsKernel::createMemory()
{
    size_t nnpz = nnz / 2 + 1;

    this->bin_size = Options::Dq;
    this->kcut = Options::Qcut;

    this->num_bins = static_cast<int>(kcut / bin_size) + 1;
    h_histogram = thrust::host_vector<float>(num_bins, 0.0f);
    h_nhist = thrust::host_vector<long int>(num_bins, 0);

    d_histogram = h_histogram;
    d_nhist = h_nhist;
    d_histogram_ptr = thrust::raw_pointer_cast(d_histogram.data());
    d_nhist_ptr = thrust::raw_pointer_cast(d_nhist.data());
    BSpline::BSpmod *bsp_modx = new BSpline::BSpmod(nnx, nny, nnz);

    thrust::host_vector<float> h_moduleX = bsp_modx->ModX();
    thrust::host_vector<float> h_moduleY = bsp_modx->ModY();
    thrust::host_vector<float> h_moduleZ = bsp_modx->ModZ();

    d_moduleX = h_moduleX;
    d_moduleY = h_moduleY;
    d_moduleZ = h_moduleZ;
    d_moduleX_ptr = thrust::raw_pointer_cast(d_moduleX.data());
    d_moduleY_ptr = thrust::raw_pointer_cast(d_moduleY.data());
    d_moduleZ_ptr = thrust::raw_pointer_cast(d_moduleZ.data());

    d_grid.resize(nx * ny * nz);
    d_gridSup.resize(nnx * nny * nnz);
    d_gridSupC.resize(nnx * nny * nnpz);
    d_gridSupAcc.resize(nnx * nny * nnpz);
    d_Iq.resize(nnx * nny * nnpz);

    d_grid_ptr = thrust::raw_pointer_cast(d_grid.data());
    d_gridSup_ptr = thrust::raw_pointer_cast(d_gridSup.data());
    d_gridSupC_ptr = thrust::raw_pointer_cast(d_gridSupC.data());
    d_gridSupAcc_ptr = thrust::raw_pointer_cast(d_gridSupAcc.data());
    d_Iq_ptr = thrust::raw_pointer_cast(d_Iq.data());
    // Do bspmod
}
/**
 * Generates a vector of multiples of 2, 3, 5, and 7 up to a given limit.
 *
 * This function generates all possible multiples of 2, 3, 5, and 7 up to the
 * specified limit, and returns them as a sorted, unique vector.
 *
 * @param limit The maximum value to generate multiples up to.
 * @return A vector of all multiples of 2, 3, 5, and 7 up to the given limit.
 */
// Function to generate multiples of 2, 3, 5, and 7 up to a given limit
std::vector<long long> saxsKernel::generateMultiples(long long limit)
{
    std::vector<long long> multiples;
    
    // Pre-compute powers to avoid repeated std::pow calls
    std::vector<long long> pow2, pow3, pow5, pow7;
    for (long long p = 1; p <= limit; p *= 2) pow2.push_back(p);
    for (long long p = 1; p <= limit; p *= 3) pow3.push_back(p);
    for (long long p = 1; p <= limit; p *= 5) pow5.push_back(p);
    for (long long p = 1; p <= limit; p *= 7) pow7.push_back(p);
    
    for (const auto& p2 : pow2)
    {
        if (p2 > limit) break;
        for (const auto& p3 : pow3)
        {
            long long p23 = p2 * p3;
            if (p23 > limit) break;
            for (const auto& p5 : pow5)
            {
                long long p235 = p23 * p5;
                if (p235 > limit) break;
                for (const auto& p7 : pow7)
                {
                    long long multiple = p235 * p7;
                    if (multiple <= limit)
                    {
                        multiples.push_back(multiple);
                    }
                    else break;
                }
            }
        }
    }
    std::sort(multiples.begin(), multiples.end());
    multiples.erase(std::unique(multiples.begin(), multiples.end()), multiples.end());
    return multiples;
}

/**
 * Finds the closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7.
 *
 * This function takes a target value N and a standard deviation sigma, and finds the closest integer
 * to N * sigma that can be expressed as a product of only the prime factors 2, 3, 5, and 7.
 *
 * @param n The target value N.
 * @param sigma The standard deviation.
 * @return The closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7.
 */
// Function to find the closest integer to N * sigma that is obtainable by multiplying only 2, 3, 5, and 7
long long saxsKernel::findClosestProduct(int n, float sigma)
{
    long long target = std::round(n * sigma);
    long long limit = target * 2; // A generous limit for generating multiples
    std::vector<long long> multiples = generateMultiples(limit);

    long long closest = target;
    long long minDifference = std::numeric_limits<long long>::max();

    for (long long multiple : multiples)
    {
        long long difference = std::abs(multiple - target);
        if (difference < minDifference)
        {
            minDifference = difference;
            closest = multiple;
        }
    }

    return closest;
}
void saxsKernel::scaledCell()
{
    sigma = Options::sigma;
    if (Options::nnx == 0)
    {
        const int nnx_new = static_cast<int>(findClosestProduct(nx, sigma));
        const int nny_new = static_cast<int>(findClosestProduct(ny, sigma));
        const int nnz_new = static_cast<int>(findClosestProduct(nz, sigma));

        this->nnx = nnx_new;
        this->nny = nny_new;
        this->nnz = nnz_new;

        Options::nnx = nnx_new;
        Options::nny = nny_new;
        Options::nnz = nnz_new;
    }
    else
    {
        this->nnx = Options::nnx;
        this->nny = Options::nny;
        this->nnz = Options::nnz;
    }
}
void saxsKernel::resetHistogramParameters(std::vector<std::vector<float>> &oc)
{
    using namespace std;
    auto qcut = Options::Qcut;
    auto dq = Options::Dq;
    int nfx{(nnx % 2 == 0) ? nnx / 2 : nnx / 2 + 1};
    int nfy{(nny % 2 == 0) ? nny / 2 : nny / 2 + 1};
    int nfz{(nnz % 2 == 0) ? nnz / 2 : nnz / 2 + 1};
    float argx{2.0f * (float)M_PI * oc[XX][XX] / sigma};
    float argy{2.0f * (float)M_PI * oc[YY][YY] / sigma};
    float argz{2.0f * (float)M_PI * oc[ZZ][ZZ] / sigma};

    std::vector<float> fx{(float)nfx - 1, (float)nfy - 1, (float)nfz - 1};

    vector<float> mydq0 = {argx, argy, argz, dq};
    vector<float> mycut0 = {argx * fx[XX], argy * fx[YY], argz * fx[ZZ], qcut};

    dq = (*std::max_element(mydq0.begin(), mydq0.end()));
    qcut = *std::min_element(mycut0.begin(), mycut0.end());
    if (qcut != Options::Qcut)
    {
        std::string formatted_string = fmt::format("----- Qcut had to be reset to {:.2f} from  {:.2f} ----", qcut, Options::Qcut);
        std::cout << "\n--------------------------------------------------\n";
        std::cout << formatted_string << "\n";
        std::cout << "--------------------------------------------------\n\n";

        Options::Qcut = qcut;
    }
    if (dq != Options::Dq)
    {
        std::string formatted_string = fmt::format("----- Dq had to be reset to {:.3f} from  {:.3f} ----", dq, Options::Dq);
        std::cout << "\n--------------------------------------------------\n";
        std::cout << formatted_string << "\n";
        std::cout << "--------------------------------------------------\n\n";

        Options::Dq = dq;
    }
}
void saxsKernel::writeBanner()
{
    std::string padding = Options::myPadding == padding::given ? Options::Wmodel : "avg Border";
    std::string banner{""};
    if (Options::myPadding == padding::avg)
    {
        banner = fmt::format(
            "*************************************************\n"
            "* {:^40}      *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<10} {:>4}      {:<10} {:>4}          *\n"
            "* {:<10} {:>4.3f}     {:<10}  {:>3.1f}          *\n"
            "* {:<10}           {:<14}           *\n"
            "*************************************************\n\n",
            "Running CuSAXS", "Cell Grid", Options::nx, Options::ny, Options::nz,
            "Supercell Grid", Options::nnx, Options::nny, Options::nnz, "Order",
            Options::order, "Sigma", Options::sigma, "Bin Size", Options::Dq, "Q Cutoff ", Options::Qcut, "Padding ", padding);
    }
    else
    {
        banner = fmt::format(
            "*************************************************\n"
            "* {:^40}      *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<19} {:>4} * {:>4} * {:>4}        *\n"
            "* {:<10} {:>4}      {:<10} {:>4}          *\n"
            "* {:<10} {:>4.3f}     {:<10}  {:>3.1f}          *\n"
            "* {:<10}           {:<14}           *\n"
            "* {:<10} {:>4d}      {:<10} {:>4d}          *\n"
            "*************************************************\n\n",
            "Running CuSAXS", "Cell Grid", Options::nx, Options::ny, Options::nz,
            "Supercell Grid", Options::nnx, Options::nny, Options::nnz, "Order",
            Options::order, "Sigma", Options::sigma, "Bin Size", Options::Dq, "Q Cutoff ",
            Options::Qcut, "Padding ", padding,
            "Na ions", Options::Sodium, "Cl Ions", Options::Chlorine);
    }
    std::cout << banner;
}

void saxsKernel::setupPinnedMemory()
{
    // Allocate pinned memory for frequently transferred data
    // Typical maximum particles per type: 100,000
    h_particles_size = 100000 * 3 * sizeof(float);  // x,y,z coordinates
    h_scatter_size = 9 * sizeof(float);  // 9 scattering factors
    
    cudaHostAlloc(&h_particles_pinned, h_particles_size, cudaHostAllocDefault);
    cudaHostAlloc(&h_scatter_pinned, h_scatter_size, cudaHostAllocDefault);
}

void saxsKernel::setupUnifiedMemory()
{
    // Allocate unified memory for data structures accessed by both CPU and GPU
    // This allows seamless data access without explicit transfers
    
    // Allocate for particle coordinates (larger for multiple atom types)
    unified_particles_size = 500000 * 3 * sizeof(float);  // x,y,z for 500k particles
    cudaMallocManaged(&unified_particles, unified_particles_size);
    
    // Allocate for scattering factors (multiple atom types)
    cudaMallocManaged(&unified_scatter, 32 * 9 * sizeof(float));  // 32 atom types, 9 factors each
    
    // Allocate for histogram data (frequently accessed by both CPU and GPU)
    unified_histogram_size = 10000 * sizeof(double);  // Large histogram bins
    cudaMallocManaged(&unified_histogram, unified_histogram_size);
    cudaMallocManaged(&unified_nhist, unified_histogram_size);
    
    // Set memory access hints for better performance
    int deviceId;
    cudaGetDevice(&deviceId);
    
    // Hint that particles data is mostly accessed by GPU
    cudaMemAdvise(unified_particles, unified_particles_size, cudaMemAdviseSetPreferredLocation, deviceId);
    cudaMemAdvise(unified_scatter, 32 * 9 * sizeof(float), cudaMemAdviseSetPreferredLocation, deviceId);
    
    // Hint that histogram is accessed by both CPU and GPU
    cudaMemAdvise(unified_histogram, unified_histogram_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemAdvise(unified_histogram, unified_histogram_size, cudaMemAdviseSetAccessedBy, deviceId);
    cudaMemAdvise(unified_nhist, unified_histogram_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    cudaMemAdvise(unified_nhist, unified_histogram_size, cudaMemAdviseSetAccessedBy, deviceId);
    
    std::cout << "Unified Memory initialized: " << 
        (unified_particles_size + 32*9*sizeof(float) + 2*unified_histogram_size) / (1024*1024) 
        << " MB allocated" << std::endl;
}

void saxsKernel::cleanupUnifiedMemory()
{
    if (unified_particles) {
        cudaFree(unified_particles);
        unified_particles = nullptr;
    }
    if (unified_scatter) {
        cudaFree(unified_scatter);
        unified_scatter = nullptr;
    }
    if (unified_histogram) {
        cudaFree(unified_histogram);
        unified_histogram = nullptr;
    }
    if (unified_nhist) {
        cudaFree(unified_nhist);
        unified_nhist = nullptr;
    }
}

void saxsKernel::initializeMemoryPools()
{
    // Initialize memory pools for common allocation sizes
    size_t total_pool_memory = 0;
    
    for (size_t size : common_sizes) {
        MemoryPool& pool = memory_pools[size];
        pool.block_size = size * sizeof(float);
        pool.total_blocks = 64;  // Start with 64 blocks per pool
        pool.allocated_count = 0;
        
        // Pre-allocate blocks
        for (size_t i = 0; i < pool.total_blocks; ++i) {
            float* block;
            cudaError_t err = cudaMalloc(&block, pool.block_size);
            if (err == cudaSuccess) {
                pool.available_blocks.push_back(block);
                total_pool_memory += pool.block_size;
            } else {
                std::cout << "Warning: Could not allocate memory pool block of size " 
                          << pool.block_size << " bytes" << std::endl;
                break;
            }
        }
    }
    
    std::cout << "Memory pools initialized: " << total_pool_memory / (1024*1024) 
              << " MB pre-allocated across " << memory_pools.size() << " pools" << std::endl;
}

void saxsKernel::cleanupMemoryPools()
{
    for (auto& [size, pool] : memory_pools) {
        // Free all available blocks
        for (float* block : pool.available_blocks) {
            cudaFree(block);
        }
        
        // Free any remaining allocated blocks
        for (float* block : pool.allocated_blocks) {
            cudaFree(block);
        }
        
        pool.available_blocks.clear();
        pool.allocated_blocks.clear();
    }
    memory_pools.clear();
}

float* saxsKernel::getPooledMemory(size_t size)
{
    size_t bytes_needed = size * sizeof(float);
    
    // Find the smallest pool that can accommodate this size
    for (size_t pool_size : common_sizes) {
        if (pool_size * sizeof(float) >= bytes_needed) {
            MemoryPool& pool = memory_pools[pool_size];
            
            if (!pool.available_blocks.empty()) {
                // Get block from pool
                float* block = pool.available_blocks.back();
                pool.available_blocks.pop_back();
                pool.allocated_blocks.push_back(block);
                pool.allocated_count++;
                return block;
            } else if (pool.allocated_count < pool.total_blocks * 2) {
                // Pool is empty but we can allocate more
                float* block;
                cudaError_t err = cudaMalloc(&block, pool.block_size);
                if (err == cudaSuccess) {
                    pool.allocated_blocks.push_back(block);
                    pool.allocated_count++;
                    pool.total_blocks++;
                    return block;
                }
            }
            break;  // Found appropriate pool but no memory available
        }
    }
    
    // Fallback to regular allocation
    float* block;
    cudaMalloc(&block, bytes_needed);
    return block;
}

void saxsKernel::returnPooledMemory(float* ptr, size_t size)
{
    if (!ptr) return;
    
    size_t bytes_needed = size * sizeof(float);
    
    // Find the appropriate pool
    for (size_t pool_size : common_sizes) {
        if (pool_size * sizeof(float) >= bytes_needed) {
            MemoryPool& pool = memory_pools[pool_size];
            
            // Check if this pointer belongs to our pool
            auto it = std::find(pool.allocated_blocks.begin(), pool.allocated_blocks.end(), ptr);
            if (it != pool.allocated_blocks.end()) {
                // Return to available pool
                pool.allocated_blocks.erase(it);
                pool.available_blocks.push_back(ptr);
                return;
            }
            break;
        }
    }
    
    // Not from our pool, free directly
    cudaFree(ptr);
}

void saxsKernel::initializeCudaStreams()
{
    nvtxRangePush("saxsKernel::initializeCudaStreams");
    
    // Create streams for overlapping operations
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);
    cudaStreamCreate(&fft_stream);
    
    // Create events for synchronization
    cudaEventCreate(&compute_done);
    cudaEventCreate(&transfer_done);
    cudaEventCreate(&fft_done);
    
    // Set stream priorities (higher priority = lower number)
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    
    cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&fft_stream, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking, priority_low);
    
    std::cout << "CUDA Streams initialized for overlapped execution" << std::endl;
    std::cout << "  Compute stream: High priority" << std::endl;
    std::cout << "  FFT stream: High priority" << std::endl;
    std::cout << "  Transfer stream: Low priority" << std::endl;
    
    nvtxRangePop();
}

void saxsKernel::cleanupCudaStreams()
{
    // Synchronize all streams before cleanup
    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(transfer_stream);
    cudaStreamSynchronize(fft_stream);
    
    // Destroy streams
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);
    cudaStreamDestroy(fft_stream);
    
    // Destroy events
    cudaEventDestroy(compute_done);
    cudaEventDestroy(transfer_done);
    cudaEventDestroy(fft_done);
}

void saxsKernel::optimizeKernelLaunchParams()
{
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Optimize for rhoCartKernel (most computationally intensive)
    int minGridSize = 0, blockSize = 256;  // Initialize with defaults
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, rhoCartKernel, 0, 0);
    if (err != cudaSuccess) {
        blockSize = 256;  // Fallback to safe default
        std::cout << "Warning: Could not determine optimal block size, using default: " << blockSize << std::endl;
    }
    optimal_block_size_1d = blockSize;
    
    // Calculate optimal 3D block dimensions for 3D kernels
    // Aim for ~256 threads per block for good occupancy
    int target_threads = 256;
    int dim_size = static_cast<int>(cbrt(target_threads));  // Cube root for 3D
    
    // Adjust based on GPU architecture
    if (prop.major >= 7) {  // Volta/Turing/Ampere architecture
        optimal_block_size_3d = dim3(16, 8, 2);  // 256 threads, optimized for memory coalescing
    } else if (prop.major >= 6) {  // Pascal architecture
        optimal_block_size_3d = dim3(8, 8, 4);   // 256 threads
    } else {  // Older architectures
        optimal_block_size_3d = dim3(8, 8, 8);   // 512 threads (older GPUs handle larger blocks better)
    }
    
    // Calculate maximum theoretical occupancy
    int temp_occupancy = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&temp_occupancy, rhoCartKernel, optimal_block_size_1d, 0);
    if (err == cudaSuccess) {
        max_occupancy = temp_occupancy;
    } else {
        max_occupancy = 1;  // Safe fallback
    }
    
    std::cout << "GPU Optimization Info:" << std::endl;
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Optimal 1D block size: " << optimal_block_size_1d << std::endl;
    std::cout << "  Optimal 3D block size: " << optimal_block_size_3d.x << "x" 
              << optimal_block_size_3d.y << "x" << optimal_block_size_3d.z << std::endl;
    std::cout << "  Max blocks per SM: " << max_occupancy << std::endl;
    
    nvtxRangePop();
}

saxsKernel::~saxsKernel()
{
    // Free pinned memory
    if (h_particles_pinned) cudaFreeHost(h_particles_pinned);
    if (h_scatter_pinned) cudaFreeHost(h_scatter_pinned);
    
    // Free unified memory
    cleanupUnifiedMemory();
    
    // Free memory pools
    cleanupMemoryPools();
    
    // Free FFT work area and destroy plan
    if (fftWorkArea) cudaFree(fftWorkArea);
    cufftDestroy(cufftPlan);
}
