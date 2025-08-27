# API Documentation

## Core Classes

### saxsKernel

The main computational engine for SAXS calculations.

```cpp
class saxsKernel
{
public:
    saxsKernel(int _nx, int _ny, int _nz, int _order);
    void runPKernel(int frame, float time, 
                   std::vector<std::vector<float>>& positions,
                   std::map<std::string, std::vector<int>>& atom_types,
                   std::vector<std::vector<float>>& box);
    std::vector<std::vector<double>> getSaxs();
    void createMemory();
    void setcufftPlan(int nnx, int nny, int nnz);
    double getCudaTime();
    ~saxsKernel();
};
```

**Constructor Parameters:**
- `_nx, _ny, _nz`: Grid dimensions for FFT calculations
- `_order`: B-spline interpolation order (1-6)

**Key Methods:**

#### `runPKernel()`
Executes the main SAXS calculation kernel on GPU.

**Parameters:**
- `frame`: Frame number being processed
- `time`: Simulation time
- `positions`: Atomic coordinates [N×3]
- `atom_types`: Map of atom types to indices
- `box`: Simulation box vectors [3×3]

#### `getSaxs()`
Returns the calculated SAXS intensity profile.

**Returns:** 2D vector containing [q, I(q)] pairs

#### `createMemory()`
Allocates GPU memory buffers for calculations. Must be called before `runPKernel()`.

---

### RunSaxs

High-level interface for SAXS calculations from MD trajectories.

```cpp
class RunSaxs
{
public:
    RunSaxs(std::string tpr, std::string xtc);
    void Run(int start_frame, int end_frame, int frame_interval);
    ~RunSaxs();
};
```

**Constructor Parameters:**
- `tpr`: Path to GROMACS topology file
- `xtc`: Path to GROMACS trajectory file

**Methods:**

#### `Run()`
Processes trajectory frames and calculates SAXS.

**Parameters:**
- `start_frame`: First frame to process
- `end_frame`: Last frame to process  
- `frame_interval`: Process every nth frame

---

### AtomCounter

Manages atomic information and scattering factors.

```cpp
class AtomCounter
{
public:
    AtomCounter();
    void addAtom(const std::string& type, const std::vector<float>& position);
    std::map<std::string, std::vector<int>> getAtomTypeMap();
    int getTotalAtoms();
    std::vector<float> getScatteringFactors(float q);
};
```

**Methods:**

#### `addAtom()`
Adds an atom to the internal registry.

**Parameters:**
- `type`: Atom type string (e.g., "C", "N", "O")
- `position`: 3D coordinates

#### `getScatteringFactors()`
Returns atomic form factors for given q-value.

**Parameters:**
- `q`: Scattering vector magnitude

**Returns:** Vector of form factors for each atom type

---

### Cell

Manages simulation cell parameters and periodic boundary conditions.

```cpp
class Cell
{
public:
    Cell(const std::vector<std::vector<float>>& box_vectors);
    void updateBox(const std::vector<std::vector<float>>& new_box);
    std::vector<float> getBoxLengths();
    float getVolume();
    bool isOrthogonal();
};
```

**Constructor Parameters:**
- `box_vectors`: 3×3 matrix of simulation box vectors

**Methods:**

#### `updateBox()`
Updates box parameters for new frame.

#### `getVolume()`
Returns simulation box volume.

---

## Utility Classes

### BSpline

B-spline interpolation utilities for atomic density mapping.

```cpp
class BSpline
{
public:
    static std::vector<float> calculateWeights(float x, int order);
    static void interpolateToGrid(const std::vector<float>& positions,
                                 std::vector<float>& grid,
                                 int nx, int ny, int nz, int order);
};
```

#### `calculateWeights()`
Computes B-spline weights for interpolation.

**Parameters:**
- `x`: Fractional grid position
- `order`: B-spline order

**Returns:** Vector of interpolation weights

---

### Scattering

Atomic form factor and scattering utilities.

```cpp
class Scattering
{
public:
    static float getFormFactor(const std::string& element, float q);
    static float getElectronDensity(const std::string& element);
    static std::map<std::string, float> getWaterModel(const std::string& model);
};
```

#### `getFormFactor()`
Returns atomic form factor for given element and q-value.

**Parameters:**
- `element`: Chemical element symbol
- `q`: Scattering vector magnitude

#### `getWaterModel()`
Returns scattering parameters for water models.

**Parameters:**
- `model`: Water model name ("tip3p", "tip4p", "spc", etc.)

---

## Configuration Classes

### Options

Global configuration parameters.

```cpp
class Options
{
public:
    static std::string tpr_file, xtc_file;
    static int nx, ny, nz;          // Original grid size
    static int nnx, nny, nnz;       // Scaled grid size
    static float sigma;              // Grid scaling factor
    static float Dq;                 // q-space bin size
    static float Qcut;              // q-space cutoff
    static int order;               // B-spline order
    static std::string Wmodel;      // Water model
    static int Sodium, Chlorine;    // Ion counts
    static std::string outFile;     // Output filename
    static std::string Simulation;  // "npt" or "nvt"
};
```

---

## GPU Kernels

### CUDA Device Functions

```cpp
__global__ void calculate_density_kernel(
    float* positions,
    float* grid,
    int num_atoms,
    int nx, int ny, int nz,
    int order
);

__global__ void calculate_histogram(
    cuFloatComplex* Iq,
    double* histogram,
    double* nhist,
    float* form_factors,
    int nx, int ny, int nz,
    float bin_size,
    float kcut,
    int num_bins,
    float normalization
);

__global__ void zeroDensityKernel(
    cuFloatComplex* data,
    int size
);
```

#### `calculate_density_kernel()`
Maps atomic positions to density grid using B-spline interpolation.

#### `calculate_histogram()`
Computes radial average of scattering intensities.

#### `zeroDensityKernel()`
Initializes GPU memory arrays to zero.

---

## Python Interface

### Embedded Python Functions

The software integrates Python through pybind11 for analysis tasks:

```python
import topology
import trajectories

# Load system information
topo = topology.Topology("system.tpr", "traj.xtc")
topo.count_molecules()

# Process trajectories  
traj = trajectories.TrajectoryProcessor("system.tpr", "traj.xtc")
frames = traj.read_frames(start=0, end=1000, step=10)
```

---

## Error Handling

### Exception Types

```cpp
namespace CuSAXS {
    class CudaError : public std::exception {
        // CUDA-related errors
    };
    
    class FileError : public std::exception {
        // File I/O errors
    };
    
    class ParameterError : public std::exception {
        // Invalid parameter errors
    };
}
```

---

## Memory Management

### GPU Memory Layout

The software uses thrust vectors for automatic memory management:

```cpp
// Device vectors (GPU memory)
thrust::device_vector<float> d_grid;
thrust::device_vector<cuFloatComplex> d_Iq;
thrust::device_vector<double> d_histogram;

// Host vectors (CPU memory)
thrust::host_vector<float> h_positions;
thrust::host_vector<double> h_histogram;
```

Raw pointers are obtained when needed for CUDA kernels:
```cpp
float* d_grid_ptr = thrust::raw_pointer_cast(d_grid.data());
```

---

## Build System Integration

### CMake Targets

```cmake
# Main executable
add_executable(CuSAXS CuSAXS.cu)

# Component libraries
add_library(saxs STATIC saxsKernel.cu saxsDeviceKernels.cu)
add_library(system STATIC AtomCounter.cpp Cell.cpp) 
add_library(utils STATIC BSpline.cpp Scattering.cpp)

# Link dependencies
target_link_libraries(CuSAXS 
    saxs system utils 
    ${CUDA_LIBRARIES} cufft cublas 
    fmt::fmt Python3::Python pybind11::module
    OpenMP::OpenMP_CXX)
```

---

## Performance Considerations

### GPU Architecture Support

The software targets CUDA compute capabilities:
- **75**: Turing (RTX 20xx, GTX 16xx)
- **80**: Ampere (RTX 30xx, A100)
- **86**: Ampere (RTX 30xx mobile)
- **89**: Ada Lovelace (RTX 40xx)

### Memory Requirements

Approximate GPU memory usage:
- Grid: `4 × nx × ny × nz` bytes (density)
- FFT: `8 × nx × ny × (nz/2+1)` bytes (complex)
- Histogram: `8 × num_bins` bytes

For a 128³ grid: ~67 MB minimum GPU memory required.

---

## Thread Safety

**Note**: The current implementation is **not thread-safe**. Global variables in the `Options` class and static members require external synchronization for multi-threaded usage.