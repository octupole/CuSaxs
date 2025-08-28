# Contributing to CuSAXS

We welcome contributions to CuSAXS! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Prioritize the community's well-being

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Development Environment**:
   - Linux system with NVIDIA GPU
   - CUDA Toolkit 11.0+
   - CMake 3.16+
   - GCC 9.0+ or Clang 10.0+

2. **Dependencies**:
   - Python 3.8+ with development headers
   - pybind11
   - Qt6 (for GUI development)
   - OpenMP

3. **Git Setup**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Repository Structure

```
CuSAXS/
├── CMakeLists.txt          # Main CMake configuration
├── CuSAXS.cu            # Main executable source
├── Exec/                  # Execution logic and CLI
├── Saxs/                  # Core SAXS calculation kernels
├── System/                # Molecular system handling
├── Utilities/             # Utility classes and functions
├── gui/                   # Qt GUI application
├── pysrc/                 # Python analysis scripts
├── include/               # Header files
├── docs/                  # Documentation
└── build/                 # Build artifacts (git-ignored)
```

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/octupole/CuSAXS.git
   cd CuSAXS
   git remote add upstream https://github.com/original-repo/CuSAXS.git
   ```

2. **Create Development Build**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_GUI=ON
   make -j$(nproc)
   ```

3. **Setup Pre-commit Hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Contributing Process

### 1. Choose or Create an Issue

- Browse [existing issues](https://github.com/your-repo/CuSAXS/issues)
- For new features, create an issue first to discuss the proposal
- Look for issues labeled `good first issue` if you're new to the project

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-brief-description
```

### 3. Make Your Changes

- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Test Your Changes

```bash
# Build and run tests
cd build
make -j$(nproc)
ctest

# Run specific component tests
./test_saxs_kernel
./test_bspline_interpolation

# Test with real data (if available)
./CuSAXS -s test_data/protein.tpr -x test_data/protein.xtc -g 64 -b 0 -e 10
```

## Coding Standards

### C++/CUDA Code

#### Style Guidelines

1. **Naming Conventions**:
   ```cpp
   // Classes: PascalCase
   class SaxsKernel {};
   
   // Functions: camelCase
   void runPKernel() {};
   
   // Variables: snake_case for locals, camelCase for members
   int frame_count;        // local/static
   int gridSize;           // member variable
   
   // Constants: UPPER_SNAKE_CASE
   const int MAX_ORDER = 6;
   
   // CUDA kernels: snake_case with _kernel suffix
   __global__ void calculate_density_kernel() {};
   ```

2. **Header Files**:
   ```cpp
   #ifndef CLASSNAME_H
   #define CLASSNAME_H
   
   // System includes first
   #include <iostream>
   #include <vector>
   
   // Third-party includes
   #include <cuda_runtime.h>
   #include <fmt/core.h>
   
   // Project includes last
   #include "LocalHeader.h"
   
   #pragma once  // Additional safety
   
   class ClassName {
   public:
       // Public interface first
       ClassName();
       ~ClassName();
       
   private:
       // Private members last
       int memberVariable;
   };
   
   #endif
   ```

3. **Function Documentation**:
   ```cpp
   /**
    * @brief Calculates SAXS intensity for given frame.
    * 
    * @param frame Frame number being processed
    * @param positions Atomic coordinates [N×3]
    * @param atom_types Map of atom types to indices
    * @param box Simulation box vectors [3×3]
    * 
    * @throws CudaError if GPU memory allocation fails
    * @throws ParameterError if grid dimensions are invalid
    */
   void runPKernel(int frame, 
                   const std::vector<std::vector<float>>& positions,
                   const std::map<std::string, std::vector<int>>& atom_types,
                   const std::vector<std::vector<float>>& box);
   ```

#### CUDA-Specific Guidelines

1. **Kernel Launch Parameters**:
   ```cpp
   // Use appropriate block sizes for target architectures
   dim3 blockDim(16, 16, 4);  // Total = 1024 threads
   dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                (ny + blockDim.y - 1) / blockDim.y,
                (nz + blockDim.z - 1) / blockDim.z);
   
   calculate_density_kernel<<<gridDim, blockDim>>>(args...);
   ```

2. **Error Checking**:
   ```cpp
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       throw CudaError(cudaGetErrorString(err));
   }
   ```

3. **Memory Management**:
   ```cpp
   // Prefer thrust vectors for automatic management
   thrust::device_vector<float> d_data(size);
   
   // Use raw pointers only when needed for kernels
   float* d_data_ptr = thrust::raw_pointer_cast(d_data.data());
   ```

### Python Code

Follow PEP 8 standards:

```python
"""
Module docstring describing purpose and usage.
"""

import os
import sys
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


class TopologyAnalyzer:
    """Analyzes molecular topology from GROMACS files."""
    
    def __init__(self, tpr_file: str, xtc_file: str) -> None:
        """
        Initialize analyzer with trajectory files.
        
        Args:
            tpr_file: Path to topology file
            xtc_file: Path to trajectory file
        """
        self.tpr_file = tpr_file
        self.xtc_file = xtc_file
    
    def analyze_structure(self) -> Dict[str, int]:
        """
        Analyze molecular structure.
        
        Returns:
            Dictionary containing molecule counts
        """
        # Implementation here
        pass
```

### CMake

```cmake
# Use modern CMake practices (3.16+)
cmake_minimum_required(VERSION 3.16)
project(CuSAXS LANGUAGES CXX C CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Find packages with proper error handling
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Use target-based linking
target_link_libraries(CuSAXS 
    PRIVATE 
        CUDA::cudart 
        CUDA::cufft 
        CUDA::cublas
        fmt::fmt
        OpenMP::OpenMP_CXX
)
```

## Testing

### Unit Tests

1. **Create test files** in appropriate subdirectories:
   ```cpp
   // test_saxs_kernel.cpp
   #include "saxsKernel.h"
   #include <cassert>
   
   void test_kernel_initialization() {
       saxsKernel kernel(64, 64, 64, 4);
       kernel.createMemory();
       // Add assertions
       assert(kernel.getCudaTime() >= 0);
   }
   ```

2. **Add tests to CMake**:
   ```cmake
   enable_testing()
   add_executable(test_saxs_kernel test_saxs_kernel.cpp)
   target_link_libraries(test_saxs_kernel saxs)
   add_test(NAME SaxsKernelTest COMMAND test_saxs_kernel)
   ```

### Integration Tests

Test with real molecular systems:

```bash
# Small protein test
./CuSAXS -s data/1ubq.tpr -x data/1ubq.xtc -g 64 -b 0 -e 100

# Large system test
./CuSAXS -s data/membrane.tpr -x data/membrane.xtc -g 128 -b 0 -e 50
```

### Performance Tests

```cpp
// Benchmark critical functions
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
kernel.runPKernel(frame, positions, atom_types, box);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Kernel execution time: " << duration.count() << " ms" << std::endl;
```

## Documentation

### Code Documentation

1. **Header Documentation**: Document all public interfaces
2. **Implementation Comments**: Explain complex algorithms
3. **CUDA Kernel Documentation**: Describe thread organization and memory access patterns

### User Documentation

1. **Update README.md** for new features
2. **Update docs/API.md** for API changes
3. **Add examples** to demonstrate new functionality

## Performance Considerations

### GPU Optimization

1. **Memory Coalescing**:
   ```cpp
   // Good: coalesced access
   __global__ void coalesced_kernel(float* data) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       data[idx] = computeValue(idx);  // Adjacent threads access adjacent memory
   }
   ```

2. **Shared Memory Usage**:
   ```cpp
   __global__ void shared_memory_kernel(float* input, float* output) {
       __shared__ float shared_data[256];
       
       int tid = threadIdx.x;
       int idx = blockIdx.x * blockDim.x + tid;
       
       // Load to shared memory
       shared_data[tid] = input[idx];
       __syncthreads();
       
       // Compute using shared data
       output[idx] = process(shared_data[tid]);
   }
   ```

3. **Occupancy Considerations**:
   - Target 50%+ occupancy for most kernels
   - Use CUDA occupancy calculator for optimization
   - Profile with nvprof or Nsight Compute

### CPU Optimization

1. **Vectorization**: Use compiler auto-vectorization where possible
2. **OpenMP**: Parallelize CPU-intensive loops
3. **Memory Layout**: Prefer structure-of-arrays for vectorization

## Submitting Changes

### Pull Request Process

1. **Ensure Quality**:
   - All tests pass
   - Code follows style guidelines
   - Documentation is updated
   - Performance impact is considered

2. **Create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   - Use descriptive title and description
   - Reference related issues
   - Include testing information

3. **Code Review Process**:
   - Maintainers will review your changes
   - Address feedback promptly
   - Update documentation as needed
   - Squash commits if requested

### Commit Messages

Use conventional commit format:

```
type(scope): brief description

Detailed explanation of changes made and why.

Fixes #123
```

Types:
- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation changes
- `perf`: performance improvements
- `refactor`: code refactoring
- `test`: adding or updating tests

### Release Process

Maintainers handle releases, but contributors should:

1. Update version numbers appropriately
2. Add entries to CHANGELOG.md
3. Ensure compatibility is maintained

## Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs or request features
- **Email**: Contact maintainers directly for sensitive issues

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic publications (where appropriate)

Thank you for contributing to CuSAXS!