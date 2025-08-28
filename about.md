---
layout: page
permalink: /about/
---
## About CuSAXS

CuSAXS is a computational tool designed for structural biology and materials science researchers who need to calculate SAXS intensity profiles from molecular dynamics simulations. The software leverages GPU acceleration to significantly speed up SAXS calculations while maintaining high accuracy through advanced B-spline interpolation methods.

## Scientific Background

Small-Angle X-ray Scattering (SAXS) is a powerful experimental technique used to study the structure and dynamics of biological macromolecules and materials in solution. CuSAXS bridges the gap between molecular dynamics simulations and experimental SAXS data by providing accurate theoretical predictions.

### Implementation Details

CuSAXS implements state-of-the-art algorithms for SAXS calculation:

1. **Density Interpolation**: Uses high-order B-spline interpolation to map atomic positions onto regular grids
2. **Form Factor Calculation**: Applies atomic form factors with proper q-dependence  
3. **Fourier Transform**: Employs cuFFT for efficient 3D FFT calculations
4. **Scattering Averaging**: Performs proper orientational and ensemble averaging

## Installation

### System Requirements

#### Hardware
- NVIDIA GPU with compute capability 7.5 or higher (GTX 1650 Ti, RTX 20xx series or newer)
- At least 4GB GPU memory (8GB+ recommended for large systems)

#### Software
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.16 or later
- **Python**: 3.8 or later
- **Qt6**: For optional GUI interface
- **Compiler**: GCC 9.0+ or Clang 10.0+

### Installation Steps

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install cmake build-essential python3-dev python3-pybind11-dev libomp-dev
# For optional GUI support
sudo apt install qt6-base-dev qt6-tools-dev qt6-tools-dev-tools
```

#### 2. Install CUDA Toolkit

Follow the [official NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads) for your system.

#### 3. Clone and Build

```bash
git clone https://github.com/your-username/CuSAXS.git
cd CuSAXS
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Optional: disable GUI if Qt6 is not available
# cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF

# Build
make -j$(nproc)
```

#### 4. Conda Environment (Recommended)

For easier dependency management:

```bash
conda env create -f environment.yml
conda activate cusaxs
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Architecture

CuSAXS is built with a modular architecture:

- **Core Engine** (`CuSAXS.cu`): Main CUDA kernels and calculation routines
- **Execution Layer** (`Exec/`): Command-line interface and parameter handling
- **SAXS Algorithms** (`Saxs/`): B-spline interpolation and scattering calculations
- **System Interface** (`System/`): Trajectory reading and atom management
- **Utilities** (`Utilities/`): Mathematical functions and error handling
- **GUI** (`gui/`): Optional Qt6-based graphical interface
- **Python Integration** (`pysrc/`): Analysis and post-processing scripts

## Performance Characteristics

### GPU Memory Usage

CuSAXS efficiently manages GPU memory through:
- Dynamic memory allocation based on grid size
- Streaming calculations for large trajectories
- Optimized data structures for CUDA kernels

### Scaling Performance

Performance scales with:
- **Grid size**: O(N³ log N) due to 3D FFT
- **Number of atoms**: Linear scaling with proper optimization
- **Trajectory length**: Linear scaling with frame processing

## Contributing

We welcome contributions to CuSAXS! Here's how to get involved:

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/your-username/CuSAXS.git`
3. Create a feature branch: `git checkout -b feature-name`
4. Set up the development environment following the installation guide
5. Make your changes with proper documentation
6. Add tests if applicable
7. Submit a pull request

### Contribution Guidelines

- Follow the existing code style and conventions
- Add documentation for new features
- Include unit tests where appropriate
- Update the changelog for significant changes
- Ensure all tests pass before submitting

### Areas for Contribution

- **Algorithm Improvements**: New interpolation methods, optimization techniques
- **File Format Support**: Additional trajectory and topology formats
- **Analysis Tools**: New Python scripts for specific analysis workflows
- **Performance Optimization**: CUDA kernel improvements, memory management
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: Unit tests, integration tests, and validation against experimental data

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/your-username/CuSAXS/blob/main/LICENSE) file for details.

## Citation

If you use CuSAXS in your research, please cite:

```
[Citation will be added upon publication]
```

## Acknowledgments

- NVIDIA for CUDA toolkit and cuFFT library
- The fmt library developers
- The pybind11 community  
- Qt project for the GUI framework
- GROMACS developers for trajectory format specifications

## Contact

For questions, suggestions, or collaborations:
- **Issues**: [GitHub Issues](https://github.com/your-username/CuSAXS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/CuSAXS/discussions)

---

**Note**: This software is provided as-is for research purposes. Please validate results against experimental data and established methods.