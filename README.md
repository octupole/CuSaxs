# CuSAXS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![C++](https://img.shields.io/badge/C++-14-blue.svg)](https://isocpp.org/)

A high-performance GPU-accelerated tool for calculating Small-Angle X-ray Scattering (SAXS) spectra from molecular dynamics trajectories using NVIDIA CUDA.

## Overview

CuSAXS is a computational tool designed for structural biology and materials science researchers who need to calculate SAXS intensity profiles from molecular dynamics simulations. The software leverages GPU acceleration to significantly speed up SAXS calculations while maintaining high accuracy through advanced B-spline interpolation methods.

### Key Features

- **GPU Acceleration**: Utilizes NVIDIA CUDA for high-performance parallel computing
- **MD Integration**: Direct support for GROMACS topology (.tpr) and trajectory (.xtc) files
- **Advanced Algorithms**: B-spline interpolation and FFT-based scattering calculations
- **Flexible Grid Systems**: Supports various grid sizes and scaling factors
- **Water Model Support**: Includes corrections for different water models
- **Ion Handling**: Accounts for sodium and chlorine ions in solution
- **GUI Interface**: Optional Qt6-based graphical user interface for parameter exploration
- **Python Integration**: Embedded Python for advanced analysis and fitting
- **Cross-Platform**: Supports Linux systems with NVIDIA GPUs

## Requirements

### Hardware
- NVIDIA GPU with compute capability 7.5 or higher (GTX 1650 Ti, RTX 20xx series or newer)
- At least 4GB GPU memory (8GB+ recommended for large systems)

### Software
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.16 or later
- **Python**: 3.8 or later
- **Qt6**: For optional GUI interface (recommended for first-time users only)
- **Compiler**: GCC 9.0+ or Clang 10.0+

### Dependencies
- CUDA Toolkit with cuFFT and cuBLAS
- Python development headers
- pybind11
- OpenMP
- Qt6 (optional, for GUI)

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install cmake build-essential python3-dev python3-pybind11-dev libomp-dev
# For optional GUI support (recommended for first-time users)
sudo apt install qt6-base-dev qt6-tools-dev qt6-tools-dev-tools
```

### 2. Install CUDA Toolkit

Follow the [official NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads) for your system.

### 3. Clone and Build

```bash
git clone https://github.com/your-username/CuSAXS.git
cd CuSAXS
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Optional: disable GUI if Qt6 is not available or not needed
# cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF

# Build
make -j$(nproc)
```

### 4. Conda Environment (Recommended)

For easier dependency management:

```bash
conda env create -f environment.yml
conda activate cusaxs
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Command Line Interface

Basic usage:
```bash
./CuSAXS -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 1000
```

#### Required Parameters
- `-s, --topology`: Topology file (.tpr format from GROMACS)
- `-x, --trajectory`: Trajectory file (.xtc format from GROMACS)
- `-g, --grid`: Grid size for FFT calculations (single value for cubic, or three values for non-cubic)
- `-b, --begin`: Starting frame number
- `-e, --end`: Ending frame number

#### Optional Parameters
- `-o, --out`: Output file prefix (default: "saxs_output")
- `--dt`: Frame interval (read every dt frames, default: 1)
- `--order`: B-spline interpolation order (default: 4, max: 6)
- `--gridS`: Scaled grid dimensions
- `--Scale`: Grid scaling factor (default: 2.5)
- `--bin, --Dq`: Histogram bin size for q-space sampling
- `-q, --qcut`: Cutoff in reciprocal space
- `--water`: Water model for scattering corrections
- `--na`: Number of sodium atoms
- `--cl`: Number of chlorine atoms
- `--simulation`: Simulation type ("npt" or "nvt")

#### Example Commands

Calculate SAXS for a protein in water:
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 --water tip4p --order 4
```

High-resolution calculation with custom grid:
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 256 256 128 -b 500 -e 2000 --Scale 3.0 --bin 0.01
```

Process with ion corrections:
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 --na 10 --cl 10 --simulation npt
```

### Graphical User Interface (Optional)

**Note**: The GUI is optional and primarily designed to help first-time users explore and determine appropriate parameters for their calculations. Once you're familiar with the command-line options, the GUI is not necessary for regular use. Once you're familiar with the command-line options, the GUI is not necessary for regular use.

Launch the GUI:
```bash
./CuSAXS-gui
```

The GUI provides:
- **Parameter exploration**: Intuitive interface for first-time users to understand available options
- **File selection dialogs**: Easy browsing for topology and trajectory files
- **Parameter validation**: Real-time feedback on parameter ranges and combinations
- **Progress monitoring**: Visual feedback during calculations
- **Advanced options dialog**: Access to expert-level parameters
- **Parameter export**: Generate command-line arguments for future batch processing

**When to use the GUI**:
- First time using CuSAXS to understand parameter effects
- Exploring different parameter combinations
- Quick one-off calculations with immediate visual feedback
- Learning the relationship between grid sizes and performance

**When to use command-line**:
- Production calculations and batch processing
- Automated workflows and scripts
- High-throughput analysis of multiple systems
- Integration with job schedulers and HPC environments

### Python Integration

The software includes Python scripts for post-processing and analysis:

```python
# Example: Compare SAXS profiles
python pysrc/compareIqs.py profile1.dat profile2.dat

# Fit bilayer form factors
python pysrc/fitBilayer.py saxs_profile.dat

# Analyze molecular topology
python pysrc/topology.py system.tpr
```

## Output Files

cudaSAXS generates several output files:

- `saxs_profile.dat`: SAXS intensity I(q) vs scattering vector q
- `histogram.dat`: Radial distribution of scattering intensities
- `parameters.log`: Calculation parameters and statistics
- `timing.log`: Performance metrics and GPU utilization

## Performance Optimization

### GPU Memory Management
- Use smaller grid sizes (64-128) for initial testing
- Monitor GPU memory usage with `nvidia-smi`
- For large systems, process trajectories in chunks

### Grid Size Selection
- Start with cubic grids (128³) for most systems
- Increase to 256³ for high-resolution requirements
- Use non-cubic grids for highly anisotropic systems

### Frame Processing
- Use `--dt` parameter to skip frames for faster processing
- Process representative portions of long trajectories
- Balance between statistics and computational time

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce grid size with `-g` parameter
- Process fewer frames per batch
- Close other GPU-accelerated applications

**Slow Performance:**
- Ensure CUDA drivers are properly installed
- Check GPU utilization with `nvidia-smi`
- Verify optimal grid size for your GPU

**Build Errors:**
- Check CUDA version compatibility
- Ensure all dependencies are installed
- Verify CMake finds CUDA toolkit correctly

### Getting Help

1. Check the [Issues](https://github.com/your-username/cudaSAXS/issues) page
2. Verify your system meets the requirements
3. Enable verbose output for debugging: `./CuSAXS --verbose ...`

## Scientific Background

CuSAXS implements state-of-the-art algorithms for SAXS calculation:

1. **Density Interpolation**: Uses high-order B-spline interpolation to map atomic positions onto regular grids
2. **Form Factor Calculation**: Applies atomic form factors with proper q-dependence
3. **Fourier Transform**: Employs cuFFT for efficient 3D FFT calculations
4. **Scattering Averaging**: Performs proper orientational and ensemble averaging

### References

If you use CuSAXS in your research, please cite:

```
[Citation will be added upon publication]
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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