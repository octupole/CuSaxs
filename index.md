---
layout: home
title: Home
---

# CuSAXS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![C++](https://img.shields.io/badge/C++-14-blue.svg)](https://isocpp.org/)

A high-performance GPU-accelerated tool for calculating Small-Angle X-ray Scattering (SAXS) spectra from molecular dynamics trajectories using NVIDIA CUDA.

## Quick Start

Get up and running with CuSAXS in minutes:

```bash
git clone https://github.com/octupole/CuSAXS.git
cd CuSAXS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Basic usage:
```bash
./CuSAXS -s topology.tpr -x trajectory.xtc -g 128 -b 0 -e 1000
```

## Key Features

- **🚀 GPU Acceleration**: Leverages NVIDIA CUDA for high-performance parallel computing
- **🧬 MD Integration**: Direct support for GROMACS topology (.tpr) and trajectory (.xtc) files
- **⚡ Advanced Algorithms**: B-spline interpolation and FFT-based scattering calculations
- **🔧 Flexible Grid Systems**: Supports various grid sizes and scaling factors
- **💧 Water Model Support**: Includes corrections for different water models
- **🧪 Ion Handling**: Accounts for sodium and chlorine ions in solution
- **🖥️ GUI Interface**: Optional Qt6-based graphical interface for parameter exploration
- **🐍 Python Integration**: Embedded Python for advanced analysis and fitting

## Who Uses CuSAXS?

CuSAXS is designed for structural biology and materials science researchers who need to:
- Calculate SAXS intensity profiles from molecular dynamics simulations
- Compare simulation results with experimental SAXS data
- Study protein conformational changes in solution
- Analyze membrane and lipid bilayer structures
- Investigate nanoparticle and polymer assemblies

## Getting Started

1. **[Installation Guide](about#installation)** - Set up CuSAXS on your system
2. **[Basic Tutorial](tutorials)** - Your first SAXS calculation
3. **[API Reference](api)** - Complete command-line options and parameters
4. **[Python Scripts](tutorials/python-analysis)** - Post-processing and analysis tools

## System Requirements

- **GPU**: NVIDIA GPU with compute capability 7.5+ (GTX 1650 Ti, RTX 20xx series or newer)
- **Memory**: At least 4GB GPU memory (8GB+ recommended)
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 11.0 or later

## Performance

CuSAXS delivers significant speedups over CPU-based methods:
- **10-100x faster** than traditional CPU implementations
- Process large trajectories (>10,000 frames) in minutes
- Real-time parameter exploration with GUI interface
- Efficient memory management for large molecular systems

## Community & Support

- **Documentation**: Comprehensive guides and API reference
- **Issues**: Report bugs or request features on [GitHub](https://github.com/octupole/CuSAXS/issues)
- **Discussions**: Get help from the community
- **Contributing**: We welcome contributions - see our [contributing guide](about#contributing)