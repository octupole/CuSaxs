# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation for GitHub publication
- API documentation with detailed class and method descriptions
- Contributing guidelines with development setup instructions
- Conda environment file for easy dependency management
- MIT license for open source distribution

### Changed
- Updated README with detailed installation and usage instructions
- Improved code documentation and examples

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of CuSAXS
- GPU-accelerated SAXS calculation engine using CUDA
- Support for GROMACS topology (.tpr) and trajectory (.xtc) files
- Optional Qt6-based graphical user interface for parameter exploration
- Python integration with pybind11 for analysis scripts
- B-spline interpolation for atomic density mapping
- FFT-based scattering calculations using cuFFT
- Water model corrections and ion handling
- Command-line interface with CLI11 library
- Multi-architecture CUDA support (compute capability 7.5+)
- OpenMP parallelization for CPU tasks
- Advanced options for grid scaling and q-space sampling

### Features
- **Core SAXS Engine** (`saxsKernel`): High-performance GPU calculations
- **Molecular System Handling**: Topology and trajectory processing
- **B-Spline Utilities**: High-order interpolation methods
- **Scattering Calculations**: Atomic form factors and corrections
- **GUI Application**: User-friendly interface for parameter input
- **Python Analysis Scripts**: Post-processing and fitting tools
- **Memory Management**: Automatic GPU memory handling with thrust
- **Error Handling**: Comprehensive error checking and reporting

### Dependencies
- CUDA Toolkit 11.0+
- Python 3.8+ with pybind11
- Qt6 (optional, for GUI)
- OpenMP for CPU parallelization
- fmt library for string formatting
- CLI11 for command-line parsing

### Supported Platforms
- Linux systems with NVIDIA GPUs
- CUDA compute capabilities: 75, 80, 86, 89
- Tested on Ubuntu 20.04+ and similar distributions

### Performance
- Significant speedup over CPU-based SAXS calculations
- Optimized memory access patterns for GPU architectures
- Scalable to large molecular systems (10K+ atoms)
- Efficient trajectory processing with frame-by-frame analysis

### File Formats
- Input: GROMACS .tpr (topology) and .xtc (trajectory) files
- Output: ASCII data files with q vs I(q) profiles
- Parameter files: JSON-based configuration storage

### Command-Line Interface
- Required parameters: topology file, trajectory file, grid size, frame range
- Optional parameters: output file, interpolation order, water model, ion counts
- Flexible grid sizing: cubic or non-cubic grids
- Frame processing options: interval skipping, time range selection

### Graphical User Interface (Optional)
- **Purpose**: Parameter exploration tool for new users learning the software
- File selection dialogs with format validation
- Parameter input forms with range checking and real-time feedback
- Real-time progress monitoring during calculations
- Advanced options dialog for expert parameters
- Settings persistence between sessions
- **Note**: Not required for production use - command-line interface is sufficient

### Python Integration
- Embedded Python interpreter for analysis tasks
- Topology analysis and molecule classification
- Trajectory processing utilities
- Form factor fitting scripts
- Comparative analysis tools

### Documentation
- Comprehensive API documentation
- Usage examples and tutorials
- Installation and build instructions
- Performance optimization guidelines
- Troubleshooting guide

## Development History

### Recent Commits (from git log)
- `4678b11` - Removed bug on new fmt
- `645232a` - gui with .ui
- `d5eb0f3` - test
- `95271fa` - Version based on Qt6 gui without .ui files
- `7a32906` - Added new version of fmt
- `248e154` - Make it faster, but removing reference to xtc library
- `59a14cf` - added conda environment and pybond11
- `fcaf67e` - Modified
- `3ba314b` - Added python scripts
- `df73e76` - Place histograms in double, now reproduce nvt results

### Notable Improvements
- Transitioned from Qt5 to Qt6 for modern GUI support
- Updated to latest fmt library for improved string formatting
- Added conda environment support for easier dependency management
- Enhanced Python integration with pybind11
- Improved numerical precision with double-precision histograms
- Performance optimizations for trajectory processing
- Bug fixes in GUI components and core calculations

## Future Plans

### Version 1.1.0 (Planned)
- Enhanced water model support
- Additional trajectory formats (DCD, PDB)
- Improved error handling and user feedback
- Performance profiling and optimization tools
- Extended Python API for custom analysis

### Version 1.2.0 (Planned)
- Multi-GPU support for large systems
- Real-time visualization of SAXS profiles
- Automated parameter optimization
- Integration with other MD software packages
- Web-based interface option

### Long-term Goals
- Machine learning-enhanced form factor fitting
- Cloud computing support
- Integration with experimental data formats
- Advanced visualization and analysis tools
- Community plugin system

---

**Note**: This changelog follows semantic versioning. Breaking changes will increment the major version, new features will increment the minor version, and bug fixes will increment the patch version.