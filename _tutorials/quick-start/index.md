---
title: Quick Start Guide
layout: page
---

# Quick Start Guide

Get up and running with CuSAXS in just a few minutes.

## Prerequisites

- NVIDIA GPU with CUDA capability 7.5+
- Linux system (Ubuntu 20.04+ recommended)
- GROMACS trajectory files (.tpr and .xtc)

## Installation

```bash
git clone https://github.com/your-username/CuSAXS.git
cd CuSAXS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Your First Calculation

Run a basic SAXS calculation:

```bash
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 100
```

This command:
- Uses `system.tpr` as topology
- Reads `trajectory.xtc` as trajectory
- Sets 128³ grid size
- Processes frames 0-100

## Output

CuSAXS creates these files:
- `saxs_output_profile.dat` - SAXS intensity I(q) vs q
- `saxs_output_parameters.log` - Calculation parameters
- `saxs_output_timing.log` - Performance metrics

## Next Steps

- [Understanding Parameters](parameters.html)
- [GUI Tutorial](gui-tutorial.html)
- [Performance Optimization](performance.html)