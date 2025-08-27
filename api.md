---
layout: page
title: API Reference
permalink: /api/
---

# CuSAXS API Reference

Complete reference for CuSAXS command-line interface, parameters, and file formats.

## Quick Navigation

### [Command Line Interface](api/command-line)
Complete command-line reference with examples.

### Parameters Reference
Detailed description of all parameters and their effects.

### File Formats
Input and output file format specifications.

### Python API
Documentation for the Python analysis scripts.

## Command Line Interface

### Basic Syntax

```bash
./CuSAXS [OPTIONS] -s topology.tpr -x trajectory.xtc -g GRID -b BEGIN -e END
```

### Required Parameters

- `-s, --topology` - Topology file (.tpr format from GROMACS)
- `-x, --trajectory` - Trajectory file (.xtc format from GROMACS)  
- `-g, --grid` - Grid size for FFT calculations
- `-b, --begin` - Starting frame number
- `-e, --end` - Ending frame number

### Common Examples

**Basic usage:**
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000
```

**With water corrections:**
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 --water tip4p
```

**High resolution:**
```bash  
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 2000 --order 6
```

## Output Files

CuSAXS generates several output files:

- `{prefix}_profile.dat` - SAXS intensity I(q) vs scattering vector q
- `{prefix}_histogram.dat` - Radial distribution of scattering intensities
- `{prefix}_parameters.log` - Calculation parameters and statistics  
- `{prefix}_timing.log` - Performance metrics and GPU utilization

## Performance Guidelines

### Grid Size Selection

| System Size | Recommended Grid | Memory Usage |
|-------------|------------------|--------------|
| Small (<5k atoms) | 64³ | ~1GB |
| Medium (5-20k atoms) | 128³ | ~4GB |
| Large (>20k atoms) | 256³ | ~16GB |

### Interpolation Order Effects

| Order | Accuracy | Performance | Use Case |
|-------|----------|-------------|----------|
| 1-2 | Low | Fastest | Quick tests |
| 4 | High | Good | Recommended |
| 5-6 | Highest | Slow | High precision |

---

For complete details, see the [full API reference](api/) sections.