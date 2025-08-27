---
layout: default
title: API Reference
permalink: /api/
---

# CuSAXS API Reference

Complete reference for CuSAXS command-line interface, parameters, and file formats.

## Command Line Interface

### Basic Syntax

```bash
./CuSAXS [OPTIONS] -s topology.tpr -x trajectory.xtc -g GRID -b BEGIN -e END
```

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `-s, --topology` | string | Topology file (.tpr format from GROMACS) |
| `-x, --trajectory` | string | Trajectory file (.xtc format from GROMACS) |
| `-g, --grid` | int[1-3] | Grid size for FFT calculations |
| `-b, --begin` | int | Starting frame number |
| `-e, --end` | int | Ending frame number |

#### Grid Size Specification

The `-g` parameter accepts:
- **Single value**: `128` (creates 128³ cubic grid)
- **Three values**: `256 128 64` (creates 256×128×64 non-cubic grid)

### Optional Parameters

#### Output Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-o, --out` | string | "saxs_output" | Output file prefix |
| `--dt` | int | 1 | Frame interval (read every dt frames) |

#### Algorithm Parameters  
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--order` | int | 4 | 1-6 | B-spline interpolation order |
| `--Scale` | float | 2.5 | >0 | Grid scaling factor |
| `--bin, --Dq` | float | auto | >0 | Histogram bin size for q-space sampling |
| `-q, --qcut` | float | auto | >0 | Cutoff in reciprocal space |

#### Grid Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gridS` | int[3] | auto | Scaled grid dimensions |

#### Water and Ion Corrections
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--water` | string | none | Water model for scattering corrections |
| `--na` | int | 0 | Number of sodium atoms |
| `--cl` | int | 0 | Number of chlorine atoms |

#### Simulation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--simulation` | string | auto | Simulation type ("npt" or "nvt") |

### Water Models

Supported water models for `--water` parameter:

| Model | Description |
|-------|-------------|
| `tip3p` | TIP3P water model |
| `tip4p` | TIP4P water model |
| `tip4pew` | TIP4P-Ew water model |
| `tip5p` | TIP5P water model |
| `spce` | SPC/E water model |
| `spc` | SPC water model |

## Examples

### Basic Usage Examples

#### Protein in water
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 --water tip4p --order 4
```

#### High-resolution calculation
```bash  
./CuSAXS -s system.tpr -x traj.xtc -g 256 256 128 -b 500 -e 2000 --Scale 3.0 --bin 0.01
```

#### With ion corrections
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 --na 10 --cl 10 --simulation npt
```

#### Processing subset of frames
```bash
./CuSAXS -s traj.tpr -x traj.xtc -g 128 -b 1000 -e 5000 --dt 10 -o membrane_saxs
```

### Advanced Configuration Examples

#### Custom grid scaling
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.0 --gridS 256 256 256 -b 0 -e 1000
```

#### High-order interpolation
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 --order 6 --water spce
```

## Output Files

CuSAXS generates several output files with the specified prefix:

| File | Description |
|------|-------------|
| `{prefix}_profile.dat` | SAXS intensity I(q) vs scattering vector q |
| `{prefix}_histogram.dat` | Radial distribution of scattering intensities |
| `{prefix}_parameters.log` | Calculation parameters and statistics |
| `{prefix}_timing.log` | Performance metrics and GPU utilization |

### Output File Formats

#### SAXS Profile (`_profile.dat`)
```
# q (nm^-1)    I(q) (arbitrary units)
0.050000      1.234567e+03
0.055000      1.198765e+03
0.060000      1.145678e+03
...
```

#### Parameters Log (`_parameters.log`)
```
CuSAXS Calculation Parameters
=============================
Topology file: protein.tpr
Trajectory file: protein.xtc
Grid size: 128 x 128 x 128
Frame range: 0 to 1000 (step 1)
Interpolation order: 4
Water model: tip4p
...
```

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 1 | Invalid command line arguments | Check parameter syntax |
| 2 | File not found | Verify file paths |
| 3 | CUDA initialization failed | Check CUDA installation |
| 4 | Insufficient GPU memory | Reduce grid size |
| 5 | Trajectory read error | Verify trajectory format |

## Performance Guidelines

### Grid Size Selection

| System Size | Recommended Grid | Memory Usage | Performance |
|-------------|------------------|--------------|-------------|
| Small (<5k atoms) | 64³ | ~1GB | Fast |
| Medium (5-20k atoms) | 128³ | ~4GB | Good |
| Large (>20k atoms) | 256³ | ~16GB | Slower |

### Memory Requirements

Approximate GPU memory usage:
- **64³ grid**: ~1GB
- **128³ grid**: ~4GB  
- **256³ grid**: ~16GB
- **512³ grid**: ~64GB

### Interpolation Order Effects

| Order | Accuracy | Performance | Use Case |
|-------|----------|-------------|----------|
| 1 | Low | Fastest | Quick tests |
| 2-3 | Good | Fast | General use |
| 4 | High | Moderate | Recommended |
| 5-6 | Highest | Slow | High precision |

## Python API

### Analysis Scripts

Location: `pysrc/` directory

#### Compare SAXS Profiles
```python
python pysrc/compareIqs.py profile1.dat profile2.dat
```

#### Fit Bilayer Form Factors  
```python
python pysrc/fitBilayer.py saxs_profile.dat
```

#### Topology Analysis
```python
python pysrc/topology.py system.tpr
```

#### Trajectory Interface
```python
from trajectory_interface import CuSAXSTrajectory

# Load trajectory
traj = CuSAXSTrajectory("system.tpr", "traj.xtc")

# Access frame data
frame = traj.get_frame(100)
positions = frame.positions
```

## GUI Reference

### Main Window Parameters

The GUI provides access to all command-line parameters through:
- **Basic tab**: Essential parameters (topology, trajectory, grid, frames)
- **Advanced tab**: Algorithm parameters (order, scaling, water model)
- **Output tab**: File naming and location options

### Parameter Validation

The GUI provides real-time validation:
- **Red indicators**: Invalid parameter values
- **Yellow indicators**: Suboptimal but valid parameters  
- **Green indicators**: Recommended parameter ranges

---

## See Also

- [Quick Start Tutorial](../tutorials/quick-start.html)
- [Performance Optimization Guide](../tutorials/performance.html)
- [Troubleshooting](../tutorials/troubleshooting.html)