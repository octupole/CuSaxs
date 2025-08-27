---
title: Command Line Reference
layout: page
---

# Command Line Reference

Complete reference for CuSAXS command-line interface.

## Basic Syntax

```bash
./CuSAXS [OPTIONS] -s topology.tpr -x trajectory.xtc -g GRID -b BEGIN -e END
```

## Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `-s, --topology` | string | GROMACS topology file | `-s protein.tpr` |
| `-x, --trajectory` | string | GROMACS trajectory file | `-x md.xtc` |
| `-g, --grid` | int[1-3] | Grid size for FFT | `-g 128` or `-g 256 128 64` |
| `-b, --begin` | int | Starting frame number | `-b 0` |
| `-e, --end` | int | Ending frame number | `-e 1000` |

## Optional Parameters

### Output Control
```bash
-o, --out STRING         Output file prefix (default: "saxs_output")
--dt INT                 Frame interval (default: 1)
```

### Algorithm Parameters
```bash
--order INT              B-spline interpolation order (1-6, default: 4)
--Scale FLOAT            Grid scaling factor (default: 2.5)
--bin, --Dq FLOAT        Histogram bin size for q-space
-q, --qcut FLOAT         Cutoff in reciprocal space
--gridS INT INT INT      Scaled grid dimensions
```

### Water and Ion Corrections
```bash
--water STRING           Water model (tip3p, tip4p, tip4pew, tip5p, spce, spc)
--na INT                 Number of sodium atoms (default: 0)
--cl INT                 Number of chlorine atoms (default: 0)
--simulation STRING      Simulation type (npt, nvt)
```

## Complete Examples

### Basic protein calculation
```bash
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 \
         --water tip4p --order 4 -o protein_saxs
```

### High-resolution membrane system
```bash
./CuSAXS -s membrane.tpr -x membrane.xtc -g 256 128 128 -b 500 -e 2000 \
         --Scale 3.0 --bin 0.005 --water tip4p --simulation npt
```

### Ion solution with custom output
```bash
./CuSAXS -s solution.tpr -x solution.xtc -g 128 -b 0 -e 5000 \
         --na 20 --cl 20 --dt 5 -o ion_solution --order 6
```

### Quick test run
```bash
./CuSAXS -s test.tpr -x test.xtc -g 64 -b 0 -e 100 \
         --order 2 -o quick_test
```

## Output Files

With prefix `output_name`, CuSAXS generates:

- `output_name_profile.dat` - SAXS intensity I(q) vs scattering vector q
- `output_name_histogram.dat` - Radial distribution of intensities  
- `output_name_parameters.log` - Complete calculation parameters
- `output_name_timing.log` - Performance and GPU utilization metrics

## Error Handling

Common exit codes:

| Code | Meaning | Solution |
|------|---------|----------|
| 0 | Success | - |
| 1 | Invalid arguments | Check parameter syntax |
| 2 | File not found | Verify file paths exist |
| 3 | CUDA error | Check GPU and CUDA installation |
| 4 | Memory error | Reduce grid size or free GPU memory |
| 5 | Trajectory error | Verify trajectory format and integrity |

## Performance Tips

1. **Start small**: Test with `-g 64` and small frame ranges first
2. **Monitor memory**: Use `nvidia-smi` to watch GPU memory usage
3. **Use --dt**: Skip frames to reduce calculation time for long trajectories
4. **Choose appropriate order**: Order 4 is usually the best balance
5. **Grid selection**: Use cubic grids unless system is highly anisotropic