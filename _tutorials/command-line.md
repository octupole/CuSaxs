---
title: Command Line Interface
layout: default
---

# Command Line Interface Tutorial

Master the CuSAXS command-line interface for efficient SAXS calculations.

## Basic Command Structure

```bash
./CuSAXS [OPTIONS] -s topology.tpr -x trajectory.xtc -g GRID -b BEGIN -e END
```

## Step-by-Step Walkthrough

### 1. Prepare Your Files

Ensure you have:
- **Topology file** (`.tpr`): Contains system structure and parameters
- **Trajectory file** (`.xtc`): Contains atomic positions over time

```bash
# Check files exist
ls -la system.tpr trajectory.xtc
```

### 2. Basic Calculation

Start with a simple calculation:

```bash
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 100
```

**What this does:**
- Uses 128³ grid size (good for most systems)
- Processes frames 0-100 (first 101 frames)
- Creates output files with default prefix `saxs_output`

### 3. Understanding Output

CuSAXS generates several files:

```bash
# Main SAXS profile
saxs_output_profile.dat    # I(q) vs q data

# Diagnostic files
saxs_output_parameters.log # All calculation parameters
saxs_output_timing.log     # Performance metrics
saxs_output_histogram.dat  # Radial distribution
```

### 4. Common Workflows

#### Protein in Water
```bash
./CuSAXS -s protein.tpr -x protein_md.xtc -g 128 -b 0 -e 1000 \
         --water tip4p --order 4 -o protein_saxs
```

#### Large System (High Memory)
```bash
./CuSAXS -s large_system.tpr -x trajectory.xtc -g 256 -b 500 -e 2000 \
         --Scale 3.0 -o large_system_saxs
```

#### Quick Test Run
```bash
./CuSAXS -s test.tpr -x test.xtc -g 64 -b 0 -e 50 \
         --order 2 -o quick_test
```

#### Process Every 10th Frame
```bash
./CuSAXS -s system.tpr -x long_traj.xtc -g 128 -b 0 -e 10000 \
         --dt 10 -o sampled_saxs
```

### 5. Parameter Optimization

#### Grid Size Selection
```bash
# Small system (<5k atoms)
-g 64

# Medium system (5-20k atoms)  
-g 128

# Large system (>20k atoms)
-g 256

# Non-cubic for anisotropic systems
-g 256 128 64
```

#### Interpolation Order
```bash
--order 2    # Fast, lower accuracy
--order 4    # Recommended (default)
--order 6    # Highest accuracy, slower
```

#### Water Model Corrections
```bash
--water tip3p    # TIP3P water model
--water tip4p    # TIP4P water model
--water spce     # SPC/E water model
```

### 6. Advanced Options

#### Ion Corrections
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --na 10 --cl 10 --simulation npt
```

#### Custom Scaling
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --Scale 4.0 --bin 0.005
```

#### High Precision
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 1000 \
         --order 6 --Scale 3.5 -q 2.0
```

## Troubleshooting Commands

### Check GPU Memory
```bash
nvidia-smi
```

### Test Small Calculation First
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 32 -b 0 -e 10 --order 1
```

### Verbose Output
```bash
./CuSAXS --verbose -s system.tpr -x traj.xtc -g 128 -b 0 -e 100
```

## Performance Tips

1. **Start small**: Always test with small grid sizes first
2. **Monitor memory**: Watch `nvidia-smi` output during calculations
3. **Use --dt**: Skip frames for long trajectories to save time
4. **Choose grid wisely**: Larger grids need exponentially more memory
5. **Save intermediate results**: Use descriptive output prefixes

## Common Error Solutions

| Error | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce grid size with `-g 64` |
| "File not found" | Check file paths are correct |
| "Invalid trajectory" | Verify .xtc file integrity |
| "GPU initialization failed" | Check CUDA drivers |

## Next Steps

- [Performance Optimization](/tutorials/performance/) - Maximize your GPU utilization
- [Water Models Tutorial](/tutorials/water-models/) - Handle different water types
- [Python Analysis](/tutorials/python-analysis/) - Post-process your results