---
title: Understanding Parameters
layout: default
---

# Understanding CuSAXS Parameters

Learn how to optimize parameters for your calculations.

## Essential Parameters

### Grid Size (`-g`)
Controls calculation accuracy and memory usage:

```bash
# Cubic grid (recommended for most systems)
-g 128

# Non-cubic grid (for anisotropic systems)  
-g 256 128 64
```

**Guidelines:**
- Small systems (<5k atoms): 64³
- Medium systems (5-20k atoms): 128³  
- Large systems (>20k atoms): 256³

### Frame Range (`-b`, `-e`)
Specify which trajectory frames to analyze:

```bash
# Analyze frames 1000-2000
-b 1000 -e 2000

# Skip frames with --dt
-b 0 -e 1000 --dt 10  # Every 10th frame
```

## Algorithm Parameters

### Interpolation Order (`--order`)
Higher order = better accuracy but slower:

```bash
--order 2    # Fast, lower accuracy
--order 4    # Recommended balance  
--order 6    # Highest accuracy, slowest
```

### Grid Scaling (`--Scale`)
Affects grid spacing and q-range:

```bash
--Scale 2.0    # Larger spacing, lower q-max
--Scale 3.0    # Recommended default
--Scale 4.0    # Smaller spacing, higher q-max
```

## System-Specific Parameters

### Water Models
Include water scattering corrections:

```bash
--water tip3p   # TIP3P water model
--water tip4p   # TIP4P water model  
--water spce    # SPC/E water model
```

### Ion Corrections
Account for ions in solution:

```bash
--na 10 --cl 10    # 10 Na+ and 10 Cl- ions
--simulation npt   # NPT ensemble
```

## Performance Optimization

### Memory vs Speed
Balance memory usage and calculation speed:

| Grid Size | GPU Memory | Speed | Use Case |
|-----------|------------|-------|----------|
| 64³ | ~1GB | Fast | Testing |
| 128³ | ~4GB | Good | Production |
| 256³ | ~16GB | Slow | High resolution |

### Example Configurations

**Quick test:**
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 100 --order 2
```

**Production run:**
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 2000 --order 4 --water tip4p
```

**High precision:**
```bash
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 1000 --order 6 --Scale 4.0
```