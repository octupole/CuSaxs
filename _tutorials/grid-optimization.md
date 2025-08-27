---
title: Grid Optimization
layout: default
---

# Grid Optimization Tutorial

Master grid size selection and scaling for optimal SAXS calculation accuracy and performance.

## Understanding Grids in CuSAXS

CuSAXS uses regular 3D grids for:
- **Density interpolation**: Mapping atoms to grid points
- **FFT calculations**: Computing scattering via Fourier transforms
- **q-space sampling**: Determining resolution and range

Grid choice affects:
- **Calculation accuracy**
- **Memory usage** (scales as N³)
- **Computation time** (scales as N³ log N)
- **q-range coverage**

## Grid Size Fundamentals

### Cubic Grids (Most Common)
```bash
# Single value creates cubic grid
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000
# Creates 128³ = 2,097,152 grid points
```

### Non-Cubic Grids (Anisotropic Systems)
```bash
# Three values for different dimensions
./CuSAXS -s system.tpr -x traj.xtc -g 256 128 64 -b 0 -e 1000
# Creates 256×128×64 = 2,097,152 grid points (same total)
```

### Grid Size Effects

| Grid Size | Memory | Time | q_max | Use Case |
|-----------|--------|------|-------|----------|
| 32³ | ~32 MB | Seconds | Low | Quick tests |
| 64³ | ~256 MB | Minutes | Medium | Development |
| 128³ | ~2 GB | ~Hour | Good | Production |
| 256³ | ~16 GB | Hours | High | High resolution |
| 512³ | ~128 GB | Days | Very high | Extreme cases |

## Grid Selection Strategy

### Step 1: System Size Assessment
```bash
# Check system dimensions
gmx traj -f trajectory.xtc -s system.tpr -ob box_dims.xvg

# Typical protein system: 8×8×8 nm box → start with 128³
# Large membrane system: 15×15×10 nm → try 256×256×128
```

### Step 2: Convergence Testing
```bash
# Test grid convergence on small subset
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 50 -o test_64
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 50 -o test_128
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 50 -o test_256

# Compare profiles
python pysrc/compareIqs.py test_64_profile.dat test_128_profile.dat
python pysrc/compareIqs.py test_128_profile.dat test_256_profile.dat
```

### Step 3: Quality vs Performance Trade-off
```bash
# Identify minimum acceptable grid size
# Rule of thumb: when profiles differ by <5%, convergence is reached
```

## Grid Scaling Factor

### What is Grid Scaling?
The `--Scale` parameter controls grid spacing:
- **Larger scale** → finer grid spacing → higher q_max
- **Smaller scale** → coarser grid spacing → lower q_max

### Default and Recommendations
```bash
# Default scaling (usually optimal)
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.5

# Fine sampling for high-q data
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 4.0

# Coarse sampling for low-q focus
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.0
```

### Scale Factor Effects

| Scale | Grid Spacing | q_max | Memory | Use Case |
|-------|--------------|-------|---------|----------|
| 1.5 | Coarse | Low | Less | Low-q analysis |
| 2.5 | Standard | Medium | Standard | General use |
| 4.0 | Fine | High | More | High-q details |
| 5.0 | Very fine | Very high | Much more | Atomic details |

### Optimization Example
```bash
# Test different scale factors
for scale in 2.0 2.5 3.0 3.5 4.0; do
    ./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 \
             --Scale $scale -o scale_$scale
done

# Compare q-ranges achieved
for file in scale_*_profile.dat; do
    echo "$file max q:" $(tail -1 $file | awk '{print $1}')
done
```

## Advanced Grid Techniques

### Scaled Grid Dimensions
```bash
# Override automatic grid scaling
./CuSAXS -s system.tpr -x traj.xtc -g 128 --gridS 256 256 256 \
         --Scale 3.0 -b 0 -e 1000

# Use when you need specific grid spacing control
```

### System-Specific Optimization

#### Globular Proteins
```bash
# Typical: cubic grids work well
./CuSAXS -s protein.tpr -x protein.xtc -g 128 \
         --Scale 2.5 --order 4
```

#### Membrane Systems
```bash
# Anisotropic: xy-plane larger than z
./CuSAXS -s membrane.tpr -x membrane.xtc -g 256 256 128 \
         --Scale 3.0 --order 4
```

#### DNA/RNA
```bash
# Long, thin molecules: adapt grid accordingly
./CuSAXS -s dna.tpr -x dna.xtc -g 256 128 128 \
         --Scale 3.5 --order 4
```

#### Protein Complexes
```bash
# Large systems: may need bigger grids
./CuSAXS -s complex.tpr -x complex.xtc -g 256 \
         --Scale 2.5 --order 4
```

## Grid Memory Management

### Memory Estimation
```python
# Approximate GPU memory usage (GB)
def estimate_memory(grid_size):
    if isinstance(grid_size, int):
        total_points = grid_size**3
    else:
        total_points = grid_size[0] * grid_size[1] * grid_size[2]
    
    # Rough estimate: 8 bytes per grid point
    memory_gb = total_points * 8 / (1024**3)
    return memory_gb

# Examples
print(f"64³ grid: {estimate_memory(64):.1f} GB")      # ~2 GB
print(f"128³ grid: {estimate_memory(128):.1f} GB")    # ~16 GB
print(f"256³ grid: {estimate_memory(256):.1f} GB")    # ~128 GB
```

### Memory Optimization Strategies

#### Strategy 1: Non-Cubic Grids
```bash
# Instead of 256³ (16 GB)
# Use 320×256×192 (similar volume, ~12 GB)
./CuSAXS -s system.tpr -x traj.xtc -g 320 256 192
```

#### Strategy 2: Frame Batching
```bash
# Process trajectory in chunks
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 500 -o part1
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 500 -e 1000 -o part2

# Average results
python pysrc/average_profiles.py part1_profile.dat part2_profile.dat
```

#### Strategy 3: Adaptive Grid Sizing
```bash
# Start small and scale up
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 100   # Quick test
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 500  # Medium test
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 1000 # Final run
```

## Grid Quality Assessment

### Convergence Criteria
1. **Profile difference** < 5% between grid sizes
2. **q_max coverage** sufficient for your analysis
3. **Statistical noise** doesn't increase significantly

### Quality Metrics
```bash
# Compare successive grid sizes
python << EOF
import numpy as np

# Load profiles
q1, I1 = np.loadtxt('grid_128_profile.dat', unpack=True)
q2, I2 = np.loadtxt('grid_256_profile.dat', unpack=True)

# Interpolate to common q-grid
I1_interp = np.interp(q2, q1, I1)

# Calculate relative difference
rel_diff = np.abs(I2 - I1_interp) / I1_interp
max_diff = np.max(rel_diff)
mean_diff = np.mean(rel_diff)

print(f"Max relative difference: {max_diff:.3f}")
print(f"Mean relative difference: {mean_diff:.3f}")

# Convergence if max_diff < 0.05 (5%)
if max_diff < 0.05:
    print("Grid convergence achieved!")
else:
    print("Consider larger grid size.")
EOF
```

## Common Grid Problems

### Problem 1: Insufficient Resolution
**Symptoms**: 
- Truncated high-q region
- Missing oscillations
- Poor fit to experimental data at high q

**Solution**:
```bash
# Increase grid size or scale factor
./CuSAXS -s system.tpr -x traj.xtc -g 256 --Scale 3.5
```

### Problem 2: GPU Memory Overflow
**Symptoms**:
- CUDA out of memory errors
- Calculation crashes
- System becomes unresponsive

**Solution**:
```bash
# Reduce grid size
./CuSAXS -s system.tpr -x traj.xtc -g 64 --Scale 2.5

# Or use non-cubic grid
./CuSAXS -s system.tpr -x traj.xtc -g 160 128 96
```

### Problem 3: Slow Performance
**Symptoms**:
- Hours for small calculations
- Low GPU utilization
- Excessive computation time

**Solutions**:
```bash
# Reduce grid size first
./CuSAXS -s system.tpr -x traj.xtc -g 64

# Optimize other parameters
./CuSAXS -s system.tpr -x traj.xtc -g 128 --order 2 --dt 5
```

### Problem 4: Grid Artifacts
**Symptoms**:
- Unphysical oscillations
- Discontinuities in profile
- Strange low-q behavior

**Solutions**:
```bash
# Check grid scaling
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.5

# Increase interpolation order
./CuSAXS -s system.tpr -x traj.xtc -g 128 --order 6
```

## Grid Optimization Workflow

### Phase 1: Quick Assessment
```bash
# 5-minute test with small grid
./CuSAXS -s system.tpr -x traj.xtc -g 32 -b 0 -e 10 -o quick_test
```

### Phase 2: Convergence Study
```bash
# Test 2-3 grid sizes on representative subset
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 100 -o conv_64
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 -o conv_128
./CuSAXS -s system.tpr -x traj.xtc -g 256 -b 0 -e 100 -o conv_256
```

### Phase 3: Scale Factor Optimization
```bash
# Fine-tune scaling with chosen grid size
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.0 -b 0 -e 200 -o scale_20
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.5 -b 0 -e 200 -o scale_25
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 3.0 -b 0 -e 200 -o scale_30
```

### Phase 4: Production Run
```bash
# Full calculation with optimized parameters
./CuSAXS -s system.tpr -x traj.xtc -g 128 --Scale 2.5 \
         --order 4 -b 0 -e 2000 -o final_production
```

## System-Specific Grid Recommendations

### Small Proteins (<100 residues)
```bash
Grid: 64³ or 128³
Scale: 2.5
Memory: 1-2 GB
Time: Minutes to 1 hour
```

### Medium Proteins (100-500 residues)
```bash
Grid: 128³
Scale: 2.5-3.0  
Memory: 2-4 GB
Time: 1-3 hours
```

### Large Proteins/Complexes (>500 residues)
```bash
Grid: 128³ or 256³
Scale: 2.5-3.0
Memory: 4-16 GB  
Time: 3-8 hours
```

### Membrane Systems
```bash
Grid: 256×256×128 (anisotropic)
Scale: 3.0
Memory: 8-12 GB
Time: 4-8 hours
```

## Grid Performance Tips

### 1. Start Small, Scale Up
Always test with small grids first to validate parameters.

### 2. Monitor GPU Memory
```bash
# Watch memory during calculation
watch -n 1 nvidia-smi
```

### 3. Use Appropriate Hardware
- **16 GB GPU**: Up to 256³ grids
- **8 GB GPU**: Up to 128³ grids  
- **4 GB GPU**: Up to 64³ grids

### 4. Consider Non-Cubic Grids
For anisotropic systems, non-cubic grids can save memory while maintaining accuracy.

### 5. Balance Grid and Order
Higher interpolation orders can compensate for smaller grid sizes.

## Next Steps

- [Performance Optimization](performance/) - System-wide optimization strategies
- [Command Line Interface](command-line/) - Complete grid parameter reference
- [Water Models Tutorial](water-models/) - Combine with proper corrections
- [Python Analysis](python-analysis/) - Analyze grid convergence effects