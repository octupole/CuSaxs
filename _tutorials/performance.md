---
title: Performance Optimization
layout: default
---

# Performance Optimization

Maximize GPU utilization and minimize calculation time with these optimization strategies.

## Understanding Performance Factors

### GPU Memory Usage
CuSAXS performance is primarily limited by GPU memory:

| Grid Size | Memory Usage | Atoms Supported | Performance |
|-----------|--------------|-----------------|-------------|
| 64³ | ~1GB | <10k | Fastest |
| 128³ | ~4GB | <20k | Good |
| 256³ | ~16GB | <50k | Slower |
| 512³ | ~64GB | <100k | Very slow |

### Computational Scaling
- **Grid size**: O(N³ log N) due to 3D FFT
- **Number of atoms**: Linear with proper optimization
- **Trajectory length**: Linear scaling

## GPU Memory Management

### Monitor GPU Usage
```bash
# Check GPU memory before starting
nvidia-smi

# Monitor during calculation
watch -n 1 nvidia-smi
```

### Free GPU Memory
```bash
# Kill other GPU processes
sudo nvidia-smi -c 3  # Compute mode exclusive
```

### Optimize Memory Usage
```bash
# Reduce grid size if memory limited
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 1000

# Use smaller frame batches
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 500
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 500 -e 1000
```

## Parameter Optimization

### Grid Size Selection Strategy

#### Step 1: Find Minimum Grid Size
```bash
# Test different grid sizes
./CuSAXS -s system.tpr -x traj.xtc -g 32 -b 0 -e 10 -o test_32
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 10 -o test_64
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 10 -o test_128

# Compare profiles to find convergence point
python pysrc/compareIqs.py test_32_profile.dat test_64_profile.dat
```

#### Step 2: System Size Guidelines
```bash
# Small proteins (<5k atoms)
-g 64

# Medium systems (5-20k atoms)
-g 128  

# Large complexes (20-50k atoms)
-g 256

# Very large systems (>50k atoms)
-g 512  # If you have >32GB GPU memory
```

### Interpolation Order Optimization

#### Performance vs Accuracy
```bash
# Fast but lower quality
--order 1    # ~2x faster than order 4
--order 2    # ~1.5x faster than order 4

# Recommended balance  
--order 4    # Best speed/quality ratio

# High accuracy but slow
--order 5    # ~1.5x slower than order 4
--order 6    # ~2x slower than order 4
```

#### Finding Optimal Order
```bash
# Test different orders on small subset
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 50 --order 2 -o test_ord2
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 50 --order 4 -o test_ord4
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 50 --order 6 -o test_ord6

# Compare quality difference
python pysrc/compareIqs.py test_ord2_profile.dat test_ord4_profile.dat
```

## Frame Processing Optimization

### Smart Frame Selection
```bash
# Skip frames to reduce calculation time
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 10000 --dt 5

# Process equilibrated portion only
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 1000 -e 5000

# Sample different time periods
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 1000 -e 2000 -o early
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 8000 -e 9000 -o late
```

### Trajectory Analysis Strategy
1. **Quick survey**: Process every 10th-20th frame
2. **Convergence test**: Check if results stabilize
3. **Production run**: Use optimal sampling from survey

## System-Specific Optimizations

### Water-Heavy Systems
```bash
# Use appropriate water model for better accuracy
--water tip4p    # Most common
--water spce     # Alternative

# Consider removing bulk water for speed
# (requires trajectory preprocessing)
```

### Membrane Systems
```bash
# Use non-cubic grids for anisotropic systems
-g 256 256 128   # xy-plane larger than z

# Optimize scaling for membrane thickness
--Scale 3.0      # For thick membranes
--Scale 2.0      # For thin membranes
```

### Ion Solutions
```bash
# Include ion corrections for accuracy
--na 20 --cl 20 --simulation npt

# But note: ion corrections add computational cost
```

## Hardware Optimization

### GPU Selection
- **RTX 3090/4090**: 24GB VRAM, excellent for large systems
- **RTX 3080/4080**: 10-16GB VRAM, good for medium systems  
- **RTX 3070/4070**: 8-12GB VRAM, suitable for small-medium systems
- **Tesla/Quadro**: Professional cards with more VRAM

### System Configuration
```bash
# Set GPU to performance mode
sudo nvidia-smi -pl 300  # Set power limit to maximum

# Ensure good cooling
nvidia-smi -q -d TEMPERATURE
```

## Benchmarking and Profiling

### Performance Testing
```bash
# Time your calculations
time ./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 100

# Check GPU utilization
nvidia-smi dmon -s u -i 0
```

### Performance Metrics
Monitor these values during calculations:
- **GPU Memory Usage**: Should be <90% of available
- **GPU Utilization**: Should be >80% during calculation
- **Power Draw**: Should be near maximum for your card
- **Temperature**: Should be <85°C

## Optimization Workflow

### 1. Baseline Test
```bash
# Start with safe, small calculation
./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 10 --order 2
```

### 2. Scale Up Systematically
```bash
# Increase grid size
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 10 --order 2

# Increase interpolation order
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 10 --order 4

# Increase frame count  
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 100 --order 4
```

### 3. Find Optimal Balance
- **Quality requirement**: What precision do you need?
- **Time constraint**: How fast must calculation complete?
- **Memory limitation**: What's your GPU memory limit?

## Performance Troubleshooting

### Common Issues

#### Slow Performance
```bash
# Check GPU is being used
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check for memory swapping
nvidia-smi -q -d MEMORY
```

#### Memory Errors
```bash
# Reduce grid size
-g 64

# Process fewer frames per run  
-e 500

# Close other GPU applications
sudo pkill -f cuda
```

#### Quality Issues
```bash
# Increase interpolation order
--order 6

# Increase grid size
-g 256

# Adjust scaling factor
--Scale 2.0
```

## Performance Summary

### Quick Optimization Checklist
- ✅ Start with grid size 64-128
- ✅ Use interpolation order 4
- ✅ Monitor GPU memory usage
- ✅ Process equilibrated trajectory portion
- ✅ Use frame skipping for long trajectories
- ✅ Include water model if present
- ✅ Test parameter convergence on small subset

### Production Run Strategy
1. **Optimize on small test**: Find best parameters
2. **Batch process**: Split large trajectories
3. **Monitor resources**: Watch memory and temperature
4. **Validate results**: Compare with experimental data

## Next Steps

- [Water Models Tutorial](/tutorials/water-models/) - System-specific corrections
- [Python Analysis](/tutorials/python-analysis/) - Process your optimized results
- [Command Line Reference](/api/command-line/) - Advanced parameter options