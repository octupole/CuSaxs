---
title: Troubleshooting Guide  
layout: default
---

# Troubleshooting Guide

Solutions to common CuSAXS problems and error messages.

## Installation Issues

### CUDA Not Found
**Error**: `CUDA runtime error: CUDA driver version is insufficient`
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# If missing, install CUDA toolkit
# Ubuntu:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify installation
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build Errors  
**Error**: `CMake could not find CUDA`
```bash
# Specify CUDA path explicitly
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
         -DCMAKE_BUILD_TYPE=Release

# Check CMake finds CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release --debug-find
```

**Error**: `undefined reference to cuFFT functions`
```bash
# Ensure CUDA libraries are linked
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_CUBLAS_LIBRARIES=/usr/local/cuda/lib64/libcublas.so \
         -DCUDA_CUFFT_LIBRARIES=/usr/local/cuda/lib64/libcufft.so
```

## Runtime Errors

### Memory Issues

#### CUDA Out of Memory
**Error**: `CUDA error: out of memory`
```bash
# Solution 1: Reduce grid size
./CuSAXS -s system.tpr -x trajectory.xtc -g 64 -b 0 -e 1000  # Instead of 128

# Solution 2: Free GPU memory
nvidia-smi  # Check what's using GPU
sudo nvidia-smi -c 3  # Set compute mode exclusive

# Solution 3: Process fewer frames
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 500  # Instead of 2000

# Solution 4: Use non-cubic grid
./CuSAXS -s system.tpr -x trajectory.xtc -g 160 128 96  # Same total points as 128³
```

#### Memory Monitoring
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv

# Monitor during calculation
watch -n 1 nvidia-smi
```

#### System RAM Issues
**Error**: `std::bad_alloc` or system hangs
```bash
# Check system RAM
free -h

# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Or reduce trajectory processing
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 --dt 5
```

### File Errors

#### Cannot Read Topology
**Error**: `Error reading topology file`
```bash
# Check file exists and permissions
ls -la system.tpr
file system.tpr  # Should show "GROMACS trajectory"

# Verify GROMACS version compatibility
gmx check -s system.tpr

# Regenerate if corrupted
gmx grompp -f system.mdp -c system.gro -p system.top -o new_system.tpr
```

#### Trajectory Read Errors
**Error**: `Cannot read trajectory frame`
```bash
# Check trajectory integrity
gmx check -f trajectory.xtc

# Verify topology matches trajectory  
gmx check -f trajectory.xtc -s system.tpr

# Fix trajectory issues
gmx trjconv -f trajectory.xtc -s system.tpr -o fixed_trajectory.xtc -pbc mol
```

#### Frame Range Errors
**Error**: `Frame X not found in trajectory`
```bash
# Check actual frame count
gmx check -f trajectory.xtc | grep frames

# Adjust frame range
gmx traj -f trajectory.xtc -s system.tpr  # Shows available frames
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 5000  # Adjust -e value
```

### CUDA Errors

#### GPU Device Issues  
**Error**: `No CUDA-capable device found`
```bash
# Check GPU visibility
nvidia-smi
lspci | grep -i nvidia

# Check CUDA device enumeration
deviceQuery  # From CUDA samples

# Set specific GPU (if multiple)
export CUDA_VISIBLE_DEVICES=0
```

#### CUDA Version Mismatch
**Error**: `CUDA runtime version mismatch`  
```bash
# Check versions
nvidia-smi  # Driver version
nvcc --version  # Toolkit version

# Toolkit version must be ≤ Driver version
# Update driver if needed:
sudo apt update
sudo apt install nvidia-driver-470  # Or latest version
```

#### Kernel Launch Errors
**Error**: `CUDA kernel launch failed`
```bash
# Usually indicates GPU overheating or hardware issue
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Check temperature < 85°C
# If overheating:
# 1. Clean GPU fans
# 2. Improve case ventilation  
# 3. Reduce GPU power limit:
sudo nvidia-smi -pl 200  # Reduce to 200W
```

## Calculation Problems

### Poor Results Quality

#### Noisy SAXS Profile
**Symptoms**: Jagged, noisy curve instead of smooth profile
```bash
# Solution 1: More frames
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 5000 \
         --dt 2  # Double frame count

# Solution 2: Higher interpolation order
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
         --order 6  # Instead of 4

# Solution 3: Better equilibration
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 2000 -e 5000  # Skip early frames
```

#### Unphysical Profile Shape
**Symptoms**: Negative intensities, strange oscillations
```bash
# Check grid convergence
./CuSAXS -s system.tpr -x trajectory.xtc -g 64 -b 0 -e 100 -o test_64
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 100 -o test_128
python pysrc/compareIqs.py test_64_profile.dat test_128_profile.dat

# Increase grid size if profiles differ significantly
./CuSAXS -s system.tpr -x trajectory.xtc -g 256 -b 0 -e 1000
```

#### Poor Experimental Agreement
**Symptoms**: Simulation profile doesn't match experiment
```bash
# Check water model
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
         --water tip3p -o test_tip3p
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
         --water tip4p -o test_tip4p

# Check ion corrections
gmx select -s system.tpr -select "resname NA"  # Count Na+
gmx select -s system.tpr -select "resname CL"  # Count Cl-
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 \
         --na 10 --cl 15 --water tip4p

# Try different trajectory region
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 5000 -e 8000  # Later part
```

### Performance Problems  

#### Slow Calculations
**Symptoms**: Hours for simple calculations, low GPU usage
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# If GPU usage <50%:
# 1. Check other processes using GPU
nvidia-smi

# 2. Verify CUDA installation
deviceQuery

# 3. Check CPU bottlenecks
htop  # Look for high CPU usage

# 4. Use smaller test first
./CuSAXS -s system.tpr -x trajectory.xtc -g 32 -b 0 -e 10  # Should be fast
```

#### Memory Bandwidth Issues
**Symptoms**: GPU memory usage high but slow progress
```bash
# Reduce memory pressure
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 -b 0 -e 1000 --dt 5  # Skip frames

# Use more efficient grid
./CuSAXS -s system.tpr -x trajectory.xtc -g 160 128 96  # Non-cubic, same points as 128³
```

### Parameter Issues

#### Wrong Grid Size Effects
**Too small**: Limited q-range, missing high-q features
```bash
# Increase grid size
./CuSAXS -s system.tpr -x trajectory.xtc -g 256 --Scale 3.0
```

**Too large**: Out of memory, very slow
```bash
# Reduce grid size or use non-cubic
./CuSAXS -s system.tpr -x trajectory.xtc -g 160 128 96
```

#### Interpolation Order Problems
**Too low**: Blocky, pixelated profiles
```bash
# Increase order
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 --order 6
```

**Too high**: Very slow, diminishing returns
```bash
# Reduce to optimal
./CuSAXS -s system.tpr -x trajectory.xtc -g 128 --order 4
```

## System-Specific Issues

### Large Systems

#### Very Large Proteins (>50k atoms)
```bash
# Strategy 1: Non-cubic grids
./CuSAXS -s large.tpr -x large.xtc -g 320 256 192 --Scale 2.5

# Strategy 2: Frame batching
./CuSAXS -s large.tpr -x large.xtc -g 256 -b 0 -e 1000 -o part1
./CuSAXS -s large.tpr -x large.xtc -g 256 -b 1000 -e 2000 -o part2
# Average results with Python

# Strategy 3: Reduce interpolation order
./CuSAXS -s large.tpr -x large.xtc -g 256 --order 2
```

### Membrane Systems

#### Anisotropic Box Issues
**Symptoms**: Poor sampling in membrane normal direction
```bash
# Use appropriate grid ratios
# For 15×15×10 nm membrane box:
./CuSAXS -s membrane.tpr -x membrane.xtc -g 256 256 128 --Scale 3.0
```

#### Lipid Bilayer Artifacts
**Symptoms**: Strange low-q oscillations
```bash
# Check PBC treatment
gmx trjconv -f membrane.xtc -s membrane.tpr -o membrane_pbc.xtc -pbc mol

# Use fixed membrane trajectory
./CuSAXS -s membrane.tpr -x membrane_pbc.xtc -g 256 256 128
```

## Diagnostic Tools

### Debug Mode
```bash
# Enable verbose output
./CuSAXS --verbose -s system.tpr -x trajectory.xtc -g 64 -b 0 -e 10

# Check intermediate outputs
./CuSAXS --debug -s system.tpr -x trajectory.xtc -g 64 -b 0 -e 10
```

### Quick System Check
```bash
#!/bin/bash
# system_check.sh - Comprehensive system validation

echo "=== CuSAXS System Check ==="

# CUDA check
echo "1. CUDA Installation:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "❌ nvcc not found"
fi

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv
else
    echo "❌ nvidia-smi not found"
fi

# File check
echo -e "\n2. Input Files:"
for file in "$@"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists ($(du -h "$file" | cut -f1))"
        if [[ "$file" == *.tpr ]]; then
            gmx check -s "$file" 2>/dev/null | grep -E "atoms|box"
        elif [[ "$file" == *.xtc ]]; then
            gmx check -f "$file" 2>/dev/null | grep -E "frames|time"
        fi
    else
        echo "❌ $file missing"
    fi
done

# Quick test
echo -e "\n3. Quick Test:"
if [ -f "$1" ] && [ -f "$2" ]; then
    timeout 60 ./CuSAXS -s "$1" -x "$2" -g 32 -b 0 -e 5 -o system_test
    if [ $? -eq 0 ]; then
        echo "✅ Basic calculation successful"
        rm -f system_test_*
    else
        echo "❌ Basic calculation failed"
    fi
else
    echo "Skipping test - missing files"
fi

echo "System check complete."
```

### Log File Analysis
```bash
# Check for common errors in logs
check_logs() {
    local logfile="$1"
    
    echo "=== Log Analysis: $logfile ==="
    
    # Memory errors
    grep -i "memory\|malloc\|allocation" "$logfile"
    
    # CUDA errors
    grep -i "cuda\|gpu" "$logfile"
    
    # File errors
    grep -i "file\|read\|write" "$logfile"
    
    # Performance metrics
    grep -i "time\|performance" "$logfile"
}

# Usage
check_logs protein_saxs_parameters.log
check_logs protein_saxs_timing.log
```

## Emergency Procedures

### System Unresponsive
```bash
# Kill CuSAXS processes
pkill -f CuSAXS

# Reset GPU
sudo nvidia-smi --gpu-reset

# Clear GPU memory
sudo nvidia-smi -r
```

### Corrupted Results
```bash
# Verify file integrity
for file in *_profile.dat; do
    if [ ! -s "$file" ]; then
        echo "Empty file: $file"
    fi
    
    # Check for NaN/Inf
    if grep -q "nan\|inf\|NaN\|Inf" "$file"; then
        echo "Invalid values in: $file"
    fi
done

# Re-run with conservative parameters
./CuSAXS -s system.tpr -x trajectory.xtc -g 64 --order 2 -b 0 -e 100
```

### Hardware Issues
```bash
# Check GPU health
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv -l 1

# Stress test GPU
# (Only if you suspect hardware problems)
gpu_burn 300  # Run for 5 minutes

# Check for hardware errors
dmesg | grep -i "gpu\|nvidia\|cuda"
```

## Getting Help

### Information to Collect
```bash
# System information
./CuSAXS --version
nvcc --version  
nvidia-smi
lsb_release -a

# Error logs
cat *_parameters.log
cat *_timing.log

# Command used
echo "Command: $0 $@"

# File information
ls -la *.tpr *.xtc
gmx check -s system.tpr
gmx check -f trajectory.xtc
```

### Minimal Test Case
```bash
# Create minimal failing example
./CuSAXS -s system.tpr -x trajectory.xtc -g 32 -b 0 -e 5 -o minimal_test

# If this fails, the issue is fundamental
# If this works, the issue is parameter-specific
```

### Community Support
- **GitHub Issues**: https://github.com/your-username/CuSAXS/issues
- **Include**: System info, error logs, minimal test case
- **Describe**: What you expected vs what happened
- **Provide**: Complete command line used

## Prevention Tips

### Best Practices
1. **Always test small first**: Use 32³ grid with 5-10 frames
2. **Monitor resources**: Watch GPU memory and temperature
3. **Validate inputs**: Check topology-trajectory compatibility  
4. **Document parameters**: Keep records of successful runs
5. **Regular backups**: Save important results immediately

### Parameter Safety
```bash
# Safe starting parameters for any system
./CuSAXS -s system.tpr -x trajectory.xtc \
         -g 64 -b 0 -e 100 --order 4 \
         --water tip4p --dt 5

# Scale up gradually:
# 64³ → 128³ → 256³
# 100 frames → 1000 frames → full trajectory
```

## Next Steps

- [Performance Optimization](performance/) - Prevent performance issues
- [Protein Workflow](protein-workflow/) - Complete working example
- [Grid Optimization](grid-optimization/) - Avoid grid-related problems
- [Command Line Interface](command-line/) - Parameter reference