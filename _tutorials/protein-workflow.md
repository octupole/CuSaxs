---
title: Protein in Solution Workflow
layout: default
---

# Protein in Solution Workflow

Complete step-by-step workflow for calculating SAXS profiles from protein molecular dynamics simulations.

## Overview

This tutorial covers the complete process:
1. **System preparation** and validation
2. **Parameter selection** and optimization  
3. **CuSAXS calculation** execution
4. **Results analysis** and validation
5. **Experimental comparison**

## Prerequisites

### Required Files
- ✅ **system.tpr**: GROMACS topology file
- ✅ **trajectory.xtc**: MD trajectory file  
- ✅ **experimental.dat**: Experimental SAXS profile (optional)

### System Requirements
- NVIDIA GPU with ≥4GB memory
- CUDA 11.0+ installation
- CuSAXS compiled and tested

## Step 1: System Analysis

### Check System Composition
```bash
# Examine topology file
gmx dump -s protein.tpr | head -50

# Key information to note:
# - Total number of atoms
# - Box dimensions  
# - Water model used
# - Ion types and counts
# - Protein residues
```

### Analyze Trajectory
```bash
# Get trajectory statistics
gmx check -f trajectory.xtc -s protein.tpr

# Check trajectory length and box size
gmx traj -f trajectory.xtc -s protein.tpr -ob box.xvg -b 0 -e 1000

# Output shows:
# - Total frames available
# - Time range covered
# - Box size variations (important for NPT)
```

### Example System Inspection
```bash
# Typical output for protein in water:
# Atoms: 45,123 total
#   - Protein: 2,547 atoms
#   - Water: 42,456 atoms (14,152 molecules)  
#   - Na+: 8 atoms
#   - Cl-: 12 atoms
# Box: ~8.5 × 8.5 × 8.5 nm
# Time: 0-100 ns, 10,001 frames
```

## Step 2: Initial Parameter Selection

### Choose Grid Size
Based on system size and available GPU memory:

```bash
# Small protein (<5k atoms, <8 nm box)
GRID_SIZE=64

# Medium protein (5-20k atoms, 8-12 nm box)  
GRID_SIZE=128

# Large protein/complex (>20k atoms, >12 nm box)
GRID_SIZE=256

echo "Selected grid size: $GRID_SIZE³"
```

### Identify Water Model
```bash
# Check force field files or simulation parameters
grep -i "tip\|spc" *.mdp *.top

# Common models:
# - TIP3P: tip3p
# - TIP4P: tip4p  
# - SPC/E: spce
WATER_MODEL="tip4p"  # Adjust based on your system
```

### Count Ions
```bash
# Count Na+ and Cl- ions
NA_COUNT=$(gmx select -s protein.tpr -select "resname NA" -on /dev/null 2>&1 | grep -c "1")
CL_COUNT=$(gmx select -s protein.tpr -select "resname CL" -on /dev/null 2>&1 | grep -c "1")

echo "Na+ ions: $NA_COUNT"
echo "Cl- ions: $CL_COUNT"
```

## Step 3: Quick Validation Test

### Run Short Test Calculation
```bash
# 5-minute validation test
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g 64 -b 0 -e 50 \
         --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
         --order 2 -o quick_test

# Check output files were created
ls -la quick_test_*

# Look for obvious problems in profile
head quick_test_profile.dat
tail quick_test_profile.dat
```

### Validate Test Results
```bash
# Profile should show:
# - Reasonable I(q) values (positive, decreasing)
# - No NaN or infinite values
# - Smooth curve without artifacts

# Check parameters log
cat quick_test_parameters.log | grep -A5 -B5 "Error\|Warning"
```

## Step 4: Grid Convergence Study

### Test Different Grid Sizes
```bash
# Test on representative trajectory subset
for grid in 64 128 256; do
    ./CuSAXS -s protein.tpr -x trajectory.xtc \
             -g $grid -b 0 -e 100 \
             --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
             --order 4 -o grid_$grid
    
    echo "Grid $grid³ completed"
done
```

### Compare Grid Convergence
```bash
# Compare successive grid sizes
python pysrc/compareIqs.py grid_64_profile.dat grid_128_profile.dat
python pysrc/compareIqs.py grid_128_profile.dat grid_256_profile.dat

# Look for:
# - < 5% difference indicates convergence
# - Higher q_max with larger grids
# - Smooth profiles without artifacts
```

### Select Optimal Grid
```bash
# Based on convergence analysis:
# - If 64³ vs 128³ differ by >5%: use 128³ minimum
# - If 128³ vs 256³ differ by <3%: 128³ is sufficient
# - Consider GPU memory limitations

OPTIMAL_GRID=128  # Typical choice for most proteins
```

## Step 5: Frame Sampling Optimization

### Analyze Trajectory Equilibration
```bash
# Check when system reaches equilibrium
gmx energy -f md.edr -s protein.tpr -o temperature.xvg << EOF
Temperature
EOF

# Look for stable temperature region
# Skip initial equilibration period
EQUILIBRATION_END=1000  # frames (adjust based on your system)
```

### Test Frame Sampling
```bash
# Test different frame intervals
for dt in 1 5 10; do
    ./CuSAXS -s protein.tpr -x trajectory.xtc \
             -g $OPTIMAL_GRID -b $EQUILIBRATION_END -e 2000 \
             --dt $dt --water $WATER_MODEL \
             --na $NA_COUNT --cl $CL_COUNT \
             --order 4 -o frames_dt$dt
done

# Compare sampling effects
python pysrc/compareIqs.py frames_dt1_profile.dat frames_dt5_profile.dat
```

## Step 6: Production Calculation

### Final Parameter Setup
```bash
# Optimized parameters from previous steps
FINAL_GRID=128
FINAL_ORDER=4
FINAL_DT=5
START_FRAME=1000
END_FRAME=10000

echo "Production parameters:"
echo "Grid: $FINAL_GRID³"
echo "Frames: $START_FRAME to $END_FRAME (every $FINAL_DT)"
echo "Water model: $WATER_MODEL"
echo "Ions: Na=$NA_COUNT, Cl=$CL_COUNT"
```

### Execute Production Run
```bash
# Full production calculation
time ./CuSAXS -s protein.tpr -x trajectory.xtc \
              -g $FINAL_GRID -b $START_FRAME -e $END_FRAME \
              --dt $FINAL_DT --order $FINAL_ORDER \
              --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
              --simulation npt -o protein_saxs_final

# Monitor progress and GPU usage
# Expected time: 1-4 hours for typical protein system
```

### Validate Production Results
```bash
# Check calculation completed successfully
if [ -f protein_saxs_final_profile.dat ]; then
    echo "✅ SAXS calculation completed successfully"
    
    # Basic profile validation
    lines=$(wc -l < protein_saxs_final_profile.dat)
    echo "Profile contains $lines q-points"
    
    # Check q-range
    q_min=$(head -2 protein_saxs_final_profile.dat | tail -1 | awk '{print $1}')
    q_max=$(tail -1 protein_saxs_final_profile.dat | awk '{print $1}')
    echo "q-range: $q_min to $q_max nm⁻¹"
else
    echo "❌ Calculation failed - check error logs"
    exit 1
fi
```

## Step 7: Results Analysis

### Basic Profile Analysis
```python
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Load SAXS profile
q, I = np.loadtxt('protein_saxs_final_profile.dat', unpack=True)

# Basic analysis
print(f"q-range: {q[0]:.3f} to {q[-1]:.3f} nm⁻¹")
print(f"I(0): {I[0]:.2e}")
print(f"Data points: {len(q)}")

# Guinier analysis (automated)
def guinier_analysis(q, I, q_min=0.02, q_max=0.08):
    mask = (q >= q_min) & (q <= q_max)
    if np.sum(mask) < 5:
        print("Warning: Insufficient points for Guinier analysis")
        return None, None
    
    q_gui = q[mask]
    ln_I = np.log(I[mask])
    
    # Linear fit: ln(I) = ln(I0) - Rg²q²/3
    coeffs = np.polyfit(q_gui**2, ln_I, 1)
    Rg = np.sqrt(-3 * coeffs[0])
    I0 = np.exp(coeffs[1])
    
    return Rg, I0

Rg, I0 = guinier_analysis(q, I)
if Rg:
    print(f"Guinier Rg: {Rg:.2f} nm")
    print(f"I(0) extrapolated: {I0:.2e}")

# Create basic plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Log-log plot
ax1.loglog(q, I, 'b-', linewidth=2)
ax1.set_xlabel('q (nm⁻¹)')
ax1.set_ylabel('I(q)')
ax1.set_title('SAXS Profile')
ax1.grid(True, alpha=0.3)

# Kratky plot
ax2.plot(q, I * q**2, 'r-', linewidth=2)
ax2.set_xlabel('q (nm⁻¹)')
ax2.set_ylabel('I(q) × q²')
ax2.set_title('Kratky Plot')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('protein_saxs_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Structural Parameters
```python
# Calculate additional parameters
def calculate_porod_volume(q, I, q_min=0.3):
    """Estimate molecular volume from Porod region"""
    mask = q >= q_min
    if np.sum(mask) < 10:
        return None
    
    # Porod volume: V = 2π²I(0)/Q_p
    # where Q_p = ∫ I(q)q² dq (Porod invariant)
    Q_p = np.trapz(I[mask] * q[mask]**2, q[mask])
    V_p = 2 * np.pi**2 * I[0] / Q_p
    return V_p

V_porod = calculate_porod_volume(q, I)
if V_porod:
    print(f"Porod volume: {V_porod:.1f} nm³")

# Estimate molecular weight (rough)
# MW ≈ V_porod × 0.83 (g/mol per nm³)
if V_porod:
    MW_est = V_porod * 0.83
    print(f"Estimated MW: {MW_est:.0f} kDa")
```

## Step 8: Experimental Comparison

### Load Experimental Data
```bash
# If you have experimental SAXS data
# Ensure it's in same format: q(nm⁻¹) I(q)

# Compare with simulation
python pysrc/compareIqs.py protein_saxs_final_profile.dat experimental.dat

# Look for:
# - Overall shape agreement
# - Similar Rg values
# - Reasonable intensity scaling
# - Good fit in low-q region
```

### Quantitative Comparison
```python
import numpy as np

# Load both datasets
q_sim, I_sim = np.loadtxt('protein_saxs_final_profile.dat', unpack=True)
q_exp, I_exp = np.loadtxt('experimental.dat', unpack=True)

# Interpolate simulation to experimental q-grid
I_sim_interp = np.interp(q_exp, q_sim, I_sim)

# Scale simulation to match experiment (if needed)
# Method 1: Scale by I(0)
scale_factor = I_exp[0] / I_sim_interp[0]
I_sim_scaled = I_sim_interp * scale_factor

# Calculate chi-squared
chi2 = np.sum((I_sim_scaled - I_exp)**2 / I_exp) / len(I_exp)
print(f"χ² per point: {chi2:.3f}")

# Good agreement: χ² < 2
# Reasonable agreement: χ² < 5  
# Poor agreement: χ² > 10
```

## Step 9: Troubleshooting Common Issues

### Issue 1: Poor Experimental Agreement
```bash
# Try different water model
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g $FINAL_GRID -b $START_FRAME -e $END_FRAME \
         --water spce --na $NA_COUNT --cl $CL_COUNT \
         -o protein_alt_water

# Try different ion counts  
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g $FINAL_GRID -b $START_FRAME -e $END_FRAME \
         --water $WATER_MODEL --na 0 --cl 8 \
         -o protein_alt_ions
```

### Issue 2: Noisy Profile
```bash
# Increase sampling (more frames)
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g $FINAL_GRID -b $START_FRAME -e $((END_FRAME*2)) \
         --dt 2 --water $WATER_MODEL \
         --na $NA_COUNT --cl $CL_COUNT \
         -o protein_more_frames

# Or higher interpolation order
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g $FINAL_GRID -b $START_FRAME -e $END_FRAME \
         --order 6 --water $WATER_MODEL \
         --na $NA_COUNT --cl $CL_COUNT \
         -o protein_high_order
```

### Issue 3: Limited q-Range
```bash
# Increase grid size or scaling
./CuSAXS -s protein.tpr -x trajectory.xtc \
         -g 256 -b $START_FRAME -e $END_FRAME \
         --Scale 3.5 --water $WATER_MODEL \
         --na $NA_COUNT --cl $CL_COUNT \
         -o protein_high_q
```

## Step 10: Documentation and Archiving

### Document Calculation
```bash
# Create calculation summary
cat > calculation_summary.txt << EOF
Protein SAXS Calculation Summary
===============================
Date: $(date)
System: protein.tpr + trajectory.xtc

Parameters:
- Grid: $FINAL_GRID³
- Frames: $START_FRAME to $END_FRAME (every $FINAL_DT)
- Water model: $WATER_MODEL
- Ions: Na+ = $NA_COUNT, Cl- = $CL_COUNT
- Interpolation order: $FINAL_ORDER

Results:
- Guinier Rg: ${Rg} nm
- q-range: ${q_min} to ${q_max} nm⁻¹
- Calculation time: $(date -d @$SECONDS -u +%H:%M:%S)

Files generated:
- protein_saxs_final_profile.dat
- protein_saxs_final_parameters.log
- protein_saxs_final_timing.log
EOF
```

### Archive Results
```bash
# Create results archive
mkdir -p results/$(date +%Y%m%d)_protein_saxs
cp protein_saxs_final_* results/$(date +%Y%m%d)_protein_saxs/
cp calculation_summary.txt results/$(date +%Y%m%d)_protein_saxs/
cp protein_saxs_analysis.png results/$(date +%Y%m%d)_protein_saxs/
```

## Complete Workflow Script

```bash
#!/bin/bash
# complete_protein_workflow.sh

# Configuration
SYSTEM="protein.tpr"
TRAJECTORY="trajectory.xtc"
WATER_MODEL="tip4p"
GRID_SIZE=128
START_FRAME=1000
END_FRAME=10000
FRAME_STEP=5

# Count ions automatically
NA_COUNT=$(gmx select -s $SYSTEM -select "resname NA" 2>/dev/null | grep -c "atom")
CL_COUNT=$(gmx select -s $SYSTEM -select "resname CL" 2>/dev/null | grep -c "atom")

echo "Starting protein SAXS workflow..."
echo "System: $SYSTEM"
echo "Trajectory: $TRAJECTORY"
echo "Ions: Na+=$NA_COUNT, Cl-=$CL_COUNT"

# Quick validation
echo "Step 1: Quick validation test..."
./CuSAXS -s $SYSTEM -x $TRAJECTORY -g 64 -b 0 -e 50 \
         --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
         --order 2 -o quick_test

# Grid convergence test
echo "Step 2: Grid convergence test..."
for grid in 64 128; do
    ./CuSAXS -s $SYSTEM -x $TRAJECTORY -g $grid -b 0 -e 200 \
             --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
             --order 4 -o grid_$grid
done

# Production run
echo "Step 3: Production calculation..."
time ./CuSAXS -s $SYSTEM -x $TRAJECTORY \
              -g $GRID_SIZE -b $START_FRAME -e $END_FRAME \
              --dt $FRAME_STEP --order 4 \
              --water $WATER_MODEL --na $NA_COUNT --cl $CL_COUNT \
              --simulation npt -o protein_final

echo "Workflow completed successfully!"
echo "Main result: protein_final_profile.dat"
```

## Next Steps

- [Python Analysis Tutorial](python-analysis/) - Advanced post-processing
- [Experimental Comparison](experimental-comparison/) - Detailed validation methods
- [Performance Optimization](performance/) - Speed up calculations
- [Water Models Tutorial](water-models/) - Refine water corrections