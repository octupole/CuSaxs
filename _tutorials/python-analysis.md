---
title: Python Analysis Tools
layout: default
---

# Python Analysis Tools

Learn to use CuSAXS's built-in Python scripts for post-processing and analysis.

## Available Python Scripts

CuSAXS includes several analysis tools in the `pysrc/` directory:

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `compareIqs.py` | Compare SAXS profiles | Multiple .dat files | Plots + statistics |
| `fitBilayer.py` | Fit bilayer form factors | SAXS profile | Fitted parameters |
| `fitBilayer2.py` | Advanced bilayer fitting | SAXS profile | Enhanced fitting |
| `fitLamellar.py` | Fit lamellar structures | SAXS profile | Structural parameters |
| `topology.py` | Analyze system topology | .tpr file | System information |
| `trajectories.py` | Trajectory utilities | .xtc/.tpr files | Frame analysis |
| `trajectory_interface.py` | Python interface | Various | Programmatic access |

## Basic Usage Examples

### Comparing SAXS Profiles

#### Compare Two Profiles
```bash
# Compare simulation vs experiment
python pysrc/compareIqs.py simulation_profile.dat experimental.dat

# Compare different water models  
python pysrc/compareIqs.py tip3p_profile.dat tip4p_profile.dat
```

#### Compare Multiple Profiles
```bash
# Compare grid size effects
python pysrc/compareIqs.py \
    grid64_profile.dat \
    grid128_profile.dat \
    grid256_profile.dat
```

#### Output Analysis
```bash
# The script provides:
# - Overlay plots of all profiles
# - Chi-squared statistics  
# - Residual analysis
# - R-factor calculations
```

### System Topology Analysis

#### Basic Topology Info
```bash
# Get system composition
python pysrc/topology.py system.tpr

# Output includes:
# - Number of atoms by type
# - Molecular composition
# - Box dimensions
# - Force field information
```

#### Detailed Analysis
```bash
# Extended topology information
python pysrc/topology.py system.tpr --verbose

# Additional output:
# - Detailed atom types
# - Bond information  
# - Residue composition
# - Mass distribution
```

### Trajectory Analysis

#### Frame Statistics
```bash
# Analyze trajectory properties
python pysrc/trajectories.py system.tpr trajectory.xtc

# Provides:
# - Number of frames
# - Time range  
# - Box size variations
# - Atom count consistency
```

#### Frame Extraction
```bash
# Extract specific frames for analysis
python pysrc/trajectories.py system.tpr trajectory.xtc \
    --extract 100 200 500

# Creates individual .pdb files for specified frames
```

## Advanced Analysis Workflows

### Bilayer Structure Analysis

#### Basic Bilayer Fitting
```bash
# Fit simple bilayer model
python pysrc/fitBilayer.py membrane_saxs_profile.dat

# Output parameters:
# - Bilayer thickness
# - Electron density profile  
# - Head group separation
# - Tail region density
```

#### Advanced Bilayer Models
```bash
# Use sophisticated fitting
python pysrc/fitBilayer2.py membrane_saxs_profile.dat \
    --model complex --chains 2

# Additional features:
# - Multiple chain types
# - Asymmetric bilayers
# - Water penetration
# - Ion binding effects
```

### Lamellar Structure Analysis

#### Standard Lamellar Fitting
```bash
# Fit lamellar repeat structures
python pysrc/fitLamellar.py lamellar_saxs_profile.dat

# Determines:
# - Lamellar spacing
# - Form factor parameters
# - Structure factor effects
# - Correlation lengths
```

#### Multi-Component Systems
```bash
# Complex lamellar systems
python pysrc/fitLamellar.py complex_profile.dat \
    --components 3 --background auto

# Handles:
# - Multiple lamellar phases  
# - Mixed structural components
# - Background subtraction
# - Peak deconvolution
```

## Programmatic Interface

### Python API Usage

#### Import CuSAXS Interface
```python
# trajectory_interface.py provides Python API
from pysrc.trajectory_interface import CuSAXSTrajectory

# Load trajectory
traj = CuSAXSTrajectory("system.tpr", "trajectory.xtc")

# Access basic properties
print(f"Frames: {traj.n_frames}")
print(f"Atoms: {traj.n_atoms}")
print(f"Box: {traj.box_dimensions}")
```

#### Frame-by-Frame Analysis
```python
# Iterate through frames
for i, frame in enumerate(traj):
    positions = frame.positions
    box = frame.box
    time = frame.time
    
    # Custom analysis here
    com = calculate_center_of_mass(positions)
    print(f"Frame {i}: COM = {com}")
```

#### Custom SAXS Calculations
```python
# Access individual frames for custom processing
frame_data = traj.get_frame(100)

# Extract positions for specific atom types
protein_atoms = frame_data.select_atoms("protein")
water_atoms = frame_data.select_atoms("water")

# Custom scattering calculations
custom_profile = calculate_custom_saxs(protein_atoms, water_atoms)
```

### Batch Processing Scripts

#### Multiple System Analysis
```python
#!/usr/bin/env python3
# batch_analysis.py

import os
import glob
from pysrc.compareIqs import compare_profiles

# Find all SAXS profiles
profiles = glob.glob("*_profile.dat")

# Compare each with reference
reference = "experimental_reference.dat"

for profile in profiles:
    chi2 = compare_profiles(profile, reference)
    print(f"{profile}: χ² = {chi2:.3f}")
```

#### Parameter Sweep Analysis
```python
#!/usr/bin/env python3
# parameter_sweep.py

# Analyze parameter effects
parameters = ["grid64", "grid128", "grid256"]
orders = [2, 4, 6]

results = {}
for param in parameters:
    for order in orders:
        filename = f"{param}_order{order}_profile.dat"
        if os.path.exists(filename):
            results[f"{param}_ord{order}"] = analyze_profile(filename)

# Generate comparison report
generate_report(results)
```

## Custom Analysis Scripts

### Creating Your Own Scripts

#### Template Script Structure
```python
#!/usr/bin/env python3
"""
Custom CuSAXS analysis script template
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_saxs_profile(filename):
    """Load SAXS profile from CuSAXS output"""
    data = np.loadtxt(filename)
    q = data[:, 0]  # Scattering vector
    I = data[:, 1]  # Intensity
    return q, I

def analyze_profile(q, I):
    """Custom analysis function"""
    # Your analysis code here
    guinier_rg = calculate_guinier_radius(q, I)
    porod_volume = calculate_porod_volume(q, I)
    return guinier_rg, porod_volume

def main():
    parser = argparse.ArgumentParser(description='Custom SAXS analysis')
    parser.add_argument('profile', help='SAXS profile file')
    args = parser.parse_args()
    
    q, I = load_saxs_profile(args.profile)
    rg, volume = analyze_profile(q, I)
    
    print(f"Guinier Rg: {rg:.2f} nm")
    print(f"Porod Volume: {volume:.2f} nm³")

if __name__ == "__main__":
    main()
```

### Specialized Analysis Functions

#### Guinier Analysis
```python
def guinier_analysis(q, I, q_min=0.02, q_max=0.08):
    """Perform Guinier analysis to determine Rg"""
    # Select Guinier region
    mask = (q >= q_min) & (q <= q_max)
    q_gui = q[mask]
    ln_I = np.log(I[mask])
    
    # Linear fit: ln(I) = ln(I0) - (Rg²q²)/3
    coeffs = np.polyfit(q_gui**2, ln_I, 1)
    Rg = np.sqrt(-3 * coeffs[0])
    I0 = np.exp(coeffs[1])
    
    return Rg, I0
```

#### Kratky Plot Analysis  
```python
def kratky_plot(q, I):
    """Generate Kratky plot for folding analysis"""
    kratky_y = I * q**2
    
    plt.figure(figsize=(8, 6))
    plt.plot(q, kratky_y)
    plt.xlabel('q (nm⁻¹)')
    plt.ylabel('I(q) × q²')
    plt.title('Kratky Plot')
    plt.show()
    
    return q, kratky_y
```

## Data Visualization

### Enhanced Plotting Scripts

#### Multi-Profile Comparison Plot
```python
#!/usr/bin/env python3
"""Enhanced profile comparison with subplots"""

import matplotlib.pyplot as plt
import numpy as np

def enhanced_comparison_plot(profiles, labels, experimental=None):
    """Create comprehensive comparison plot"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Main comparison
    for profile, label in zip(profiles, labels):
        q, I = np.loadtxt(profile, unpack=True)
        ax1.loglog(q, I, label=label)
    
    if experimental:
        q_exp, I_exp = np.loadtxt(experimental, unpack=True)
        ax1.loglog(q_exp, I_exp, 'k-', label='Experiment', linewidth=2)
    
    ax1.set_xlabel('q (nm⁻¹)')
    ax1.set_ylabel('I(q)')
    ax1.legend()
    ax1.set_title('SAXS Profile Comparison')
    
    # Guinier plots, Kratky plots, etc.
    # ... additional subplot code ...
    
    plt.tight_layout()
    plt.show()
```

## Integration with Experimental Data

### Data Format Conversion
```python
def convert_experimental_data(exp_file, output_file):
    """Convert experimental SAXS data to CuSAXS format"""
    # Handle different experimental file formats
    # Convert units (Å⁻¹ to nm⁻¹, etc.)
    # Apply necessary corrections
    pass
```

### Chi-Squared Calculation
```python
def calculate_chi_squared(sim_file, exp_file, q_range=None):
    """Calculate chi-squared between simulation and experiment"""
    q_sim, I_sim = load_saxs_profile(sim_file)
    q_exp, I_exp = load_saxs_profile(exp_file)
    
    # Interpolate to common q-grid
    I_sim_interp = np.interp(q_exp, q_sim, I_sim)
    
    if q_range:
        mask = (q_exp >= q_range[0]) & (q_exp <= q_range[1])
        q_exp = q_exp[mask]
        I_exp = I_exp[mask]  
        I_sim_interp = I_sim_interp[mask]
    
    chi2 = np.sum((I_sim_interp - I_exp)**2 / I_exp)
    return chi2 / len(I_exp)
```

## Next Steps

- [Command Line Interface](command-line/) - Generate data for analysis
- [Water Models Tutorial](water-models/) - Optimize input parameters
- [Performance Optimization](performance/) - Efficient data generation
- [API Reference](../api/) - Complete parameter reference