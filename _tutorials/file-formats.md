---
title: File Formats
layout: default
---

# File Formats Guide

Understanding GROMACS file formats and CuSAXS input/output requirements.

## Input File Formats

### GROMACS Topology Files (.tpr)

The `.tpr` file contains:
- **System topology**: Atom types, bonds, angles, dihedrals
- **Force field parameters**: Masses, charges, van der Waals parameters
- **Simulation parameters**: Box dimensions, periodic boundary conditions
- **Run parameters**: Temperature, pressure, constraints

#### Creating .tpr Files
```bash
# From GROMACS preprocessing
gmx grompp -f mdrun.mdp -c system.gro -p system.top -o system.tpr

# Check .tpr file contents
gmx check -f system.tpr
```

#### What CuSAXS Reads from .tpr
- Atomic coordinates and types
- Box dimensions
- Atomic form factors (derived from atom types)
- Molecular topology for proper grouping

### GROMACS Trajectory Files (.xtc)

The `.xtc` file contains:
- **Atomic positions** over time
- **Box dimensions** for each frame (if NPT)
- **Compressed coordinates** (lossy compression)
- **Time stamps** for each frame

#### Alternative Trajectory Formats
```bash
# Convert other formats to .xtc
gmx trjconv -f trajectory.trr -o trajectory.xtc  # From .trr
gmx trjconv -f trajectory.dcd -o trajectory.xtc  # From .dcd
```

#### Trajectory Quality Check
```bash
# Check trajectory integrity
gmx check -f trajectory.xtc

# Get trajectory statistics
gmx traj -f trajectory.xtc -s system.tpr
```

## File Compatibility

### Supported Input Combinations

| Topology | Trajectory | Status | Notes |
|----------|------------|---------|-------|
| .tpr | .xtc | ✅ Fully supported | Recommended |
| .tpr | .trr | ❌ Not supported | Convert to .xtc first |
| .gro + .top | .xtc | ❌ Not supported | Generate .tpr file |
| .pdb | .dcd | ❌ Not supported | Use GROMACS workflow |

### File Size Considerations

```bash
# Check file sizes
ls -lh system.tpr trajectory.xtc

# Typical sizes:
# .tpr: 1-50 MB (depends on system size)
# .xtc: 100 MB - 10 GB (depends on trajectory length)
```

## Output File Formats

### SAXS Profile (.dat)

Format: Two-column ASCII
```
# q (nm^-1)    I(q) (arbitrary units)
0.050000      1.234567e+03
0.055000      1.198765e+03
0.060000      1.145678e+03
...
```

#### Loading in Analysis Software
```python
# Python/NumPy
import numpy as np
q, I = np.loadtxt('saxs_profile.dat', unpack=True)

# R
data <- read.table('saxs_profile.dat', header=FALSE)
q <- data[,1]
I <- data[,2]
```

### Parameters Log (.log)

Human-readable parameter summary:
```
CuSAXS Calculation Parameters
=============================
Input files:
  Topology: system.tpr
  Trajectory: trajectory.xtc

Grid settings:
  Size: 128 x 128 x 128
  Scale factor: 2.5
  Interpolation order: 4

Frame processing:
  Begin: 0
  End: 1000
  Step: 1
  Total frames processed: 1001

System information:
  Total atoms: 12547
  Water model: tip4p
  Na+ ions: 10
  Cl- ions: 10
...
```

### Timing Log (.log)

Performance metrics:
```
CuSAXS Performance Report
========================
Total calculation time: 145.7 seconds

Phase breakdown:
  File loading: 12.3 s (8.4%)
  Grid interpolation: 89.2 s (61.2%)
  FFT calculations: 32.1 s (22.0%)
  Profile generation: 8.7 s (6.0%)
  File output: 3.4 s (2.3%)

GPU utilization:
  Memory used: 3.2 GB / 8.0 GB (40%)
  Average utilization: 85%
  Peak temperature: 67°C
...
```

### Histogram Data (.dat)

Radial distribution format:
```
# r (nm)       g(r)
0.100000      0.000000
0.105000      0.012345
0.110000      0.045678
...
```

## File Preparation Workflows

### Standard GROMACS Workflow

```bash
# 1. System preparation
gmx pdb2gmx -f protein.pdb -o protein.gro -p protein.top

# 2. Add solvent and ions
gmx editconf -f protein.gro -o protein_box.gro -c -d 1.0 -bt cubic
gmx solvate -cp protein_box.gro -cs spc216.gro -o protein_solv.gro -p protein.top
gmx grompp -f ions.mdp -c protein_solv.gro -p protein.top -o ions.tpr
gmx genion -s ions.tpr -o protein_ions.gro -p protein.top -pname NA -nname CL -neutral

# 3. Energy minimization and equilibration
gmx grompp -f minim.mdp -c protein_ions.gro -p protein.top -o em.tpr
gmx mdrun -v -deffnm em

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p protein.top -o nvt.tpr
gmx mdrun -deffnm nvt

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p protein.top -o npt.tpr
gmx mdrun -deffnm npt

# 4. Production MD
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p protein.top -o md.tpr
gmx mdrun -deffnm md

# 5. Trajectory conversion for CuSAXS
gmx trjconv -s md.tpr -f md.xtc -o trajectory_centered.xtc -center -pbc mol -ur compact
```

### File Validation

#### Check Topology-Trajectory Compatibility
```bash
# Verify files match
gmx check -f trajectory.xtc -s system.tpr

# Should report:
# - Same number of atoms
# - Compatible time ranges
# - No corruption detected
```

#### Common File Issues

| Problem | Symptoms | Solution |
|---------|-----------|----------|
| Mismatched atom count | CuSAXS crashes on load | Regenerate .tpr or fix trajectory |
| Corrupted trajectory | Read errors, missing frames | Use `gmx check` and `gmx trjconv -b -e` |
| Wrong periodicity | Unphysical SAXS profile | Check PBC treatment in trajectory |
| Missing water model | Incorrect scattering | Verify force field in .tpr |

### File Size Optimization

#### Trajectory Compression
```bash
# Reduce trajectory size
gmx trjconv -f input.xtc -o output.xtc -dt 10  # Keep every 10th frame
gmx trjconv -f input.xtc -o output.xtc -b 1000 -e 5000  # Time window

# Remove unnecessary atoms
gmx trjconv -f input.xtc -s system.tpr -o protein_only.xtc -n index.ndx
```

#### Storage Recommendations
- **Development**: Keep full precision trajectories
- **Production**: Use compressed .xtc format
- **Archive**: Compress with standard tools (.gz, .xz)

## Advanced File Handling

### Multiple Trajectory Segments
```bash
# Process trajectory segments separately
./CuSAXS -s system.tpr -x segment1.xtc -g 128 -b 0 -e 1000 -o seg1
./CuSAXS -s system.tpr -x segment2.xtc -g 128 -b 0 -e 1000 -o seg2

# Combine results
python pysrc/combine_profiles.py seg1_profile.dat seg2_profile.dat
```

### Custom Atom Groups
```bash
# Create index file for specific atoms
gmx make_ndx -f system.tpr -o custom.ndx

# Extract subset trajectory
gmx trjconv -s system.tpr -f full.xtc -n custom.ndx -o subset.xtc
```

### Format Conversion Scripts

#### From CHARMM/NAMD
```bash
# Convert CHARMM .dcd to GROMACS
gmx trjconv -f trajectory.dcd -s system.psf -o trajectory.xtc
```

#### From AMBER
```bash
# Convert AMBER trajectory
cpptraj << EOF
parm system.prmtop
trajin trajectory.nc
trajout trajectory.xtc
run
EOF
```

## Troubleshooting File Issues

### Common Error Messages

**"Cannot read topology file"**
- Check file path and permissions
- Verify .tpr file is not corrupted
- Ensure file was generated with compatible GROMACS version

**"Trajectory frame mismatch"** 
- Topology and trajectory have different atom counts
- Use same system.tpr for both MD and CuSAXS

**"Invalid box dimensions"**
- Box vectors are zero or invalid
- Check if simulation box is properly defined

**"Unknown atom type"**
- Force field parameters missing
- Verify .tpr contains all necessary atom types

### File Diagnostic Tools

```bash
# Comprehensive file check
gmx check -f trajectory.xtc -s system.tpr -vdwfac 0.8

# Atom type analysis
gmx dump -s system.tpr | grep atomtypes

# Box dimension check  
gmx traj -f trajectory.xtc -s system.tpr -ob box.xvg
```

## Best Practices

### File Organization
```
project/
├── input/
│   ├── system.tpr
│   └── trajectory.xtc
├── output/
│   ├── saxs_profile.dat
│   ├── parameters.log
│   └── timing.log
└── analysis/
    ├── experimental.dat
    └── comparison_plots.py
```

### Quality Control
1. **Always validate** topology-trajectory compatibility
2. **Check trajectory** for physical reasonableness  
3. **Monitor file sizes** - unexpectedly large files may indicate issues
4. **Keep backup copies** of original files
5. **Document** file origins and processing steps

## Next Steps

- [Quick Start Guide](quick-start/) - Basic usage with your files
- [Command Line Interface](command-line/) - File specification options
- [Performance Optimization](performance/) - Handling large files efficiently