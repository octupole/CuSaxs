---
title: Water Models Tutorial
layout: default
---

# Water Models Tutorial

Learn how to apply proper water model corrections for accurate SAXS calculations.

## Why Water Models Matter

Water is typically the major component in biological SAXS experiments. Different water models used in MD simulations have:
- **Different electron densities**
- **Varying scattering contributions**  
- **Different structural properties**

Ignoring water model differences can lead to **systematic errors** in calculated SAXS profiles.

## Supported Water Models

CuSAXS supports these common water models:

| Model | Description | Typical Use | Command |
|-------|-------------|-------------|---------|
| `tip3p` | TIP3P - 3-site rigid | General purpose, fast | `--water tip3p` |
| `tip4p` | TIP4P - 4-site with virtual site | More accurate, common | `--water tip4p` |
| `tip4pew` | TIP4P-Ew - Ewald optimized | PME simulations | `--water tip4pew` |
| `tip5p` | TIP5P - 5-site model | High accuracy | `--water tip5p` |
| `spce` | SPC/E - 3-site flexible | Popular alternative | `--water spce` |
| `spc` | SPC - Simple Point Charge | Basic 3-site model | `--water spc` |

## Identifying Your Water Model

### Check Your Simulation Files

#### GROMACS `.mdp` files
```bash
grep -i water system.mdp
grep -i tip system.mdp
grep -i spc system.mdp
```

#### Topology files
```bash
# Check .top file for water model reference
grep -i "tip\|spc" system.top
```

#### GROMACS documentation
```bash
# Check force field documentation
ls /usr/share/gromacs/top/*/tip*
```

### Common Simulation Setups

| Force Field | Default Water | Alternative |
|-------------|---------------|-------------|
| AMBER | TIP3P | TIP4P-Ew |
| CHARMM | TIP3P | TIP4P |
| OPLS-AA | TIP3P/4P | SPC/E |
| GROMOS | SPC | SPC/E |

## Basic Water Model Usage

### Simple Protein in Water
```bash
# TIP3P water (most common)
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 \
         --water tip3p -o protein_tip3p

# TIP4P water (more accurate)
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 1000 \
         --water tip4p -o protein_tip4p
```

### Comparing Water Models
```bash
# Calculate with different models
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 500 \
         --water tip3p -o system_tip3p
         
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 500 \
         --water tip4p -o system_tip4p

# Compare results
python pysrc/compareIqs.py system_tip3p_profile.dat system_tip4p_profile.dat
```

## Advanced Water Corrections

### Water with Ions
```bash
# System with salt
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --water tip4p --na 10 --cl 10 -o protein_salt
```

### High Salt Concentrations
```bash
# High ionic strength
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --water spce --na 50 --cl 50 --simulation npt
```

## Water Model Effects on SAXS

### Electron Density Differences

| Water Model | Electrons per molecule | Relative density |
|-------------|------------------------|------------------|
| TIP3P | 10 | 1.00 |
| TIP4P | 10 | 0.998 |
| TIP5P | 10 | 1.002 |
| SPC/E | 10 | 0.997 |

### Scattering Impact

#### Low q-region (< 0.1 nm⁻¹)
- **Large differences** between water models
- **Forward scattering** most affected
- **Critical for** Guinier analysis

#### High q-region (> 0.5 nm⁻¹)
- **Smaller differences** between models  
- **Oscillations** may be shifted
- **Important for** detailed structure

## Validation and Testing

### Experimental Comparison
```bash
# Calculate profile with best water model
./CuSAXS -s protein.tpr -x protein.xtc -g 128 -b 0 -e 2000 \
         --water tip4p --order 4 -o protein_final

# Compare with experimental data
python pysrc/compareIqs.py protein_final_profile.dat experimental.dat
```

### Water Model Optimization Workflow

#### 1. Literature Research
- **Check original paper** for simulation details
- **Force field documentation** for recommended water
- **Similar studies** in your field

#### 2. Test Different Models
```bash
# Quick test with multiple models
for water in tip3p tip4p spce; do
    ./CuSAXS -s system.tpr -x traj.xtc -g 64 -b 0 -e 100 \
             --water $water -o test_$water
done
```

#### 3. Compare Against Experiment
```bash
# Calculate chi-squared for each model
python pysrc/compareIqs.py test_tip3p_profile.dat experiment.dat
python pysrc/compareIqs.py test_tip4p_profile.dat experiment.dat  
python pysrc/compareIqs.py test_spce_profile.dat experiment.dat
```

## Special Cases

### Mixed Water Systems
Some simulations use multiple water types:
```bash
# If your system has mixed waters, use dominant type
--water tip4p    # For majority TIP4P system
```

### Modified Water Models
For custom or modified water models:
```bash
# Use closest standard model
--water tip4p    # For modified TIP4P variants
--water spce     # For modified SPC variants
```

### No Water Corrections
For systems without water or unknown models:
```bash
# Omit --water flag entirely
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000
```

## Common Issues and Solutions

### Wrong Water Model
**Problem**: Poor agreement with experiment
```bash
# Try different water model
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --water tip4pew -o test_tip4pew
```

### Ion Concentration Mismatch
**Problem**: Low-q discrepancies
```bash
# Check actual ion numbers in system
gmx check -f system.tpr | grep -i ion

# Use correct ion counts
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --water tip4p --na 15 --cl 15
```

### Water Box Size Effects
**Problem**: Artificial periodicity effects
```bash
# Use appropriate grid scaling
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --water tip4p --Scale 3.5
```

## Best Practices

### 1. Always Use Water Corrections
```bash
# Good: Include water model
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000 --water tip4p

# Poor: Missing water corrections
./CuSAXS -s system.tpr -x traj.xtc -g 128 -b 0 -e 1000
```

### 2. Match Simulation Parameters
- **Same water model** as MD simulation
- **Same ion concentrations**
- **Same simulation conditions** (NPT vs NVT)

### 3. Validate Against Experiment
- **Compare multiple water models**
- **Check literature values**
- **Consider experimental conditions**

### 4. Document Your Choice
```bash
# Keep record of water model used
echo "Used TIP4P water model for system XYZ" >> calculation_log.txt
```

## Water Model Reference

### Quick Selection Guide

| Your Simulation | Recommended Flag |
|-----------------|------------------|
| AMBER force field | `--water tip3p` |
| CHARMM force field | `--water tip3p` |
| OPLS-AA force field | `--water tip4p` |
| GROMOS force field | `--water spce` |
| PME with Ewald | `--water tip4pew` |
| High accuracy needed | `--water tip5p` |
| Unknown/mixed | `--water tip4p` |

### Performance Impact
Water model corrections add minimal computational cost (~1-2% overhead).

## Next Steps

- [Ion Corrections Tutorial](/tutorials/ion-corrections/) - Handle ionic solutions
- [Python Analysis](/tutorials/python-analysis/) - Compare water model effects  
- [Performance Optimization](/tutorials/performance/) - Efficient water calculations