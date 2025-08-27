---
title: Graphical User Interface Tutorial
layout: default
---

# GUI Tutorial

Learn to use the CuSAXS graphical interface for parameter exploration and easy calculations.

## When to Use the GUI

The GUI is **optional** and designed for:
- **First-time users** exploring parameter effects
- **Quick one-off calculations** with visual feedback  
- **Parameter discovery** before moving to command-line
- **Learning** the relationship between settings and performance

**Note**: For production runs and batch processing, use the command-line interface.

## Starting the GUI

```bash
./CuSAXS-gui
```

## Main Window Overview

### File Selection Panel
- **Topology File**: Browse for your `.tpr` file
- **Trajectory File**: Browse for your `.xtc` file  
- **Output Prefix**: Set custom output file names

### Basic Parameters Tab

#### Grid Settings
- **Grid Size**: Choose from dropdown or enter custom values
  - `64³` - Fast, good for testing
  - `128³` - Recommended for most systems  
  - `256³` - High resolution, needs more memory
  - **Custom**: Enter three values for non-cubic grids

#### Frame Selection
- **Begin Frame**: Starting frame number (usually 0)
- **End Frame**: Final frame to process
- **Frame Step (dt)**: Process every nth frame (1 = all frames)

#### Quick Settings
- **Interpolation Order**: Slider from 1 (fast) to 6 (accurate)
- **Water Model**: Dropdown with common models (tip3p, tip4p, etc.)

### Advanced Parameters Tab

#### Algorithm Settings
- **Grid Scale Factor**: Fine-tune grid spacing (default: 2.5)
- **Q-space Cutoff**: Maximum scattering vector
- **Bin Size**: Histogram resolution for output

#### System Corrections
- **Ion Count**: Number of Na⁺ and Cl⁻ ions
- **Simulation Type**: NPT or NVT ensemble

### Progress and Output Panel
- **Real-time progress bar** during calculations
- **Memory usage indicator** 
- **Estimated time remaining**
- **Log output** showing calculation steps

## Step-by-Step Workflow

### 1. Load Files
1. Click **Browse** next to "Topology File"
2. Select your `.tpr` file
3. Click **Browse** next to "Trajectory File"  
4. Select your `.xtc` file
5. Set output prefix (e.g., "protein_saxs")

### 2. Set Basic Parameters
1. **Grid Size**: Start with 128³ for most systems
2. **Frames**: Set begin=0, end=100 for initial test
3. **Order**: Leave at 4 (recommended)
4. **Water**: Select appropriate model if system contains water

### 3. Preview Settings
- **Parameter Summary** panel shows all current settings
- **Memory Estimate** indicates GPU memory requirements
- **Warning indicators** highlight potential issues

### 4. Run Calculation
1. Click **Start Calculation**
2. Monitor progress in real-time
3. Check memory usage doesn't exceed GPU capacity
4. Wait for completion message

### 5. View Results
- **Output files** created in working directory
- **Quick plot** button shows SAXS profile
- **Export parameters** saves settings for command-line use

## GUI Features

### Parameter Validation
- **Red indicators**: Invalid parameter combinations
- **Yellow warnings**: Suboptimal but functional settings
- **Green checkmarks**: Recommended parameter ranges

### Memory Management
- **Live memory estimate** updates as you change parameters
- **GPU memory bar** shows current usage
- **Automatic warnings** when approaching memory limits

### Parameter Export
- **Command-line generator**: Creates equivalent command  
- **Batch script export**: Generate scripts for multiple runs
- **Parameter file save**: Store settings for later use

## Common GUI Workflows

### Exploring Parameter Effects

1. **Load small test system**
2. **Start with minimal settings**: 64³ grid, 10 frames
3. **Systematically vary one parameter**:
   - Try different grid sizes
   - Compare interpolation orders
   - Test water model effects
4. **Note performance vs quality tradeoffs**

### Preparing Production Runs

1. **Use GUI to find optimal parameters**
2. **Click "Export Command"** 
3. **Copy generated command-line**
4. **Use command-line for actual production**

### Quick Analysis

1. **Load your system**
2. **Set reasonable parameters** (128³, order 4)
3. **Process representative frames** (e.g., 100-200)
4. **Use built-in plotting** for quick visualization
5. **Export data** for further analysis

## Tips for GUI Usage

### Performance Optimization
- **Start small**: Always test with 64³ grid first
- **Monitor memory**: Watch the memory indicator
- **Use frame skipping**: Set dt > 1 for long trajectories  
- **Close other GPU applications**: Free up memory

### Parameter Discovery
- **Grid size**: Start small, increase until quality plateaus
- **Interpolation order**: 4 is usually optimal balance
- **Water model**: Match your simulation's water type
- **Scaling factor**: 2.5-3.0 works for most systems

### Troubleshooting GUI Issues

| Problem | Solution |
|---------|----------|
| GUI won't start | Check Qt6 installation |
| File dialogs crash | Verify file permissions |
| Memory estimate wrong | Update GPU drivers |
| Slow parameter updates | Close background applications |
| Progress bar stuck | Check CUDA installation |

## Converting GUI Settings to Command-Line

The GUI generates equivalent command-line arguments:

**GUI Settings:**
- Grid: 128³
- Frames: 0-1000  
- Order: 4
- Water: tip4p

**Generated Command:**
```bash
./CuSAXS -s protein.tpr -x traj.xtc -g 128 -b 0 -e 1000 \
         --order 4 --water tip4p -o protein_saxs
```

## Next Steps

- [Command Line Interface](/tutorials/command-line/) - Move to production workflows
- [Performance Optimization](/tutorials/performance/) - Maximize efficiency  
- [Python Analysis](/tutorials/python-analysis/) - Process your results