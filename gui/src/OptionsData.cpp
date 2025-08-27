#include "OptionsData.h"
#include <QFileInfo>
#include <QDir>

QStringList OptionsData::toCommandLineArgs() const
{
    QStringList args;

    // Required arguments
    args << "-s" << tprFile;
    args << "-x" << xtcFile;

    // Grid size
    args << "-g";
    for (int val : gridSize)
    {
        args << QString::number(val);
    }

    args << "-b" << QString::number(startFrame);
    args << "-e" << QString::number(endFrame);

    // Optional arguments
    if (!outputFile.isEmpty())
    {
        args << "-o" << outputFile;
    }

    if (frameInterval != 1)
    {
        args << "--dt" << QString::number(frameInterval);
    }

    if (bsplineOrder != 4)
    {
        args << "--order" << QString::number(bsplineOrder);
    }

    if (!scaledGrid.isEmpty())
    {
        args << "--gridS";
        for (int val : scaledGrid)
        {
            args << QString::number(val);
        }
    }

    if (scaleFactor != 2.5f)
    {
        args << "--Scale" << QString::number(scaleFactor);
    }

    if (binSize != 0.05f)
    {
        args << "--bin" << QString::number(binSize);
    }

    if (qCutoff != 4.0f)
    {
        args << "-q" << QString::number(qCutoff);
    }

    if (!waterModel.isEmpty())
    {
        args << "--water" << waterModel;
    }

    if (sodiumAtoms > 0)
    {
        args << "--na" << QString::number(sodiumAtoms);
    }

    if (chlorineAtoms > 0)
    {
        args << "--cl" << QString::number(chlorineAtoms);
    }

    if (simulationType != "npt")
    {
        args << "--simulation" << simulationType;
    }

    return args;
}

bool OptionsData::isValid() const
{
    // Check required files exist
    if (tprFile.isEmpty() || !QFileInfo::exists(tprFile))
    {
        return false;
    }

    if (xtcFile.isEmpty() || !QFileInfo::exists(xtcFile))
    {
        return false;
    }

    // Check grid size
    if (gridSize.isEmpty() || (gridSize.size() != 1 && gridSize.size() != 3))
    {
        return false;
    }

    for (int val : gridSize)
    {
        if (val <= 0)
            return false;
    }

    // Check frame range
    if (startFrame < 0 || endFrame < startFrame)
    {
        return false;
    }

    // Check optional parameters
    if (frameInterval <= 0)
        return false;
    if (bsplineOrder <= 0)
        return false;
    if (scaleFactor <= 0)
        return false;
    if (binSize <= 0)
        return false;
    if (qCutoff <= 0)
        return false;

    // Check scaled grid if provided
    if (!scaledGrid.isEmpty() && scaledGrid.size() != 1 && scaledGrid.size() != 3)
    {
        return false;
    }

    for (int val : scaledGrid)
    {
        if (val <= 0)
            return false;
    }

    // Check simulation type
    if (simulationType != "npt" && simulationType != "nvt")
    {
        return false;
    }

    return true;
}

QString OptionsData::validationError() const
{
    // Check file existence and readability
    QFileInfo tprInfo(tprFile);
    if (tprFile.isEmpty())
        return "Topology file (.tpr) is required";
    if (!tprInfo.exists())
        return "Topology file does not exist: " + tprFile;
    if (!tprInfo.isReadable())
        return "Topology file is not readable: " + tprFile;
    if (!tprFile.endsWith(".tpr", Qt::CaseInsensitive))
        return "Topology file must have .tpr extension";
    
    QFileInfo xtcInfo(xtcFile);
    if (xtcFile.isEmpty())
        return "Trajectory file (.xtc) is required";
    if (!xtcInfo.exists())
        return "Trajectory file does not exist: " + xtcFile;
    if (!xtcInfo.isReadable())
        return "Trajectory file is not readable: " + xtcFile;
    if (!xtcFile.endsWith(".xtc", Qt::CaseInsensitive))
        return "Trajectory file must have .xtc extension";
    
    // Check output file directory
    if (outputFile.isEmpty())
        return "Output file is required";
    
    QFileInfo outputInfo(outputFile);
    QDir outputDir = outputInfo.absoluteDir();
    if (!outputDir.exists())
        return "Output directory does not exist: " + outputDir.absolutePath();
    if (!outputDir.isReadable())
        return "Output directory is not writable: " + outputDir.absolutePath();
    
    // Validate grid size
    if (gridSize.isEmpty())
        return "Grid size is required";
    
    if (gridSize.size() != 1 && gridSize.size() != 3)
        return "Grid size must have 1 or 3 dimensions";
    
    for (int size : gridSize)
    {
        if (size <= 0)
            return "Grid dimensions must be positive";
        if (size < 8)
            return "Grid dimensions too small (minimum 8)";
        if (size > 512)
            return "Grid dimensions too large (maximum 512)";
    }
    
    // Validate frame range
    if (endFrame <= startFrame)
        return "End frame must be greater than start frame";
    
    if (startFrame < 0)
        return "Start frame cannot be negative";
    
    if (frameInterval <= 0)
        return "Frame interval must be positive";
    
    // Validate processing parameters
    if (bsplineOrder < 1 || bsplineOrder > 6)
        return "B-spline order must be between 1 and 6";
    
    if (scaleFactor <= 1.0)
        return "Scale factor must be greater than 1.0";
    if (scaleFactor > 10.0)
        return "Scale factor too large (maximum 10.0)";
    
    if (binSize <= 0)
        return "Bin size must be positive";
    if (binSize > 1.0)
        return "Bin size too large (maximum 1.0 Å⁻¹)";
    
    if (qCutoff <= 0)
        return "Q cutoff must be positive";
    if (qCutoff > 20.0)
        return "Q cutoff too large (maximum 20.0 Å⁻¹)";
    
    // Validate scaled grid if specified
    if (!scaledGrid.isEmpty())
    {
        if (scaledGrid.size() != gridSize.size())
            return "Scaled grid must have same number of dimensions as base grid";
        
        for (int i = 0; i < scaledGrid.size(); ++i)
        {
            if (scaledGrid[i] <= gridSize[i])
                return QString("Scaled grid dimension %1 (%2) must be larger than base grid (%3)")
                    .arg(i+1).arg(scaledGrid[i]).arg(gridSize[i]);
        }
    }
    
    // Check ion counts are non-negative
    if (sodiumAtoms < 0)
        return "Sodium atom count cannot be negative";
    if (chlorineAtoms < 0)
        return "Chlorine atom count cannot be negative";
    
    // Check simulation type
    if (simulationType != "npt" && simulationType != "nvt")
        return "Simulation type must be 'npt' or 'nvt'";
    
    return QString();
}

QStringList OptionsData::validationWarnings() const
{
    QStringList warnings;
    
    // Check grid sizes for FFT efficiency
    for (int size : gridSize)
    {
        if ((size & (size - 1)) != 0) // Not a power of 2
        {
            warnings << QString("Grid size %1 is not a power of 2. Powers of 2 provide better FFT performance.").arg(size);
        }
    }
    
    // Check memory usage
    double memoryGB = estimateMemoryUsage();
    if (memoryGB > 8.0)
        warnings << QString("Estimated GPU memory usage: %.1f GB. Ensure sufficient GPU memory.").arg(memoryGB);
    else if (memoryGB > 4.0)
        warnings << QString("Estimated GPU memory usage: %.1f GB").arg(memoryGB);
    
    // Check frame count
    int frameCount = endFrame - startFrame + 1;
    int processedFrames = (frameCount + frameInterval - 1) / frameInterval;
    
    if (processedFrames > 10000)
        warnings << QString("Will process %1 frames. Consider increasing frame interval for faster computation.").arg(processedFrames);
    else if (processedFrames > 5000)
        warnings << QString("Will process %1 frames. This may take significant time.").arg(processedFrames);
    
    // Check bin size vs q cutoff
    int numBins = static_cast<int>(qCutoff / binSize);
    if (numBins > 1000)
        warnings << QString("Large number of q-bins (%1). Consider increasing bin size.").arg(numBins);
    
    // Check B-spline order vs grid size
    int minGridSize = *std::min_element(gridSize.begin(), gridSize.end());
    if (bsplineOrder > minGridSize / 4)
        warnings << "High B-spline order relative to grid size may cause artifacts";
    
    // File size warnings
    QFileInfo xtcInfo(xtcFile);
    if (xtcInfo.exists() && xtcInfo.size() > 10LL * 1024 * 1024 * 1024) // 10GB
        warnings << "Large trajectory file detected. Processing may be slow.";
    
    return warnings;
}

double OptionsData::estimateMemoryUsage() const
{
    if (gridSize.isEmpty()) return 0.0;
    
    // Calculate effective grid size (use scaled grid if available)
    QVector<int> effectiveGrid = scaledGrid.isEmpty() ? gridSize : scaledGrid;
    
    // Ensure we have 3D dimensions
    while (effectiveGrid.size() < 3) {
        effectiveGrid.append(effectiveGrid.last());
    }
    
    qint64 totalCells = static_cast<qint64>(effectiveGrid[0]) * effectiveGrid[1] * effectiveGrid[2];
    
    // Memory components:
    // - 4 float grids (original + 3 working grids): 4 * totalCells * 4 bytes
    // - 2 complex grids for FFT: 2 * totalCells * 8 bytes  
    // - Form factors and misc: ~10 MB overhead
    
    double gridMemory = totalCells * (4 * 4 + 2 * 8) / 1e9; // Convert to GB
    double overhead = 0.01; // 10 MB
    
    return gridMemory + overhead;
}