#include "OptionsData.h"
#include <QFileInfo>

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
    if (tprFile.isEmpty())
    {
        return "Topology file (.tpr) is required";
    }
    if (!QFileInfo::exists(tprFile))
    {
        return "Topology file does not exist";
    }

    if (xtcFile.isEmpty())
    {
        return "Trajectory file (.xtc) is required";
    }
    if (!QFileInfo::exists(xtcFile))
    {
        return "Trajectory file does not exist";
    }

    if (gridSize.isEmpty())
    {
        return "Grid size is required";
    }
    if (gridSize.size() != 1 && gridSize.size() != 3)
    {
        return "Grid size must have 1 or 3 values";
    }

    for (int val : gridSize)
    {
        if (val <= 0)
        {
            return "Grid size values must be positive";
        }
    }

    if (startFrame < 0)
    {
        return "Start frame must be non-negative";
    }
    if (endFrame < startFrame)
    {
        return "End frame must be greater than or equal to start frame";
    }

    if (frameInterval <= 0)
    {
        return "Frame interval must be positive";
    }
    if (bsplineOrder <= 0)
    {
        return "B-spline order must be positive";
    }
    if (scaleFactor <= 0)
    {
        return "Scale factor must be positive";
    }
    if (binSize <= 0)
    {
        return "Bin size must be positive";
    }
    if (qCutoff <= 0)
    {
        return "Q-cutoff must be positive";
    }

    if (!scaledGrid.isEmpty())
    {
        if (scaledGrid.size() != 1 && scaledGrid.size() != 3)
        {
            return "Scaled grid must have 1 or 3 values";
        }
        for (int val : scaledGrid)
        {
            if (val <= 0)
            {
                return "Scaled grid values must be positive";
            }
        }
    }

    if (simulationType != "npt" && simulationType != "nvt")
    {
        return "Simulation type must be 'npt' or 'nvt'";
    }

    return "";
}