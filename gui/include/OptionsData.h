#ifndef OPTIONSDATA_H
#define OPTIONSDATA_H

#include <QString>
#include <QVector>

// Data structure to hold all options - matches your Options class
struct OptionsData
{
    // Required options
    QString tprFile;
    QString xtcFile;
    QVector<int> gridSize; // nx, ny, nz
    int startFrame;
    int endFrame;

    // Optional options
    QString outputFile = "saxs.dat";
    int frameInterval = 1;    // dt
    int bsplineOrder = 4;     // order
    QVector<int> scaledGrid;  // nnx, nny, nnz
    float scaleFactor = 2.5f; // sigma
    float binSize = 0.05f;    // Dq
    float qCutoff = 4.0f;     // Qcut
    QString waterModel = "";  // empty means average padding
    int sodiumAtoms = 0;
    int chlorineAtoms = 0;
    QString simulationType = "npt"; // npt or nvt

    // Convert to command line arguments
    QStringList toCommandLineArgs() const;

    // Validate the data
    bool isValid() const;
    QString validationError() const;
};

#endif // OPTIONSDATA_H
