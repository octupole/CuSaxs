/// <summary>
/// Provides a set of options and configuration parameters for the application.
/// </summary>
/// <remarks>
/// The `Options` class is a static class that holds various configuration parameters and options for the application. These include file paths, dimensions, constants, and other settings that are used throughout the codebase.
/// </remarks>
#ifndef OPTIONS_H
#define OPTIONS_H
#include <string>
#include <vector>
#include <map>
#pragma once
enum class padding
{
    avg,
    given
};

class Options
{
public:
    static std::string tpr_file, xtc_file;
    static int nx, ny, nz;
    static int nnx, nny, nnz;
    static float sigma, Dq, Qcut;
    static int order;
    static std::string Wmodel;
    static int Sodium, Chlorine;
    static padding myPadding;
    static std::map<std::string, float> myWmodel;
    static std::string outFile;
    static std::string Simulation;

private:
    Options() {};
    ~Options() {};
};

#endif