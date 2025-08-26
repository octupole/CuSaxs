/**
 * @file TrajectoryWrapper.h
 * @brief Optimized C++ wrapper for Python trajectory interface using pybind11
 * 
 * This header provides a clean C++ interface that efficiently manages
 * Python trajectory operations through pybind11, with improved error
 * handling, memory management, and performance optimizations.
 */

#ifndef TRAJECTORY_WRAPPER_H
#define TRAJECTORY_WRAPPER_H

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <iostream>

#pragma once

namespace py = pybind11;

/**
 * @brief Exception class for trajectory-related errors
 */
class TrajectoryError : public std::runtime_error {
public:
    explicit TrajectoryError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Structure to hold frame data efficiently
 */
struct FrameData {
    std::vector<std::vector<float>> coordinates;
    std::vector<std::vector<float>> box_dimensions;
    float time;
    int step;
    
    FrameData() : time(0.0f), step(0) {}
};

/**
 * @brief Structure to hold system information
 */
struct SystemInfo {
    int n_atoms;
    int n_frames;
    int n_molecules;
    int n_proteins;
    int n_waters;
    int n_ions;
    int n_others;
    std::vector<std::string> atom_types;
};

/**
 * @brief High-level C++ wrapper for Python trajectory interface
 * 
 * This class provides an efficient C++ interface to MDAnalysis-based
 * trajectory processing, with optimizations for SAXS calculations:
 * 
 * Features:
 * - RAII-based resource management
 * - Batch frame processing
 * - Error handling with detailed diagnostics
 * - Memory-efficient data transfer
 * - Performance monitoring
 */
class TrajectoryWrapper {
public:
    /**
     * @brief Constructor - initializes Python interface
     * 
     * @param tpr_file Path to GROMACS topology file
     * @param xtc_file Path to GROMACS trajectory file
     * @param use_optimized Whether to use optimized interface (default: true)
     * 
     * @throws TrajectoryError if initialization fails
     */
    explicit TrajectoryWrapper(const std::string& tpr_file, 
                              const std::string& xtc_file,
                              bool use_optimized = true);
    
    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~TrajectoryWrapper();
    
    // Disable copy constructor and assignment (resource management)
    TrajectoryWrapper(const TrajectoryWrapper&) = delete;
    TrajectoryWrapper& operator=(const TrajectoryWrapper&) = delete;
    
    // Enable move semantics
    TrajectoryWrapper(TrajectoryWrapper&& other) noexcept;
    TrajectoryWrapper& operator=(TrajectoryWrapper&& other) noexcept;
    
    /**
     * @brief Get atom type to index mapping
     * 
     * @return Map from element symbol to vector of atom indices
     * @throws TrajectoryError if not initialized
     */
    std::map<std::string, std::vector<int>> getAtomIndices();
    
    /**
     * @brief Get comprehensive system information
     * 
     * @return SystemInfo structure with molecular system details
     */
    SystemInfo getSystemInfo();
    
    /**
     * @brief Read a specific frame from trajectory
     * 
     * @param frame_number Frame index to read (0-based)
     * @return FrameData structure with coordinates, box, time, step
     * @throws TrajectoryError if frame reading fails
     */
    FrameData readFrame(int frame_number);
    
    /**
     * @brief Read multiple frames efficiently (batch processing)
     * 
     * @param frame_numbers Vector of frame indices to read
     * @return Vector of FrameData structures
     * @throws TrajectoryError if batch reading fails
     */
    std::vector<FrameData> readFrameBatch(const std::vector<int>& frame_numbers);
    
    /**
     * @brief Read frame range with step interval
     * 
     * @param start_frame First frame to read
     * @param end_frame Last frame to read  
     * @param step Frame interval (default: 1)
     * @return Vector of FrameData structures
     */
    std::vector<FrameData> readFrameRange(int start_frame, int end_frame, int step = 1);
    
    /**
     * @brief Validate trajectory integrity
     * 
     * @return Map of validation results
     */
    std::map<std::string, bool> validateTrajectory();
    
    /**
     * @brief Get performance statistics
     * 
     * @return Map with performance metrics
     */
    std::map<std::string, double> getPerformanceStats();
    
    /**
     * @brief Get current frame data (after readFrame call)
     * 
     * @return Reference to current frame data
     * @throws TrajectoryError if no frame loaded
     */
    const FrameData& getCurrentFrameData() const;
    
    /**
     * @brief Check if trajectory is properly initialized
     * 
     * @return true if ready for operations
     */
    bool isInitialized() const { return initialized_ && py_interface_; }
    
    /**
     * @brief Get total number of frames in trajectory
     * 
     * @return Number of frames
     */
    int getNumFrames() const { return num_frames_; }
    
    /**
     * @brief Get total number of atoms
     * 
     * @return Number of atoms
     */
    int getNumAtoms() const { return num_atoms_; }
    
    /**
     * @brief Explicit cleanup of Python resources
     * 
     * Called automatically in destructor, but can be called manually
     * for explicit resource management.
     */
    void cleanup();

private:
    // Private data members
    std::string tpr_file_;
    std::string xtc_file_;
    bool use_optimized_;
    bool initialized_;
    
    py::object py_interface_;  // Python trajectory interface
    FrameData current_frame_data_;
    
    // Cached system information
    int num_frames_;
    int num_atoms_;
    std::map<std::string, std::vector<int>> atom_indices_;
    bool atom_indices_cached_;
    
    // Performance tracking
    mutable std::chrono::high_resolution_clock::time_point last_operation_time_;
    mutable double total_read_time_;
    mutable int read_count_;
    
    /**
     * @brief Initialize Python interpreter and modules
     * 
     * @throws TrajectoryError if initialization fails
     */
    void initializePython();
    
    /**
     * @brief Convert Python dictionary to FrameData
     * 
     * @param py_dict Python dictionary with frame data
     * @return FrameData structure
     */
    FrameData convertPyDictToFrameData(const py::dict& py_dict);
    
    /**
     * @brief Handle Python exceptions and convert to C++ exceptions
     * 
     * @param operation_name Name of operation for error context
     */
    void handlePythonException(const std::string& operation_name);
    
    /**
     * @brief Update performance statistics
     * 
     * @param operation_duration Duration of last operation
     */
    void updatePerformanceStats(double operation_duration) const;
    
    /**
     * @brief Validate frame number
     * 
     * @param frame_number Frame number to validate
     * @throws TrajectoryError if frame number is invalid
     */
    void validateFrameNumber(int frame_number) const;
};

/**
 * @brief Factory function to create trajectory wrapper
 * 
 * @param tpr_file Path to topology file
 * @param xtc_file Path to trajectory file
 * @param use_optimized Whether to use optimized interface
 * @return Unique pointer to TrajectoryWrapper
 */
std::unique_ptr<TrajectoryWrapper> createTrajectoryWrapper(
    const std::string& tpr_file,
    const std::string& xtc_file,
    bool use_optimized = true
);

/**
 * @brief Quick trajectory analysis function
 * 
 * @param tpr_file Path to topology file
 * @param xtc_file Path to trajectory file
 * @return SystemInfo structure with analysis results
 */
SystemInfo analyzeTrajectory(const std::string& tpr_file, const std::string& xtc_file);

#endif // TRAJECTORY_WRAPPER_H