/**
 * @file TrajectoryWrapper.cpp
 * @brief Implementation of optimized C++ wrapper for Python trajectory interface
 */

#include "TrajectoryWrapper.h"
#include <algorithm>
#include <numeric>
#include <sstream>

// Constructor
TrajectoryWrapper::TrajectoryWrapper(const std::string& tpr_file, 
                                   const std::string& xtc_file,
                                   bool use_optimized)
    : tpr_file_(tpr_file)
    , xtc_file_(xtc_file)
    , use_optimized_(use_optimized)
    , initialized_(false)
    , num_frames_(0)
    , num_atoms_(0)
    , atom_indices_cached_(false)
    , total_read_time_(0.0)
    , read_count_(0)
{
    try {
        initializePython();
        
        // Get basic system info
        auto system_info = getSystemInfo();
        num_frames_ = system_info.n_frames;
        num_atoms_ = system_info.n_atoms;
        
        initialized_ = true;
        
        std::cout << "TrajectoryWrapper initialized successfully: " 
                  << num_atoms_ << " atoms, " << num_frames_ << " frames" << std::endl;
        
    } catch (const std::exception& e) {
        cleanup();
        throw TrajectoryError("Failed to initialize TrajectoryWrapper: " + std::string(e.what()));
    }
}

// Destructor
TrajectoryWrapper::~TrajectoryWrapper() {
    cleanup();
}

// Move constructor
TrajectoryWrapper::TrajectoryWrapper(TrajectoryWrapper&& other) noexcept
    : tpr_file_(std::move(other.tpr_file_))
    , xtc_file_(std::move(other.xtc_file_))
    , use_optimized_(other.use_optimized_)
    , initialized_(other.initialized_)
    , py_interface_(std::move(other.py_interface_))
    , current_frame_data_(std::move(other.current_frame_data_))
    , num_frames_(other.num_frames_)
    , num_atoms_(other.num_atoms_)
    , atom_indices_(std::move(other.atom_indices_))
    , atom_indices_cached_(other.atom_indices_cached_)
    , total_read_time_(other.total_read_time_)
    , read_count_(other.read_count_)
{
    // Reset other object
    other.initialized_ = false;
    other.num_frames_ = 0;
    other.num_atoms_ = 0;
    other.atom_indices_cached_ = false;
    other.total_read_time_ = 0.0;
    other.read_count_ = 0;
}

// Move assignment operator
TrajectoryWrapper& TrajectoryWrapper::operator=(TrajectoryWrapper&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        cleanup();
        
        // Move data
        tpr_file_ = std::move(other.tpr_file_);
        xtc_file_ = std::move(other.xtc_file_);
        use_optimized_ = other.use_optimized_;
        initialized_ = other.initialized_;
        py_interface_ = std::move(other.py_interface_);
        current_frame_data_ = std::move(other.current_frame_data_);
        num_frames_ = other.num_frames_;
        num_atoms_ = other.num_atoms_;
        atom_indices_ = std::move(other.atom_indices_);
        atom_indices_cached_ = other.atom_indices_cached_;
        total_read_time_ = other.total_read_time_;
        read_count_ = other.read_count_;
        
        // Reset other object
        other.initialized_ = false;
        other.num_frames_ = 0;
        other.num_atoms_ = 0;
        other.atom_indices_cached_ = false;
        other.total_read_time_ = 0.0;
        other.read_count_ = 0;
    }
    return *this;
}

void TrajectoryWrapper::initializePython() {
    try {
        // Import system module and add path
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(PY_SOURCE_DIR);
        
        // Import the trajectory interface module
        py::module_ traj_module = py::module_::import("trajectory_interface");
        
        // Create trajectory interface instance
        if (use_optimized_) {
            py::object interface_class = traj_module.attr("OptimizedTrajectoryInterface");
            py_interface_ = interface_class(tpr_file_, xtc_file_);
        } else {
            py::object interface_class = traj_module.attr("Topology");
            py_interface_ = interface_class(tpr_file_, xtc_file_);
        }
        
    } catch (const py::error_already_set& e) {
        handlePythonException("initializePython");
    }
}

std::map<std::string, std::vector<int>> TrajectoryWrapper::getAtomIndices() {
    if (!initialized_) {
        throw TrajectoryError("TrajectoryWrapper not initialized");
    }
    
    if (atom_indices_cached_) {
        return atom_indices_;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Call Python method to get atom indices
        py::dict py_indices;
        if (use_optimized_) {
            py_indices = py_interface_.attr("get_atom_indices")();
        } else {
            py_indices = py_interface_.attr("get_atom_index")();
        }
        
        // Convert to C++ map
        atom_indices_.clear();
        for (auto item : py_indices) {
            std::string key = py::str(item.first);
            std::vector<int> value = item.second.cast<std::vector<int>>();
            atom_indices_[key] = value;
        }
        
        atom_indices_cached_ = true;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        updatePerformanceStats(duration);
        
        return atom_indices_;
        
    } catch (const py::error_already_set& e) {
        handlePythonException("getAtomIndices");
        return {}; // Never reached due to exception
    }
}

SystemInfo TrajectoryWrapper::getSystemInfo() {
    if (!initialized_ && !py_interface_) {
        // Allow this call during initialization
        if (!py_interface_) {
            throw TrajectoryError("Python interface not available");
        }
    }
    
    try {
        py::dict py_info = py_interface_.attr("get_system_info")();
        
        SystemInfo info;
        info.n_atoms = py_info["n_atoms"].cast<int>();
        info.n_frames = py_info["n_frames"].cast<int>();
        info.n_molecules = py_info["n_molecules"].cast<int>();
        info.n_proteins = py_info["n_proteins"].cast<int>();
        info.n_waters = py_info["n_waters"].cast<int>();
        info.n_ions = py_info["n_ions"].cast<int>();
        info.n_others = py_info["n_others"].cast<int>();
        info.atom_types = py_info["atom_types"].cast<std::vector<std::string>>();
        
        return info;
        
    } catch (const py::error_already_set& e) {
        handlePythonException("getSystemInfo");
        return {}; // Never reached due to exception
    }
}

FrameData TrajectoryWrapper::readFrame(int frame_number) {
    if (!initialized_) {
        throw TrajectoryError("TrajectoryWrapper not initialized");
    }
    
    validateFrameNumber(frame_number);
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (use_optimized_) {
            // Use optimized interface that returns structured data
            py::object py_frame_data = py_interface_.attr("read_frame")(frame_number);
            py::dict frame_dict = py_interface_.attr("get_current_frame_dict")();
            current_frame_data_ = convertPyDictToFrameData(frame_dict);
        } else {
            // Use legacy interface
            py_interface_.attr("read_frame")(frame_number);
            
            current_frame_data_.coordinates = py_interface_.attr("get_coordinates")().cast<std::vector<std::vector<float>>>();
            current_frame_data_.box_dimensions = py_interface_.attr("get_box")().cast<std::vector<std::vector<float>>>();
            current_frame_data_.time = py_interface_.attr("get_time")().cast<float>();
            current_frame_data_.step = frame_number; // Legacy interface doesn't provide step
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        updatePerformanceStats(duration);
        read_count_++;
        
        return current_frame_data_;
        
    } catch (const py::error_already_set& e) {
        handlePythonException("readFrame");
        return {}; // Never reached due to exception
    }
}

std::vector<FrameData> TrajectoryWrapper::readFrameBatch(const std::vector<int>& frame_numbers) {
    if (!initialized_) {
        throw TrajectoryError("TrajectoryWrapper not initialized");
    }
    
    // Validate all frame numbers
    for (int frame_num : frame_numbers) {
        validateFrameNumber(frame_num);
    }
    
    std::vector<FrameData> frames_data;
    frames_data.reserve(frame_numbers.size());
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (use_optimized_) {
            // Use batch processing if available
            py::list py_frame_numbers = py::cast(frame_numbers);
            py::list py_frames_data = py_interface_.attr("read_frame_batch")(py_frame_numbers);
            
            for (auto py_frame : py_frames_data) {
                py::dict frame_dict = py_frame.attr("to_dict")();
                frames_data.push_back(convertPyDictToFrameData(frame_dict));
            }
        } else {
            // Fallback to individual frame reading
            for (int frame_num : frame_numbers) {
                frames_data.push_back(readFrame(frame_num));
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        updatePerformanceStats(duration);
        read_count_ += frame_numbers.size();
        
        return frames_data;
        
    } catch (const py::error_already_set& e) {
        handlePythonException("readFrameBatch");
        return {}; // Never reached due to exception
    }
}

std::vector<FrameData> TrajectoryWrapper::readFrameRange(int start_frame, int end_frame, int step) {
    if (step <= 0) {
        throw TrajectoryError("Frame step must be positive");
    }
    
    std::vector<int> frame_numbers;
    for (int frame = start_frame; frame <= end_frame; frame += step) {
        frame_numbers.push_back(frame);
    }
    
    return readFrameBatch(frame_numbers);
}

std::map<std::string, bool> TrajectoryWrapper::validateTrajectory() {
    if (!initialized_) {
        throw TrajectoryError("TrajectoryWrapper not initialized");
    }
    
    try {
        py::dict py_validation = py_interface_.attr("validate_trajectory")();
        
        std::map<std::string, bool> validation_results;
        for (auto item : py_validation) {
            std::string key = py::str(item.first);
            bool value = item.second.cast<bool>();
            validation_results[key] = value;
        }
        
        return validation_results;
        
    } catch (const py::error_already_set& e) {
        handlePythonException("validateTrajectory");
        return {}; // Never reached due to exception
    }
}

std::map<std::string, double> TrajectoryWrapper::getPerformanceStats() {
    std::map<std::string, double> stats;
    
    // C++ side statistics
    stats["total_read_time_cpp"] = total_read_time_;
    stats["read_count"] = static_cast<double>(read_count_);
    stats["avg_read_time_cpp"] = read_count_ > 0 ? total_read_time_ / read_count_ : 0.0;
    
    if (initialized_) {
        try {
            // Python side statistics
            py::dict py_stats = py_interface_.attr("get_performance_stats")();
            for (auto item : py_stats) {
                std::string key = py::str(item.first);
                double value = item.second.cast<double>();
                stats[key] = value;
            }
        } catch (const py::error_already_set& e) {
            // Don't throw for performance stats - just log warning
            std::cerr << "Warning: Could not get Python performance stats: " << e.what() << std::endl;
        }
    }
    
    return stats;
}

const FrameData& TrajectoryWrapper::getCurrentFrameData() const {
    if (!initialized_) {
        throw TrajectoryError("TrajectoryWrapper not initialized");
    }
    
    // Check if we have valid frame data
    if (current_frame_data_.coordinates.empty()) {
        throw TrajectoryError("No frame data available. Call readFrame() first.");
    }
    
    return current_frame_data_;
}

void TrajectoryWrapper::cleanup() {
    if (initialized_ && py_interface_) {
        try {
            py_interface_.attr("cleanup")();
        } catch (const std::exception& e) {
            std::cerr << "Warning during cleanup: " << e.what() << std::endl;
        }
    }
    
    py_interface_ = py::object(); // Release Python object
    initialized_ = false;
    atom_indices_cached_ = false;
    atom_indices_.clear();
    current_frame_data_ = FrameData(); // Reset frame data
}

FrameData TrajectoryWrapper::convertPyDictToFrameData(const py::dict& py_dict) {
    FrameData frame_data;
    
    frame_data.coordinates = py_dict["coordinates"].cast<std::vector<std::vector<float>>>();
    frame_data.box_dimensions = py_dict["box_dimensions"].cast<std::vector<std::vector<float>>>();
    frame_data.time = py_dict["time"].cast<float>();
    frame_data.step = py_dict["step"].cast<int>();
    
    return frame_data;
}

void TrajectoryWrapper::handlePythonException(const std::string& operation_name) {
    std::ostringstream error_msg;
    error_msg << "Python error in " << operation_name << ": ";
    
    try {
        throw; // Re-throw the current exception
    } catch (const py::error_already_set& e) {
        error_msg << e.what();
    } catch (const std::exception& e) {
        error_msg << e.what();
    }
    
    throw TrajectoryError(error_msg.str());
}

void TrajectoryWrapper::updatePerformanceStats(double operation_duration) const {
    total_read_time_ += operation_duration;
}

void TrajectoryWrapper::validateFrameNumber(int frame_number) const {
    if (frame_number < 0 || frame_number >= num_frames_) {
        std::ostringstream error_msg;
        error_msg << "Frame number " << frame_number << " out of range [0, " 
                  << num_frames_ - 1 << "]";
        throw TrajectoryError(error_msg.str());
    }
}

// Factory function
std::unique_ptr<TrajectoryWrapper> createTrajectoryWrapper(
    const std::string& tpr_file,
    const std::string& xtc_file,
    bool use_optimized) 
{
    return std::make_unique<TrajectoryWrapper>(tpr_file, xtc_file, use_optimized);
}

// Quick analysis function
SystemInfo analyzeTrajectory(const std::string& tpr_file, const std::string& xtc_file) {
    auto wrapper = createTrajectoryWrapper(tpr_file, xtc_file, true);
    return wrapper->getSystemInfo();
}