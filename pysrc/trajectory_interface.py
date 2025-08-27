"""
Optimized trajectory interface for CuSAXS with improved pybind11 integration.
This module provides a clean, efficient interface between C++ and Python components.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import MDAnalysis as mda
import networkx as nx
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for frame-specific data to minimize Python-C++ data transfer."""

    coordinates: np.ndarray
    box_dimensions: np.ndarray
    time: float
    step: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pybind11 transfer."""
        return {
            "coordinates": self.coordinates.tolist(),
            "box_dimensions": self.box_dimensions.tolist(),
            "time": float(self.time),
            "step": int(self.step),
        }


class OptimizedTrajectoryInterface:
    """
    Optimized interface for trajectory analysis with efficient C++ integration.

    Key improvements:
    - Batch processing to reduce Python-C++ calls
    - Cached topology analysis
    - Better memory management
    - Comprehensive error handling
    """

    def __init__(self, tpr_file: str, xtc_file: str):
        """Initialize with trajectory files."""
        try:
            # Suppress MDAnalysis warnings for cleaner output
            warnings.filterwarnings(
                "ignore",
                message="No coordinate reader found for",
                category=UserWarning,
            )

            self.tpr_file: str = tpr_file
            self.xtc_file: str = xtc_file
            self.universe: mda.Universe = mda.Universe(tpr_file, xtc_file)

            # Will be populated during topology build
            self.graph: nx.Graph = nx.Graph()
            self.molecules: List[set[int]] = []
            self.protein_molecules: List[set[int]] = []
            self.water_molecules: List[set[int]] = []
            self.ion_molecules: List[set[int]] = []
            self.other_molecules: List[set[int]] = []

            # Lazy caches
            self._atom_indices: Optional[Dict[str, List[int]]] = None
            self._current_frame_data: Optional[FrameData] = None

            # Build molecular topology once
            self._build_topology()

            logger.info(
                "Initialized trajectory interface: %d atoms, %d frames",
                len(self.universe.atoms),
                len(self.universe.trajectory),
            )

        except Exception as e:
            logger.error("Failed to initialize trajectory interface: %s", e)
            raise

    def _build_topology(self) -> None:
        """Build molecular topology and atom classification."""
        try:
            start_time = time.time()

            # Build connectivity graph
            self.graph.clear()
            for bond in self.universe.bonds:
                self.graph.add_edge(bond.atoms[0].index, bond.atoms[1].index)

            # Find connected components (molecules)
            self.molecules = list(nx.connected_components(self.graph))

            # Classify molecules and build indices
            self._classify_molecules()
            self._build_atom_indices()

            elapsed = time.time() - start_time
            logger.info(
                "Topology analysis completed in %.2fs: %d molecules identified",
                elapsed,
                len(self.molecules),
            )

        except Exception as e:
            logger.error("Topology building failed: %s", e)
            raise

    def _classify_molecules(self) -> None:
        """Classify molecules into categories."""
        self.protein_molecules = []
        self.water_molecules = []
        self.ion_molecules = []
        self.other_molecules = []

        # Define selection strings for different molecule types
        selections = {
            "protein": self.universe.select_atoms("protein").residues,
            "water": self.universe.select_atoms(
                "resname TIP3 TIP4 SOL WAT OW"
            ).residues,
            "ions": self.universe.select_atoms("resname NA CL K CA MG ZN").residues,
        }

        for molecule in self.molecules:
            molecule_residues = set(self.universe.atoms[list(molecule)].residues)

            if molecule_residues & set(selections["protein"]):
                self.protein_molecules.append(molecule)
            elif molecule_residues & set(selections["water"]):
                self.water_molecules.append(molecule)
            elif molecule_residues & set(selections["ions"]):
                self.ion_molecules.append(molecule)
            else:
                self.other_molecules.append(molecule)

    @lru_cache(maxsize=1)
    def _build_atom_indices(self) -> Dict[str, List[int]]:
        """Build and cache atom type to index mapping."""
        # mypy: explicit type for empty dict literal
        atom_indices: dict[str, list[int]] = {}
        for atom in self.universe.atoms:
            element = str(atom.element)
            if element not in atom_indices:
                atom_indices[element] = []
            atom_indices[element].append(int(atom.index))

        self._atom_indices = atom_indices
        return atom_indices

    def get_atom_indices(self) -> Dict[str, List[int]]:
        """Get atom type to index mapping."""
        if self._atom_indices is None:
            self._build_atom_indices()
        # mypy: _atom_indices guaranteed set above
        return self._atom_indices  # type: ignore[return-value]

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "n_atoms": len(self.universe.atoms),
            "n_frames": len(self.universe.trajectory),
            "n_molecules": len(self.molecules),
            "n_proteins": len(self.protein_molecules),
            "n_waters": len(self.water_molecules),
            "n_ions": len(self.ion_molecules),
            "n_others": len(self.other_molecules),
            "atom_types": list(self.get_atom_indices().keys()),
        }

    def read_frame(self, frame_number: int) -> FrameData:
        """
        Read specific frame and return structured data.

        Args:
            frame_number: Frame index to read

        Returns:
            FrameData object with coordinates, box, time, step
        """
        try:
            nframes = len(self.universe.trajectory)
            if frame_number >= nframes or frame_number < 0:
                raise IndexError(
                    f"Frame {frame_number} out of range (0..{nframes - 1})"
                )

            # Read frame
            ts = self.universe.trajectory[frame_number]

            # Convert coordinates from Ångström to nm (factor 0.1)
            coordinates = self.universe.atoms.positions * 0.1  # (N, 3) float64

            # Get box dimensions and convert to nm
            box_dims = ts.triclinic_dimensions * 0.1  # (3, 3) float64

            # Create frame data (keep float64; CUDA conversion handled later)
            self._current_frame_data = FrameData(
                coordinates=coordinates,
                box_dimensions=box_dims,
                time=float(ts.time),
                step=int(ts.frame),
            )

            return self._current_frame_data

        except Exception as e:
            logger.error("Failed to read frame %d: %s", frame_number, e)
            raise

    def read_frame_batch(self, frame_numbers: List[int]) -> List[FrameData]:
        """
        Read multiple frames efficiently.

        Args:
            frame_numbers: List of frame indices

        Returns:
            List of FrameData objects
        """
        frames_data: list[FrameData] = []
        for frame_num in frame_numbers:
            frames_data.append(self.read_frame(frame_num))
        return frames_data

    def get_current_frame_dict(self) -> Dict[str, Any]:
        """Get current frame data as dictionary for pybind11 transfer."""
        if self._current_frame_data is None:
            raise RuntimeError("No frame data available. Call read_frame() first.")
        return self._current_frame_data.to_dict()

    def preprocess_coordinates_for_cuda(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Preprocess coordinates for CUDA kernel consumption.

        Args:
            coordinates: Raw coordinates from MDAnalysis

        Returns:
            Preprocessed coordinates suitable for CUDA
        """
        # Ensure contiguous memory layout
        if not coordinates.flags["C_CONTIGUOUS"]:
            coordinates = np.ascontiguousarray(coordinates)

        # Ensure float32 for GPU efficiency
        if coordinates.dtype != np.float32:
            coordinates = coordinates.astype(np.float32)

        return coordinates

    def validate_trajectory(self) -> Dict[str, bool]:
        """Validate trajectory integrity."""
        validation_results = {
            "topology_loaded": self.universe is not None,
            "trajectory_loaded": len(self.universe.trajectory) > 0,
            "bonds_present": len(self.universe.bonds) > 0,
            "atoms_classified": len(self._atom_indices or {}) > 0,
        }

        logger.info("Trajectory validation: %s", validation_results)
        return validation_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance-related statistics."""
        return {
            "memory_usage_mb": self.universe.trajectory.memory_usage(),
            "atoms_per_molecule": (
                len(self.universe.atoms) / len(self.molecules) if self.molecules else 0
            ),
            # Using totaltime as a rough proxy for "size"
            "trajectory_size_gb": self.universe.trajectory.totaltime / 1000.0,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "universe") and self.universe is not None:
                try:
                    self.universe.trajectory.close()
                except Exception:
                    # Some readers may not implement close()
                    pass
                del self.universe

            # Clear cached data
            self._atom_indices = None
            self._current_frame_data = None

            logger.info("Trajectory interface cleaned up successfully")

        except Exception as e:
            logger.warning("Cleanup warning: %s", e)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Avoid raising in destructor
            pass


# Backwards compatibility wrapper
class Topology(OptimizedTrajectoryInterface):
    """Legacy interface for backwards compatibility."""

    def __init__(self, tpr_file: str, xtc_file: str):
        super().__init__(tpr_file, xtc_file)
        logger.warning(
            "Using legacy Topology interface. Consider migrating to "
            "OptimizedTrajectoryInterface."
        )

    def get_atom_index(self) -> Dict[str, List[int]]:
        """Legacy method name."""
        return self.get_atom_indices()

    def get_coordinates(self):
        """Legacy method - returns coordinates as nested list."""
        if self._current_frame_data is None:
            raise RuntimeError("No frame loaded. Call read_frame() first.")
        return self._current_frame_data.coordinates.tolist()

    def get_box(self):
        """Legacy method - returns box dimensions as nested list."""
        if self._current_frame_data is None:
            raise RuntimeError("No frame loaded. Call read_frame() first.")
        return self._current_frame_data.box_dimensions.tolist()

    def get_time(self) -> float:
        """Legacy method - returns current frame time."""
        if self._current_frame_data is None:
            raise RuntimeError("No frame loaded. Call read_frame() first.")
        return float(self._current_frame_data.time)

    def get_step(self) -> int:
        """Legacy method - returns current frame step."""
        if self._current_frame_data is None:
            raise RuntimeError("No frame loaded. Call read_frame() first.")
        return int(self._current_frame_data.step)


# Factory function for creating interfaces
def create_trajectory_interface(
    tpr_file: str, xtc_file: str, legacy_mode: bool = False
) -> OptimizedTrajectoryInterface:
    """
    Factory function to create trajectory interfaces.

    Args:
        tpr_file: Path to topology file
        xtc_file: Path to trajectory file
        legacy_mode: Whether to use legacy interface

    Returns:
        Trajectory interface instance
    """
    if legacy_mode:
        return Topology(tpr_file, xtc_file)
    return OptimizedTrajectoryInterface(tpr_file, xtc_file)


# Module-level convenience functions
def analyze_trajectory(tpr_file: str, xtc_file: str) -> Dict[str, Any]:
    """Quick trajectory analysis function."""
    interface = OptimizedTrajectoryInterface(tpr_file, xtc_file)
    try:
        return interface.get_system_info()
    finally:
        interface.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python trajectory_interface.py <tpr_file> <xtc_file>")
        sys.exit(1)

    # Demo usage
    tpr_file, xtc_file = sys.argv[1], sys.argv[2]

    print("Analyzing trajectory...")
    info = analyze_trajectory(tpr_file, xtc_file)

    for key, value in info.items():
        print(f"{key}: {value}")
