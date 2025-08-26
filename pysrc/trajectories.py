#!/usr/bin/env python
import argparse
import MDAnalysis as mda
import networkx as nx
import sys
import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter
import time

class PDBWriterWithTitle(PDBWriter):
    def __init__(self, filename, n_atoms, title=""):
        super().__init__(filename, n_atoms)
        self.title = title

    def _write_header(self):
        """Override to include a custom title."""
        if self.title:
            self.outfile.write(f"TITLE     {self.title}\n")
        super()._write_header()
        
class TrajectoryStructures:
    def __init__(self, tpr_file, xtc_file):
        """
        Initializes the MoleculeCounter with the given TPR and XTC files.
        """
        self.tpr_file = tpr_file
        self.xtc_file = xtc_file
        self.universe = mda.Universe(tpr_file, xtc_file)
        self.G = nx.Graph()
        self.molecules = []
        self.protein_molecules = []
        self.water_molecules = []
        self.ion_molecules = []
        self.other_molecules = []
        self.__build_graph()
        self.__classify_molecules()
        
        print("Initialized TrajectoryStructures")

    def __build_graph(self):
        """
        Builds a graph where atoms are nodes and bonds are edges.
        """
        add_edge = self.G.add_edge  # Local function reference for speed

        # Add edges based on bonds
        for bond in self.universe.bonds:
            add_edge(bond.atoms[0].index, bond.atoms[1].index)

        # Find all connected components (molecules)
        self.molecules = list(nx.connected_components(self.G))

    def __classify_molecules(self):
        """
        Classifies the molecules into proteins, waters, ions, and others.
        """
        # Select residues for different types of molecules
        protein_residues = set(self.universe.select_atoms('protein').residues)
        water_residues = set(self.universe.select_atoms('resname TIP3 TIP4 SOL WAT').residues)
        ion_residues = set(self.universe.select_atoms('resname NA CL K CA MG').residues)

        # Classify each molecule based on its residues
        for molecule in self.molecules:
            molecule_residues = set(self.universe.atoms[list(molecule)].residues)
            if molecule_residues & protein_residues:
                self.protein_molecules.append(molecule)
            elif molecule_residues & water_residues:
                self.water_molecules.append(molecule)
            elif molecule_residues & ion_residues:
                self.ion_molecules.append(molecule)
            else:
                self.other_molecules.append(molecule)
        self.protein_atoms = self.universe.atoms[list(set().union(*self.protein_molecules))]
        
    def count_molecules(self):
        """
        Counts the total number of molecules and categorizes them.
        """

        num_molecules = len(self.molecules)
        num_proteins = len(self.protein_molecules)
        num_waters = len(self.water_molecules)
        num_ions = len(self.ion_molecules)
        num_others = len(self.other_molecules)

        return num_molecules, num_proteins, num_waters, num_ions, num_others

    def generate_molecule_dict(self):
        """
        Generates a dictionary with detailed information about each molecule.
        """
        molecule_dict = {
            'proteins': {},
            'waters': {},
            'ions': {},
            'others': {}
        }

        # Populate the dictionary with molecules
        for i, molecule in enumerate(self.protein_molecules):
            molecule_dict['proteins'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.water_molecules):
            molecule_dict['waters'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.ion_molecules):
            molecule_dict['ions'][f'{i}'] = list(molecule)

        for i, molecule in enumerate(self.other_molecules):
            molecule_dict['others'][f'{i}'] = list(molecule)

        return molecule_dict
 
    def get_centered_frame(self, frame):
#        self.center_on_protein_barycenter(frame)
        return self.get_atom_coordinates(frame)

    def get_dimensions_frame(self, frame):
        self.universe.trajectory[frame]
        return self.universe.dimensions
    
    def center_on_protein_barycenter(self, frame):
        """
        Centers the structure on the barycenter of the protein molecules and applies periodic boundary conditions.
        """
        # Get the coordinates of all atoms in the first frame
        self.universe.trajectory[frame]
        self.frame_time=self.universe.trajectory[frame].time


        # Compute the center of mass of the protein molecules

        protein_center_of_mass = self.protein_atoms.center_of_mass()

        # Calculate the middle of the box
        box = self.universe.dimensions[:3]  # Box dimensions
        box_center = box / 2.0

        # Translate the entire structure to center the protein at the middle of the box
        self.universe.atoms.translate(box_center - protein_center_of_mass)

        # Apply periodic boundary conditions to the molecule center of mass
        self.universe.atoms.wrap(compound='atoms')

    def get_atom_index(self):
        atom_index={}
        for atom in self.universe.atoms:
            atom_type = atom.element
            if atom_type not in atom_index:
                atom_index[atom_type] = []
            atoms2=atom.index
            atom_index[atom_type].append(atoms2)
        return atom_index            

    def get_time(self, frame):
        return self.universe.trajectory[frame].time
    
    def get_atom_coordinates(self, frame):
        """
        Returns the coordinates for each atom type found in the specified frame in dictionary format.
        """
        self.universe.trajectory[frame]  # Set to the specified frame      
        atom_coords = []
            
        atom_coords=self.universe.atoms.positions
        return atom_coords
    
    def write_pdb(self, frame, output_filename):
        """
        Writes the centered structure to a PDB file.
        """
        self.center_on_protein_barycenter(frame)
        print(f'Frame {frame} of {len(self.universe.trajectory)}')  

    # Add a custom title to the PDB file
        custom_title = f"Time {self.frame_time} fs of the simulation"
        self.universe.atoms.write(output_filename, remarks=custom_title)
        




def main():
    """
    Main function to parse arguments and initiate molecule counting.
    """
    parser = argparse.ArgumentParser(description='Read TPR and XTC files and count the number of molecules in the system.')
    parser.add_argument('-s', '--tpr', type=str, help='Path to the TPR file')
    parser.add_argument('-x', '--xtc', type=str, help='Path to the XTC file')
    parser.add_argument('-o', '--output', type=str, help='Output PDB file', default='centered_structure.pdb')

    args = parser.parse_args()
    
    # Check if both arguments are provided
    if not args.tpr or not args.xtc:
        parser.error("Both TPR file and XTC file must be provided.")
        sys.exit(1)

    # Initialize MoleculeCounter and count molecules
    traj = TrajectoryStructures(args.tpr, args.xtc)
    # traj.count_molecules()

    # # Print the results
    # print(f'Total number of molecules: {num_molecules}')
    # print(f'Number of protein molecules: {num_proteins}')
    # print(f'Number of water molecules: {num_waters}')
    # print(f'Number of ion molecules: {num_ions}')
    # if num_others:
    #     print(f'Number of other molecules: {num_others}')

    # # Generate and print the molecule dictionary
    # molecule_dict = traj.generate_molecule_dict()
    # for molecule_type, molecules in molecule_dict.items():
    #     print(f"{molecule_type}:")
    #     for key, atoms in molecules.items():
    #         print(f"  {key}: {atoms}")

    # Center the structure on the protein barycenter and apply periodic boundary conditions
    centered_coords = traj.center_on_protein_barycenter(0)

    print("Centered coordinates (first 5 atoms):")
    print(centered_coords[:5])
    # Write the centered structure to a PDB file
    ##traj.write_pdb(1000, args.output)
    exit(1)
    print(f"Centered structure written to {args.output}")

if __name__ == "__main__":
    main()
