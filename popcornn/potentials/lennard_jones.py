import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput
from popcornn.tools import radius_graph

class LennardJones(BasePotential):
    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=3.0, **kwargs):
        """
        Constructor for the Lennard-Jones Potential.

        The potential is given by:
        E_ij = 4 * epsilon * ((sigma / r_ij) ** 12 - (sigma / r_ij) ** 6)
        E = sum_{i<j} E_ij

        r_ij is the distance between atoms i and j, under minimum image convention.
        """
        super().__init__(**kwargs)
        assert self.n_atoms is not None, "Number of atoms must be defined."
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
    
    def forward(self, positions):
        positions_3d = positions.view(-1, self.n_atoms, 3)
        graph_dict = radius_graph(
            positions=positions_3d,
            cell=self.cell,
            pbc=self.pbc,
            cutoff=self.cutoff,
            max_neighbors=-1,
        )
        r = graph_dict['edge_distance'].to(dtype=self.dtype)
        v = graph_dict['edge_distance_vec'].to(dtype=self.dtype)
        # energies_decomposed = 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
        energies_part = (
            4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6) 
            - 4 * self.epsilon * (
                (self.sigma / self.cutoff) ** 12 - (self.sigma / self.cutoff) ** 6
            )
        ).unsqueeze(-1)
        # energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)
        energies = torch.zeros((len(positions_3d), 1), device=self.device, dtype=self.dtype)
        energies.index_add_(0, graph_dict['edge_index'][1] // self.n_atoms, energies_part)
        energies = energies / 2

        forces_part = (
            -24 * self.epsilon * (2 * (self.sigma / r) ** 12 - (self.sigma / r) ** 6) / r ** 2
        ).unsqueeze(-1) * v
        forces = torch.zeros((len(positions_3d) * self.n_atoms, 3), device=self.device, dtype=self.dtype)
        forces.index_add_(0, graph_dict['edge_index'][1], forces_part)
        forces = forces.view(len(positions_3d), self.n_atoms * 3)
        # forces = self.calculate_conservative_forces(energies, positions)
        # forces_decomposed = self.calculate_conservative_forces_decomposed(energies_decomposed, positions)
        return PotentialOutput(
            energies=energies,
            # energies_decomposed=energies_decomposed,
            forces=forces,
            # forces_decomposed=forces_decomposed
        )
