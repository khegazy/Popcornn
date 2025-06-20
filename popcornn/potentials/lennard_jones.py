import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput

class LennardJones(BasePotential):
    def __init__(self, epsilon=1.0, sigma=1.0, **kwargs):
        """
        Constructor for the Lennard-Jones Potential.

        The potential is given by:
        E_ij = 4 * epsilon * ((sigma / r_ij) ** 12 - (sigma / r_ij) ** 6)
        E = sum_{i<j} E_ij

        r_ij is the distance between atoms i and j, under minimum image convention.
        """
        super().__init__(**kwargs)
        assert self.n_atoms is not None, "Number of atoms must be defined."
        self.ind = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1, device=self.device)
        self.epsilon = epsilon
        self.sigma = sigma
    
    def forward(self, positions):
        positions_3d = positions.view(-1, self.n_atoms, 3)
        r = positions_3d[:, self.ind[0]] - positions_3d[:, self.ind[1]]
        if self.pbc.any():
            r = wrap_positions(r, self.cell, self.pbc, center=1.0)
        r = torch.norm(r, dim=-1)
        energies_decomposed = 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)

        forces = self.calculate_conservative_forces(energies, positions)
        forces_decomposed = self.calculate_conservative_forces_decomposed(energies_decomposed, positions)
        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed
        )
