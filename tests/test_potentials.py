import pytest
import ase
from ase.io import read
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from ase.mep import interpolate
import numpy as np
import torch

from popcornn.tools import process_images
from popcornn.paths import get_path
from popcornn.potentials import get_potential


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
def test_wolfe(dtype):
    images = process_images('images/wolfe.json', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('wolfe_schlegel',images=images, device=torch.device('cpu'), dtype=dtype)

    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies, 
        torch.tensor([[-64.81812976863], [-0.04301175469875001], [-66.45448705023]], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor(
            [
                [0.0032145200000005536, 0.04517024000000619], 
                [-2.614820315, -1.194996355], 
                [-0.0003081600000115481, -0.06473332000001358]
            ], 
            dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
def test_lennard_jones(dtype):
    raw_images = [read('images/LJ13.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies_decomposed.shape == (3, 156)
    assert potential_output.energies_decomposed.device == torch.device('cpu')
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    raw_images = [read('images/LJ35.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device == torch.device('cpu')
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
def test_repel(dtype):
    class RepelCalculator(Calculator):
        '''
        Revised from ASE LennardJones Calculator to use the repulsive potential
        '''
        implemented_properties = ['energy', 'forces']
        def __init__(self, alpha=1.7, beta=0.01, cutoff=3.0, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.beta = beta
            self.rc = cutoff
            self.nl = None

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            natoms = len(self.atoms)
            alpha = self.alpha
            beta = self.beta
            rc = self.rc
            if self.nl is None or 'numbers' in system_changes:
                self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)
            self.nl.update(self.atoms)
            positions = self.atoms.positions
            cell = self.atoms.cell

            energies = np.zeros(natoms)
            forces = np.zeros((natoms, 3))
            for ii in range(natoms):
                neighbors, offsets = self.nl.get_neighbors(ii)
                cells = np.dot(offsets, cell)
                distance_vectors = positions[neighbors] + cells - positions[ii]
                r = np.linalg.norm(distance_vectors, axis=1)
                r0 = covalent_radii[self.atoms.numbers[ii]] + covalent_radii[self.atoms.numbers[neighbors]]
                pairwise_energies = np.exp(-alpha * (r - r0) / r0) + beta * r0 / r
                pairwise_energies[r > rc] = 0.0
                pairwise_forces = - np.exp(-alpha * (r - r0) / r0) * alpha / r0 / r - beta * r0 / r ** 3
                pairwise_forces[r > rc] = 0.0
                pairwise_energies -= (np.exp(-alpha * (rc - r0) / r0) + beta * r0 / rc) * (r < rc)
                pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
                energies[ii] += 0.5 * pairwise_energies.sum()
                forces[ii] += pairwise_forces.sum(axis=0)
            energy = energies.sum()
            self.results['energy'] = energy
            self.results['forces'] = forces

    raw_images = [read('images/LJ13.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('repel', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies_decomposed.shape == (3, 156)
    assert potential_output.energies_decomposed.device == torch.device('cpu')
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    raw_images = [read('images/LJ35.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('repel', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device == torch.device('cpu')
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None