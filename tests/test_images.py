import pytest
import numpy as np
import torch
from ase import Atoms
from ase.io import read

from popcornn.tools import process_images

@pytest.mark.parametrize(
    'raw_images', 
    [np.array([[1.133, -1.486], [-1.166, 1.477]]), 'images/wolfe.npy']
)
def test_numpy(raw_images):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float32)
    assert images.image_type is np.ndarray
    assert images.positions.shape == (2, 2)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == torch.float32
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]]))
    assert images.fix_positions.shape == (2,)
    assert images.fix_positions.device == torch.device('cpu')
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(images.fix_positions == torch.zeros(2, dtype=torch.bool))
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 2

    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float64)
    assert images.positions.dtype == torch.float64

@pytest.mark.parametrize(
    'raw_images',
    [torch.tensor([[1.133, -1.486], [-1.166, 1.477]]), 'images/wolfe.pt']
)
def test_torch(raw_images):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float32)
    assert images.image_type is torch.Tensor
    assert images.positions.shape == (2, 2)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == torch.float32
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]]))
    assert images.fix_positions.shape == (2,)
    assert images.fix_positions.device == torch.device('cpu')
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(images.fix_positions == torch.zeros(2, dtype=torch.bool))
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 2

    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float64)
    assert images.positions.dtype == torch.float64

@pytest.mark.parametrize(
    'raw_images',
    [read('images/OC20NEB.xyz', index=':'), 'images/OC20NEB.xyz']
)
def test_xyz(raw_images):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float32)
    assert images.image_type is Atoms
    assert images.positions.shape == (2, 42)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == torch.float32
    assert torch.allclose(images.positions, torch.tensor(
        [
            [
                -0.        ,  0.        , 12.65334702, -0.        ,  5.43758583,
                15.82548046,  4.52291775,  2.71879292, 14.23941326,  4.52291775,
                 8.15637875, 12.0838213 ,  8.81918876,  0.22188029, 18.1014245 ,
                 8.4774802 ,  5.80893319, 21.04791724,  4.15724126,  2.98113633,
                19.7234387 ,  4.52291775,  8.15637875, 17.41154671,  6.20783699,
                 3.01397571, 23.5278272 ,  7.13967253,  3.03131271, 24.12336599,
                 5.54306086,  3.82423157, 23.8682326 ,  5.70076617,  2.04184586,
                23.67551037,  7.10627032,  2.5577278 , 21.8160334 ,  6.48094153,
                 3.25790627, 22.12484476
            ],
            [
                -0.        ,  0.        , 12.65334702, -0.        ,  5.43758583,
                15.82548046,  4.52291775,  2.71879292, 14.23941326,  4.52291775,
                 8.15637875, 12.0838213 ,  8.80245269,  0.21060994, 18.10671856,
                 8.48235489,  5.79598315, 21.03052856,  4.11078265,  3.01817014,
                19.72856215,  4.52291775,  8.15637875, 17.41154671,  6.21565284,
                 2.73169354, 26.50289708,  7.12057573,  2.75557713, 27.14212081,
                 5.55201927,  3.55544153, 26.81104362,  5.68376604,  1.77336548,
                26.668565  ,  7.13484263,  2.24811624, 24.82548202,  6.5237351 ,
                 2.95101175, 25.11474215
            ]
        ]
    ))
    assert images.fix_positions.shape == (42,)
    assert images.fix_positions.device == torch.device('cpu')
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(images.fix_positions == torch.tensor(
        [ 
             True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True, False, False, False, False, False, False,
            False, False, False,  True,  True,  True, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False
        ], 
        dtype=torch.bool
    ))
    assert images.atomic_numbers is not None
    assert images.atomic_numbers.shape == (14,)
    assert images.atomic_numbers.device == torch.device('cpu')
    assert images.atomic_numbers.dtype == torch.int
    assert torch.all(images.atomic_numbers == torch.tensor(
        [55, 55, 55, 55, 55, 55, 55, 55,  6,  1,  1,  1,  1,  8],
        dtype=torch.int
    ))
    assert images.pbc is not None
    assert images.pbc.shape == (3,)
    assert images.pbc.device == torch.device('cpu')
    assert images.pbc.dtype == torch.bool
    assert torch.all(images.pbc == torch.ones(3, dtype=torch.bool))
    assert images.cell is not None
    assert images.cell.shape == (3, 3)
    assert images.cell.device == torch.device('cpu')
    assert images.cell.dtype == torch.float32
    assert torch.allclose(images.cell, torch.tensor(
        [
            [ 9.04583549e+00,  0.00000000e+00,  5.53897714e-16],
            [-6.65912157e-16,  1.08751717e+01,  1.01654017e+00],
            [ 0.00000000e+00,  0.00000000e+00,  3.19663506e+01]
        ]
    ))
    assert images.tags is not None
    assert images.tags.shape == (14,)
    assert images.tags.device == torch.device('cpu')
    assert images.tags.dtype == torch.int
    assert torch.all(images.tags == torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2],
        dtype=torch.int
    ))
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge.device == torch.device('cpu')
    assert images.charge.dtype == torch.int
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin.device == torch.device('cpu')
    assert images.spin.dtype == torch.int
    assert images.spin == 0
    assert len(images) == 2

    images = process_images(raw_images, device=torch.device('cpu'), dtype=torch.float64)
    assert images.positions.dtype == torch.float64

