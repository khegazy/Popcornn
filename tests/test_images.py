import pytest
import numpy as np
import torch
from ase import Atoms
from ase.io import read

from popcornn.tools import process_images


@pytest.mark.parametrize(
    'raw_images',
    [[[1.133, -1.486], [-1.166, 1.477]], 'images/wolfe.json']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_list(raw_images, dtype):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    assert images.image_type is list
    assert images.positions.shape == (2, 2)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == dtype
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]], dtype=dtype))
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


@pytest.mark.parametrize(
    'raw_images', 
    [np.array([[1.133, -1.486], [-1.166, 1.477]]), 'images/wolfe.npy']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_numpy(raw_images, dtype):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    assert images.image_type is np.ndarray
    assert images.positions.shape == (2, 2)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == dtype
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]], dtype=dtype))
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


@pytest.mark.parametrize(
    'raw_images',
    [torch.tensor([[1.133, -1.486], [-1.166, 1.477]]), 'images/wolfe.pt']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_torch(raw_images, dtype):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    assert images.image_type is torch.Tensor
    assert images.positions.shape == (2, 2)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == dtype
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]], dtype=dtype))
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


@pytest.mark.parametrize(
    'raw_images',
    [read('images/OC20NEB.xyz', index=':'), 'images/OC20NEB.xyz']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_xyz(raw_images, dtype):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    assert images.image_type is Atoms
    assert images.positions.shape == (2, 42)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == dtype
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
        ], 
        dtype=dtype
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
    assert images.cell.dtype == dtype
    assert torch.allclose(images.cell, torch.tensor(
        [
            [ 9.04583549e+00,  0.00000000e+00,  5.53897714e-16],
            [-6.65912157e-16,  1.08751717e+01,  1.01654017e+00],
            [ 0.00000000e+00,  0.00000000e+00,  3.19663506e+01]
        ],
        dtype=dtype
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


@pytest.mark.parametrize(
    'raw_images',
    [read('images/OC20NEB.traj', index=':'), 'images/OC20NEB.traj']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_traj(raw_images, dtype):
    images = process_images(raw_images, device=torch.device('cpu'), dtype=dtype)
    assert images.image_type is Atoms
    assert images.positions.shape == (10, 42)
    assert images.positions.device == torch.device('cpu')
    assert images.positions.dtype == dtype
    assert torch.allclose(images.positions, torch.tensor(
        [
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.81918876e+00,  2.21880285e-01,  1.81014245e+01,
                 8.47748020e+00,  5.80893319e+00,  2.10479172e+01,
                 4.15724126e+00,  2.98113633e+00,  1.97234387e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.20783699e+00,  3.01397571e+00,  2.35278272e+01,
                 7.13967253e+00,  3.03131271e+00,  2.41233660e+01,
                 5.54306086e+00,  3.82423157e+00,  2.38682326e+01,
                 5.70076617e+00,  2.04184586e+00,  2.36755104e+01,
                 7.10627032e+00,  2.55772780e+00,  2.18160334e+01,
                 6.48094153e+00,  3.25790627e+00,  2.21248448e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.81732936e+00,  2.20627810e-01,  1.81020124e+01,
                 8.47850728e+00,  5.80828270e+00,  2.10455615e+01,
                 4.15185306e+00,  2.98525240e+00,  1.97237599e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.20864601e+00,  2.98257431e+00,  2.38585248e+01,
                 7.13764050e+00,  3.00065959e+00,  2.44588910e+01,
                 5.54398558e+00,  3.79442114e+00,  2.41952623e+01,
                 5.69878755e+00,  2.01190019e+00,  2.40082183e+01,
                 7.10942965e+00,  2.52297827e+00,  2.21505663e+01,
                 6.48558227e+00,  3.22347771e+00,  2.24571390e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.81546982e+00,  2.19375532e-01,  1.81026005e+01,
                 8.47922669e+00,  5.80714213e+00,  2.10434290e+01,
                 4.14664437e+00,  2.98936921e+00,  1.97242678e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.20947977e+00,  2.95119718e+00,  2.41891434e+01,
                 7.13558540e+00,  2.97000938e+00,  2.47943715e+01,
                 5.54492917e+00,  3.76459538e+00,  2.45222689e+01,
                 5.69684440e+00,  1.98199916e+00,  2.43408633e+01,
                 7.11259397e+00,  2.48843713e+00,  2.24849994e+01,
                 6.49029026e+00,  3.18927338e+00,  2.27893322e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.81361022e+00,  2.18123328e-01,  1.81031889e+01,
                 8.47971188e+00,  5.80561933e+00,  2.10415094e+01,
                 4.14153194e+00,  2.99348462e+00,  1.97248867e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21033711e+00,  2.91984049e+00,  2.45196939e+01,
                 7.13350530e+00,  2.93936121e+00,  2.51298090e+01,
                 5.54589260e+00,  3.73475254e+00,  2.48492532e+01,
                 5.69494072e+00,  1.95214703e+00,  2.46734409e+01,
                 7.11577461e+00,  2.45406387e+00,  2.28193538e+01,
                 6.49506209e+00,  3.15523044e+00,  2.31214810e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.81175061e+00,  2.16871128e-01,  1.81037772e+01,
                 8.48010190e+00,  5.80393551e+00,  2.10397034e+01,
                 4.13644009e+00,  2.99759888e+00,  1.97255344e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21120913e+00,  2.88849217e+00,  2.48502141e+01,
                 7.13140276e+00,  2.90871578e+00,  2.54652189e+01,
                 5.54687330e+00,  3.70489530e+00,  2.51762243e+01,
                 5.69306369e+00,  1.92232803e+00,  2.50059758e+01,
                 7.11896568e+00,  2.41976538e+00,  2.31536764e+01,
                 6.49986190e+00,  3.12124501e+00,  2.34536328e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.80989101e+00,  2.15618914e-01,  1.81043655e+01,
                 8.48048832e+00,  5.80224126e+00,  2.10379222e+01,
                 4.13133707e+00,  3.00171281e+00,  1.97261722e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21208871e+00,  2.85714276e+00,  2.51807320e+01,
                 7.12927888e+00,  2.87807515e+00,  2.58006142e+01,
                 5.54787035e+00,  3.67502594e+00,  2.55031901e+01,
                 5.69119947e+00,  1.89252591e+00,  2.53384931e+01,
                 7.12215677e+00,  2.38547556e+00,  2.34880011e+01,
                 6.50466106e+00,  3.08725325e+00,  2.37858074e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.80803142e+00,  2.14366683e-01,  1.81049538e+01,
                 8.48090853e+00,  5.80060043e+00,  2.10361185e+01,
                 4.12621609e+00,  3.00582685e+00,  1.97267902e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21297268e+00,  2.82578783e+00,  2.55112596e+01,
                 7.12713382e+00,  2.84744086e+00,  2.61360014e+01,
                 5.54888375e+00,  3.64514551e+00,  2.58301546e+01,
                 5.68933948e+00,  1.86273090e+00,  2.56710080e+01,
                 7.12534123e+00,  2.35116525e+00,  2.38223435e+01,
                 6.50944720e+00,  3.05323158e+00,  2.41180087e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.80617184e+00,  2.13114440e-01,  1.81055421e+01,
                 8.48136692e+00,  5.79902202e+00,  2.10342796e+01,
                 4.12107988e+00,  3.00994112e+00,  1.97273907e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21386098e+00,  2.79442678e+00,  2.58417982e+01,
                 7.12496760e+00,  2.81681352e+00,  2.64713822e+01,
                 5.54991363e+00,  3.61525427e+00,  2.61571188e+01,
                 5.68748040e+00,  1.83293918e+00,  2.60035256e+01,
                 7.12851664e+00,  2.31682988e+00,  2.41567072e+01,
                 6.51421886e+00,  3.01917902e+00,  2.44502345e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.80431227e+00,  2.11862190e-01,  1.81061303e+01,
                 8.48185368e+00,  5.79749055e+00,  2.10324118e+01,
                 4.11593367e+00,  3.01405558e+00,  1.97279793e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21475455e+00,  2.76306155e+00,  2.61723452e+01,
                 7.12278049e+00,  2.78619284e+00,  2.68067556e+01,
                 5.55095977e+00,  3.58535242e+00,  2.64840820e+01,
                 5.68562209e+00,  1.80315017e+00,  2.63360454e+01,
                 7.13168292e+00,  2.28247692e+00,  2.44910890e+01,
                 6.51897986e+00,  2.98510234e+00,  2.47824808e+01
            ],
            [
                -1.63483589e-33,  2.66988983e-17,  1.26533470e+01,
                -3.32956078e-16,  5.43758583e+00,  1.58254805e+01,
                 4.52291775e+00,  2.71879292e+00,  1.42394133e+01,
                 4.52291775e+00,  8.15637875e+00,  1.20838213e+01,
                 8.80245269e+00,  2.10609936e-01,  1.81067186e+01,
                 8.48235489e+00,  5.79598315e+00,  2.10305286e+01,
                 4.11078265e+00,  3.01817014e+00,  1.97285622e+01,
                 4.52291775e+00,  8.15637875e+00,  1.74115467e+01,
                 6.21565284e+00,  2.73169354e+00,  2.65028971e+01,
                 7.12057573e+00,  2.75557713e+00,  2.71421208e+01,
                 5.55201927e+00,  3.55544153e+00,  2.68110436e+01,
                 5.68376604e+00,  1.77336548e+00,  2.66685650e+01,
                 7.13484263e+00,  2.24811624e+00,  2.48254820e+01,
                 6.52373510e+00,  2.95101175e+00,  2.51147421e+01
            ]
        ],
        dtype=dtype
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
    assert images.cell.dtype == dtype
    assert torch.allclose(images.cell, torch.tensor(
        [
            [ 9.04583549e+00,  0.00000000e+00,  5.53897714e-16],
            [-6.65912157e-16,  1.08751717e+01,  1.01654017e+00],
            [ 0.00000000e+00,  0.00000000e+00,  3.19663506e+01]
        ],
        dtype=dtype
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
    assert len(images) == 10


def test_charge_spin():
    images = process_images('images/T1x.xyz', device=torch.device('cpu'), dtype=torch.float32)
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge.device == torch.device('cpu')
    assert images.charge.dtype == torch.int
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin.device == torch.device('cpu')
    assert images.spin.dtype == torch.int
    assert images.spin == 1