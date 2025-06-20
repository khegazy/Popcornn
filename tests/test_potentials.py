import pytest
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
    path = get_path('linear', images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=dtype)
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
    images = process_images('images/LJ13.xyz', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=torch.device('cpu'), dtype=dtype)

    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[-44.32680142], [-34.90609603], [-44.32680142]], dtype=dtype),
        atol=1e-5
    )
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor(
            [
                [
                    -1.22026868e-07,  4.65947868e-07, -8.69690268e-07,
                     1.46976012e-07, -1.02301428e-06, -3.09678953e-07,
                    -2.95339904e-07,  3.82525299e-07,  3.71750898e-07,
                     7.84619281e-09,  2.96609298e-07,  5.88354204e-07,
                    -2.14042781e-07,  2.89353784e-07,  1.63606890e-07,
                     2.95339903e-07, -3.82525299e-07, -3.71750897e-07,
                     2.14042781e-07, -2.89353784e-07, -1.63606889e-07,
                     4.38490266e-07, -7.83095953e-07,  1.05010648e-07,
                    -4.38490265e-07,  7.83095953e-07, -1.05010648e-07,
                    -7.84619281e-09, -2.96609299e-07, -5.88354204e-07,
                     1.22026868e-07, -4.65947868e-07,  8.69690268e-07,
                    -1.46976012e-07,  1.02301428e-06,  3.09678953e-07,
                    -2.22044605e-16, -4.44089210e-16, -1.73472348e-18
                ],
                [
                     4.48349762e+00, -1.32186255e+01,  1.41587260e+01,
                    -7.82475022e+00, -1.89677327e+01,  1.86819082e+00,
                     6.72522859e+00, -1.35441595e+01, -1.48608792e+01,
                    -1.24385188e+01, -5.71574566e+00,  1.38198016e+01,
                     1.43742609e+01,  5.38388330e+00,  1.51975831e+01,
                    -6.72522859e+00,  1.35441595e+01,  1.48608792e+01,
                    -1.43742609e+01, -5.38388330e+00, -1.51975831e+01,
                     1.89497235e+01, -7.81237880e+00, -3.58365115e-01,
                    -1.89497235e+01,  7.81237880e+00,  3.58365115e-01,
                     1.24385188e+01,  5.71574566e+00, -1.38198016e+01,
                    -4.48349762e+00,  1.32186255e+01, -1.41587260e+01,
                     7.82475022e+00,  1.89677327e+01, -1.86819082e+00,
                     0.00000000e+00,  8.88178420e-16,  1.02001740e-15
                ],
                [
                     2.89353784e-07,  2.14042781e-07,  1.63606890e-07,
                    -7.83095953e-07, -4.38490265e-07,  1.05010648e-07,
                    -2.96609298e-07,  7.84619281e-09, -5.88354204e-07,
                     4.65947868e-07,  1.22026868e-07, -8.69690268e-07,
                    -3.82525299e-07, -2.95339903e-07, -3.71750898e-07,
                     2.96609298e-07, -7.84619303e-09,  5.88354204e-07,
                     3.82525299e-07,  2.95339904e-07,  3.71750897e-07,
                     1.02301428e-06,  1.46976012e-07,  3.09678953e-07,
                    -1.02301428e-06, -1.46976012e-07, -3.09678953e-07,
                    -4.65947867e-07, -1.22026868e-07,  8.69690268e-07,
                    -2.89353784e-07, -2.14042781e-07, -1.63606890e-07,
                     7.83095953e-07,  4.38490265e-07, -1.05010649e-07,
                    -1.11022302e-16,  4.44089210e-16,  1.73472348e-16
                ]
            ], 
            dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
