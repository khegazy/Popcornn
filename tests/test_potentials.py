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
    potential = get_potential(images=images, name='wolfe_schlegel', device=torch.device('cpu'), dtype=dtype)
    assert potential is not None

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
