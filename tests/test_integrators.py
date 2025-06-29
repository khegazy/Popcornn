import pytest
import torch

from popcornn.tools import process_images
from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import ODEintegrator


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_integral(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')

    images = process_images('images/muller_brown.json', device=device, dtype=dtype)
    # path = get_path('linear', images=images, device=device, dtype=dtype)  # TODO: make this integration works even it doesn't have trainable parameters
    path = get_path('mlp', images=images, device=device, dtype=dtype)
    for param in path.parameters():
        param.data.zero_()  # Initialize parameters to zero for testing
    potential = get_potential('muller_brown', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    integrator = ODEintegrator(path_ode_names='projected_variable_reaction_energy', device=device, dtype=dtype)
    path_integral = integrator.integrate_path(path, t_init=torch.tensor([0.], device=device, dtype=dtype), t_final=torch.tensor([1.], device=device, dtype=dtype))
    assert torch.allclose(path_integral.loss, torch.tensor(234.27849339874268, device=device, dtype=dtype), atol=1e-2)  # TODO: investigate why this is not close enough

    images = process_images('images/wolfe.json', device=device, dtype=dtype)
    # path = get_path('linear', images=images, device=device, dtype=dtype)  # TODO: make this integration works even it doesn't have trainable parameters
    path = get_path('mlp', images=images, device=device, dtype=dtype)
    for param in path.parameters():
        param.data.zero_()  # Initialize parameters to zero for testing
    potential = get_potential('wolfe_schlegel', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    integrator = ODEintegrator(path_ode_names='projected_variable_reaction_energy', device=device, dtype=dtype)
    path_integral = integrator.integrate_path(path, t_init=torch.tensor([0.], device=device, dtype=dtype), t_final=torch.tensor([1.], device=device, dtype=dtype))
    assert torch.allclose(path_integral.loss, torch.tensor(131.19240801706766, device=device, dtype=dtype), atol=1e-5)

    images = process_images('images/LJ13.xyz', device=device, dtype=dtype)
    # path = get_path('linear', images=images, device=device, dtype=dtype)
    path = get_path('mlp', images=images, device=device, dtype=dtype)
    for param in path.parameters():
        param.data.zero_()  # Initialize parameters to zero for testing
    potential = get_potential('lennard_jones', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    integrator = ODEintegrator(path_ode_names='projected_variable_reaction_energy', device=device, dtype=dtype)
    path_integral = integrator.integrate_path(path, t_init=torch.tensor([0.], device=device, dtype=dtype), t_final=torch.tensor([1.], device=device, dtype=dtype))
    assert torch.allclose(path_integral.loss, torch.tensor(18.8414107735614, device=device, dtype=dtype), atol=1e-5)

    images = process_images('images/LJ35.xyz', device=device, dtype=dtype)
    # path = get_path('linear', images=images, device=device, dtype=dtype)
    path = get_path('mlp', images=images, device=device, dtype=dtype)
    for param in path.parameters():
        param.data.zero_()  # Initialize parameters to zero for testing
    potential = get_potential('lennard_jones', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    integrator = ODEintegrator(path_ode_names='projected_variable_reaction_energy', device=device, dtype=dtype)
    path_integral = integrator.integrate_path(path, t_init=torch.tensor([0.], device=device, dtype=dtype), t_final=torch.tensor([1.], device=device, dtype=dtype))
    assert torch.allclose(path_integral.loss, torch.tensor(19.353627488510817, device=device, dtype=dtype), atol=1e-2)  # TODO: investigate why this is not close enough

