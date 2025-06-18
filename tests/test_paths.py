import pytest
import torch

from popcornn.tools import process_images
from popcornn.paths import get_path

@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
def test_linear(dtype):
    images = process_images('images/wolfe.json', device=torch.device('cpu'), dtype=dtype)
    path = get_path("linear", images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=dtype)
    assert path.transform is None

    path_output = path()
    assert path_output.time.shape == (101, 1)
    assert path_output.time.device == torch.device('cpu')
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 101, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (101, 2)
    assert path_output.positions.device == torch.device('cpu')
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions, 
        torch.stack([torch.linspace(1.133, -1.166, 101, dtype=dtype), torch.linspace(-1.486, 1.477, 101, dtype=dtype)], dim=1), 
        atol=1e-5
    )

    path_output = path(torch.linspace(0, 1, 11, dtype=dtype))
    assert path_output.time.shape == (11, 1)
    assert path_output.time.device == torch.device('cpu')
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 11, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (11, 2)
    assert path_output.positions.device == torch.device('cpu')
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions, 
        torch.stack([torch.linspace(1.133, -1.166, 11, dtype=dtype), torch.linspace(-1.486, 1.477, 11, dtype=dtype)], dim=1), 
        atol=1e-5
    )

