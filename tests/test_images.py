import numpy as np
import torch
from popcornn.tools import process_images

def test_wolfe_potential():
    images = process_images('images/wolfe.npy', device=torch.device('cpu'), dtype=torch.float32)
    assert images.image_type is np.ndarray
    assert images.positions.shape == (2, 2)
    assert torch.allclose(images.positions, torch.tensor([[1.133, -1.486], [-1.166, 1.477]]))
    assert images.fix_positions.shape == (2,)
    assert torch.allclose(images.fix_positions, torch.zeros(2, dtype=torch.bool))
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None