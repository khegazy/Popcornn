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


# TODO: Implement test for input reshape
@pytest.mark.skip(reason="Input reshape tests are not implemented yet.")
def test_input():
    pass

# TODO: Implement test for output reshape
@pytest.mark.skip(reason="Output reshape tests are not implemented yet.")
def test_output():
    pass


def test_unwrap():
    images = process_images('images/wolfe.json', device=torch.device('cpu'), dtype=torch.float32)
    path = get_path("linear", images=images, unwrap_positions=True, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor([[-0.0165, -0.0045]]),
        atol=1e-5
    )
    path = get_path("linear", images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor([[-0.0165, -0.0045]]),
        atol=1e-5
    )
    
    images = process_images('images/LJ13.xyz', device=torch.device('cpu'), dtype=torch.float32)
    path = get_path("linear", images=images, unwrap_positions=True, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[ 
                 0.28877036, -0.68072619,  0.75120193, -0.40122753, -0.97341411,
                 0.00727275,  0.28339309, -0.69389574, -0.737692  , -0.68785098,
                -0.27898974,  0.74945955,  0.68677095,  0.29317372,  0.73943439,
                -0.28339309,  0.69389574,  0.737692  , -0.68677095, -0.29317372,
                -0.73943439,  0.97341411, -0.40122753,  0.00281924, -0.97341411,
                 0.40122753, -0.00281924,  0.68785098,  0.27898974, -0.74945955,
                -0.28877036,  0.68072619, -0.75120193,  0.40122753,  0.97341411,
                -0.00727275,  0.        ,  0.        ,  0.        
            ]]
        ),
        atol=1e-5
    )
    path = get_path("linear", images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[ 
                 0.28877036, -0.68072619,  0.75120193, -0.40122753, -0.97341411,
                 0.00727275,  0.28339309, -0.69389574, -0.737692  , -0.68785098,
                -0.27898974,  0.74945955,  0.68677095,  0.29317372,  0.73943439,
                -0.28339309,  0.69389574,  0.737692  , -0.68677095, -0.29317372,
                -0.73943439,  0.97341411, -0.40122753,  0.00281924, -0.97341411,
                 0.40122753, -0.00281924,  0.68785098,  0.27898974, -0.74945955,
                -0.28877036,  0.68072619, -0.75120193,  0.40122753,  0.97341411,
                -0.00727275,  0.        ,  0.        ,  0.        
            ]]
        ),
        atol=1e-5
    )
    
    images = process_images('images/LJ35.xyz', device=torch.device('cpu'), dtype=torch.float32)
    path = get_path("linear", images=images, unwrap_positions=True, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is not None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[
                 1.12246205e+00,  6.48053770e-01,  9.16486420e-01, -5.11837599e-17,
                 1.29610753e+00,  6.41540497e+00,  1.12246205e+00,  6.48053770e-01,
                 4.58243212e+00, -1.12246205e+00,  3.24026883e+00,  2.74945927e+00,
                -1.02367520e-16,  2.59221506e+00,  9.16486420e-01, -1.12246205e+00,
                 3.24026883e+00,  6.41540497e+00, -1.02367520e-16,  2.59221506e+00,
                 4.58243212e+00,  5.61231022e-01,  3.24026884e-01,  2.74945927e+00,
                -1.12246205e+00,  4.53637636e+00,  9.16486420e-01, -2.24492410e+00,
                 5.18443013e+00,  6.41540497e+00, -1.12246205e+00,  4.53637636e+00,
                 4.58243212e+00,  2.24492410e+00,  1.29610753e+00,  2.74945927e+00,
                 3.36738614e+00,  6.48053770e-01,  9.16486420e-01,  2.24492410e+00,
                 1.29610753e+00,  6.41540497e+00,  3.36738614e+00,  6.48053770e-01,
                 4.58243212e+00,  1.12246205e+00,  3.24026883e+00,  2.74945927e+00,
                 2.24492410e+00,  2.59221506e+00,  9.16486420e-01,  1.12246205e+00,
                 3.24026883e+00,  6.41540497e+00,  2.24492410e+00,  2.59221506e+00,
                 4.58243212e+00, -3.99321653e-16,  5.18443013e+00,  2.74945927e+00,
                 1.12246205e+00,  4.53637636e+00,  9.16486420e-01, -3.99321653e-16,
                 5.18443013e+00,  6.41540497e+00,  1.12246205e+00,  4.53637636e+00,
                 4.58243212e+00,  4.48984819e+00,  1.29610753e+00,  2.74945927e+00,
                 5.61231024e+00,  6.48053770e-01,  9.16486420e-01,  4.48984819e+00,
                 1.29610753e+00,  6.41540497e+00,  5.61231024e+00,  6.48053770e-01,
                 4.58243212e+00,  3.36738614e+00,  3.24026883e+00,  2.74945927e+00,
                 4.48984819e+00,  2.59221506e+00,  9.16486420e-01,  3.36738614e+00,
                 3.24026883e+00,  6.41540497e+00,  4.48984819e+00,  2.59221506e+00,
                 4.58243212e+00,  2.24492410e+00,  5.18443013e+00,  2.74945927e+00,
                 3.36738614e+00,  4.53637636e+00,  9.16486420e-01,  2.24492410e+00,
                 5.18443013e+00,  6.41540497e+00,  3.36738614e+00,  4.53637636e+00,
                 4.58243212e+00
            ]]
        ),
        atol=1e-5
    )
    path = get_path("linear", images=images, unwrap_positions=False, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is not None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[
                 1.12246205,  0.64805377,  0.91648642,  0.        ,  1.29610753,
                 6.41540497,  1.12246205,  0.64805377,  4.58243212, -1.12246205,
                 3.24026883,  2.74945927,  0.        ,  2.59221506,  0.91648642,
                -1.12246205,  3.24026883,  6.41540497,  0.        ,  2.59221506,
                 4.58243212, -1.12246205,  3.24026883,  2.74945927, -1.12246205,
                 4.53637636,  0.91648642, -2.2449241 ,  5.18443013,  6.41540497,
                -1.12246205,  4.53637636,  4.58243212,  2.2449241 ,  1.29610753,
                 2.74945927,  3.36738614,  0.64805377,  0.91648642,  2.2449241 ,
                 1.29610753,  6.41540497,  3.36738614,  0.64805377,  4.58243212,
                 1.12246205,  3.24026883,  2.74945927,  2.2449241 ,  2.59221506,
                 0.91648642,  1.12246205,  3.24026883,  6.41540497,  2.2449241 ,
                 2.59221506,  4.58243212,  0.        ,  5.18443013,  2.74945927,
                 1.12246205,  4.53637636,  0.91648642,  0.        ,  5.18443013,
                 6.41540497,  1.12246205,  4.53637636,  4.58243212,  4.48984819,
                 1.29610753,  2.74945927,  5.61231024,  0.64805377,  0.91648642,
                 4.48984819,  1.29610753,  6.41540497,  5.61231024,  0.64805377,
                 4.58243212,  3.36738614,  3.24026883,  2.74945927,  4.48984819,
                 2.59221506,  0.91648642,  3.36738614,  3.24026883,  6.41540497,
                 4.48984819,  2.59221506,  4.58243212,  2.2449241 ,  5.18443013,
                 2.74945927,  3.36738614,  4.53637636,  0.91648642,  2.2449241 ,
                 5.18443013,  6.41540497,  3.36738614,  4.53637636,  4.58243212
            ]]
        ),
        atol=1e-5
    )
