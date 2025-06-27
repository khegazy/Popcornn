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
    images = process_images('images/LJ13.xyz', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[-43.89940496348337], [-34.47869957670268], [-43.899404963483384]], dtype=dtype),
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
        torch.tensor(
            [
                [
                    -1.22026872e-07,  4.65947863e-07, -8.69690254e-07,
                     1.46976010e-07, -1.02301429e-06, -3.09678949e-07,
                    -2.95339901e-07,  3.82525294e-07,  3.71750888e-07,
                     7.84620041e-09,  2.96609302e-07,  5.88354190e-07,
                    -2.14042780e-07,  2.89353779e-07,  1.63606897e-07,
                     2.95339901e-07, -3.82525294e-07, -3.71750888e-07,
                     2.14042780e-07, -2.89353779e-07, -1.63606897e-07,
                     4.38490273e-07, -7.83095954e-07,  1.05010643e-07,
                    -4.38490273e-07,  7.83095954e-07, -1.05010643e-07,
                    -7.84620058e-09, -2.96609302e-07, -5.88354190e-07,
                     1.22026872e-07, -4.65947863e-07,  8.69690255e-07,
                    -1.46976010e-07,  1.02301429e-06,  3.09678949e-07,
                     0.00000000e+00,  4.44089210e-16, -1.29930788e-15
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
                     4.44089210e-16,  8.88178420e-16,  2.77555756e-16
                ],
                [
                     2.89353779e-07,  2.14042780e-07,  1.63606897e-07,
                    -7.83095954e-07, -4.38490272e-07,  1.05010643e-07,
                    -2.96609302e-07,  7.84620080e-09, -5.88354190e-07,
                     4.65947863e-07,  1.22026872e-07, -8.69690254e-07,
                    -3.82525294e-07, -2.95339901e-07, -3.71750888e-07,
                     2.96609302e-07, -7.84620058e-09,  5.88354190e-07,
                     3.82525294e-07,  2.95339901e-07,  3.71750888e-07,
                     1.02301429e-06,  1.46976010e-07,  3.09678949e-07,
                    -1.02301429e-06, -1.46976010e-07, -3.09678949e-07,
                    -4.65947863e-07, -1.22026872e-07,  8.69690255e-07,
                    -2.89353779e-07, -2.14042780e-07, -1.63606896e-07,
                     7.83095954e-07,  4.38490272e-07, -1.05010643e-07,
                     0.00000000e+00,  8.88178420e-16,  1.87350135e-16
                ]
            ], 
            dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    images = process_images('images/LJ35.xyz', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[-264.05734462066886], [-254.38057733122028], [-264.0573446193573]], dtype=dtype),
        atol=1e-3
    )
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device == torch.device('cpu')
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device == torch.device('cpu')
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor(
            [
                [
                    -1.22285641e-02,  7.06103062e-03,  8.96336360e-03,
                     1.38609831e-07, -1.70750605e-06, -1.26215659e-06,
                    -1.22287977e-02,  7.06116544e-03, -8.96136452e-03,
                     4.33273165e-02, -7.50458737e-02, -1.11911718e-06,
                     2.13698026e-07, -1.41214599e-02,  8.96351762e-03,
                    -4.71592519e-02,  8.16814600e-02, -1.26090681e-06,
                     2.13698023e-07, -1.41217295e-02, -8.96151854e-03,
                    -4.33274556e-02,  7.50466766e-02, -1.11911715e-06,
                     4.71252225e-02,  2.72081041e-02, -2.14250868e-01,
                     4.71586852e-02, -8.16799179e-02, -1.26090679e-06,
                     4.71252235e-02,  2.72081047e-02,  2.14253249e-01,
                    -8.66567799e-02, -1.69162409e-06, -1.11911716e-06,
                     4.77294502e-07, -5.44163317e-02, -2.14250867e-01,
                     9.43157158e-02, -1.69162410e-06, -1.26090680e-06,
                     4.77294499e-07, -5.44163329e-02,  2.14253248e-01,
                    -4.33273824e-02, -7.50460711e-02, -1.11911716e-06,
                     5.42230672e-01,  3.13057560e-01, -5.05074632e-01,
                     4.71591859e-02,  8.16812625e-02, -1.26090679e-06,
                     5.42230687e-01,  3.13057568e-01,  5.05077019e-01,
                     9.24341826e-08,  9.22086393e-07, -1.11911721e-06,
                    -4.71255538e-02,  2.72079745e-02, -2.14250868e-01,
                     9.24341815e-08,  9.15434880e-07, -1.26066773e-06,
                    -4.71255548e-02,  2.72079751e-02,  2.14253249e-01,
                     8.66558978e-02, -1.56508386e-06, -1.11911716e-06,
                     1.22291404e-02,  7.06089770e-03,  8.96324852e-03,
                    -9.43168873e-02, -1.56508386e-06, -1.26090680e-06,
                     1.22293739e-02,  7.06103252e-03, -8.96124944e-03,
                     7.55834859e-07,  8.86682148e-07, -1.11911715e-06,
                    -5.42231857e-01,  3.13057757e-01, -5.05074639e-01,
                     7.53570981e-07,  8.78762594e-07, -1.26066768e-06,
                    -5.42231872e-01,  3.13057765e-01,  5.05077026e-01,
                     4.33273424e-02,  7.50467596e-02, -1.11911718e-06,
                     5.38667032e-07, -6.26115453e-01, -5.05074622e-01,
                    -4.71587984e-02, -8.16798349e-02, -1.26090681e-06,
                     5.38667032e-07, -6.26115470e-01,  5.05077009e-01
                ],
                [
                     1.15614006e+01,  6.67497760e+00, -3.78067715e+01,
                     2.37290087e-02, -4.11013162e-02, -1.26146956e-06,
                     1.15614024e+01,  6.67497866e+00,  3.78067795e+01,
                    -7.44724937e-07,  5.27277319e-07, -1.11911718e-06,
                     2.66187136e-01, -8.83876788e-01,  5.64067743e-01,
                    -1.14738885e-06,  1.22766895e-06, -1.26110412e-06,
                     2.66187144e-01, -8.83877082e-01, -5.64065755e-01,
                    -2.56654615e+01, -1.48179607e+01, -6.29578589e-06,
                    -6.32367408e-01,  6.72462438e-01,  5.64067474e-01,
                    -2.37282879e-02,  4.10999223e-02, -1.26146953e-06,
                    -6.32367659e-01,  6.72462591e-01, -5.64065486e-01,
                     3.46386573e+01,  1.99986370e+01, -1.11911716e-06,
                     1.60682049e-01,  2.08658405e-01, -3.61141453e-01,
                     2.67475537e-02,  1.54412989e-02, -1.26078044e-06,
                     1.60682053e-01,  2.08658411e-01,  3.61143837e-01,
                    -1.95241403e-01, -1.13161808e+00, -1.11911716e-06,
                     2.61043032e-01,  3.48257077e-02, -3.61141457e-01,
                     5.49786452e-02,  3.20906263e-02, -1.26085599e-06,
                     2.61043039e-01,  3.48257088e-02,  3.61143842e-01,
                    -1.07763138e+00,  3.96725517e-01, -1.11911721e-06,
                     1.10618510e-01,  6.38644566e-02, -2.30400991e-01,
                     5.52792798e-02,  3.15680837e-02, -1.26085605e-06,
                     1.10618513e-01,  6.38644581e-02,  2.30403373e-01,
                     1.07763110e+00, -3.96726175e-01, -1.11911716e-06,
                     1.05505634e+00,  9.45684380e-03,  1.24273053e+00,
                    -5.52798667e-02, -3.15687340e-02, -1.26085600e-06,
                     1.05505657e+00,  9.45698406e-03, -1.24272852e+00,
                     1.95242915e-01,  1.13161940e+00, -1.11911715e-06,
                    -4.85607295e-01, -2.80364752e-01, -8.41241253e-01,
                    -5.49768098e-02, -3.20887183e-02, -1.26085600e-06,
                    -4.85607308e-01, -2.80364760e-01,  8.41243646e-01,
                    -3.46386608e+01, -1.99986399e+01, -1.11911718e-06,
                     5.35718463e-01,  9.08976766e-01,  1.24273067e+00,
                    -2.67487413e-02, -1.54425939e-02, -1.26078045e-06,
                     5.35718702e-01,  9.08976899e-01, -1.24272866e+00
                ],
                [
                     8.02401458e-07, -1.41208508e-02,  8.96335962e-03,
                    -4.71579644e-02,  8.16785241e-02, -1.26090682e-06,
                     8.02401460e-07, -1.41211205e-02, -8.96136054e-03,
                    -4.33292222e-02,  7.50476539e-02, -1.11911718e-06,
                     4.71268005e-02,  2.72084520e-02, -2.14250867e-01,
                     4.71569569e-02, -8.16790045e-02, -1.26090681e-06,
                     4.71268015e-02,  2.72084526e-02,  2.14253249e-01,
                     4.33281764e-02, -7.50480704e-02, -1.11911717e-06,
                    -1.22306671e-02,  7.06014442e-03,  8.96326490e-03,
                     5.82158149e-07,  3.13656980e-07, -1.26215656e-06,
                    -1.22309006e-02,  7.06027924e-03, -8.96126583e-03,
                    -4.33284575e-02, -7.50479296e-02, -1.11911716e-06,
                     5.42231718e-01,  3.13058118e-01, -5.05074630e-01,
                     4.71576114e-02,  8.16785394e-02, -1.26090680e-06,
                     5.42231733e-01,  3.13058127e-01,  5.05077017e-01,
                     1.08360135e-06,  1.03060595e-06, -1.11911716e-06,
                    -4.71271618e-02,  2.72087512e-02, -2.14250868e-01,
                     1.08133747e-06,  1.02933790e-06, -1.26066767e-06,
                    -4.71271628e-02,  2.72087518e-02,  2.14253249e-01,
                    -8.66561956e-02,  9.14992410e-07, -1.11911721e-06,
                     1.03215785e-06, -5.44176726e-02, -2.14250868e-01,
                     9.43163001e-02,  9.14992409e-07, -1.26090685e-06,
                     1.03215784e-06, -5.44176738e-02,  2.14253249e-01,
                    -6.83850426e-07, -1.57217784e-06, -1.11911716e-06,
                    -5.42231400e-01,  3.13057665e-01, -5.05074631e-01,
                    -6.79190590e-07, -1.56552633e-06, -1.26066768e-06,
                    -5.42231414e-01,  3.13057673e-01,  5.05077017e-01,
                     4.33289006e-02,  7.50474305e-02, -1.11911715e-06,
                    -6.01546003e-07, -6.26114313e-01, -5.05074632e-01,
                    -4.71573504e-02, -8.16793536e-02, -1.26090680e-06,
                    -6.01546003e-07, -6.26114330e-01,  5.05077019e-01,
                     8.66558813e-02,  3.96701154e-07, -1.11911718e-06,
                     1.22299184e-02,  7.05988057e-03,  8.96338334e-03,
                    -9.43169038e-02,  3.96701155e-07, -1.26090681e-06,
                     1.22301519e-02,  7.06001539e-03, -8.96138426e-03
                ]
            ], 
            dtype=dtype
        ),
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
    images = process_images('images/LJ13.xyz', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('repel', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[99.00971093], [103.29090201], [99.00971093]], dtype=dtype),
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
        torch.tensor(
            [
                [
                     3.74807527e+00, -9.05379999e+00,  6.23059642e+00,
                    -6.77358762e+00, -9.43164459e+00,  4.78022919e-02,
                     3.84260332e+00, -9.18356442e+00, -5.97797947e+00,
                    -5.71247278e+00, -2.24106865e+00,  9.85824326e+00,
                     5.55952318e+00,  2.45103188e+00,  9.89564747e+00,
                    -3.84260332e+00,  9.18356442e+00,  5.97797947e+00,
                    -5.55952318e+00, -2.45103188e+00, -9.89564747e+00,
                     1.14648850e+01, -1.83966645e+00,  1.08323557e-01,
                    -1.14648850e+01,  1.83966645e+00, -1.08323555e-01,
                     5.71247278e+00,  2.24106865e+00, -9.85824326e+00,
                    -3.74807527e+00,  9.05379999e+00, -6.23059642e+00,
                     6.77358763e+00,  9.43164459e+00, -4.78022919e-02,
                    -7.10542736e-10, -1.42108547e-09,  7.10542736e-10
                ],
                [
                     3.27002990e+00, -7.73801987e+00,  8.53452017e+00,
                    -4.56290143e+00, -1.10696619e+01,  1.01681149e-01,
                     3.23711573e+00, -7.89442478e+00, -8.39785981e+00,
                    -7.80702963e+00, -3.17402563e+00,  8.50933310e+00,
                     7.82584324e+00,  3.33264390e+00,  8.42300080e+00,
                    -3.23711573e+00,  7.89442478e+00,  8.39785981e+00,
                    -7.82584324e+00, -3.33264390e+00, -8.42300080e+00,
                     1.10700645e+01, -4.56286970e+00,  2.75160126e-02,
                    -1.10700645e+01,  4.56286970e+00, -2.75160126e-02,
                     7.80702963e+00,  3.17402563e+00, -8.50933309e+00,
                    -3.27002990e+00,  7.73801987e+00, -8.53452017e+00,
                     4.56290143e+00,  1.10696619e+01, -1.01681149e-01,
                     7.10542736e-10, -7.10542736e-10, -1.42108547e-09
                ],
                [
                     2.45103188e+00, -5.55952319e+00,  9.89564747e+00,
                    -1.83966645e+00, -1.14648850e+01,  1.08323556e-01,
                     2.24106865e+00, -5.71247278e+00, -9.85824326e+00,
                    -9.05379999e+00, -3.74807527e+00,  6.23059642e+00,
                     9.18356442e+00,  3.84260332e+00,  5.97797947e+00,
                    -2.24106865e+00,  5.71247278e+00,  9.85824326e+00,
                    -9.18356442e+00, -3.84260332e+00, -5.97797947e+00,
                     9.43164459e+00, -6.77358762e+00, -4.78022926e-02,
                    -9.43164459e+00,  6.77358763e+00,  4.78022926e-02,
                     9.05379999e+00,  3.74807527e+00, -6.23059642e+00,
                    -2.45103188e+00,  5.55952318e+00, -9.89564747e+00,
                     1.83966645e+00,  1.14648850e+01, -1.08323555e-01,
                     7.10542736e-10, -0.00000000e+00, -7.10542736e-10
                ]
            ],
            dtype=dtype
        ),
        atol=1e-5
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    images = process_images('images/LJ35.xyz', device=torch.device('cpu'), dtype=dtype)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=dtype)
    potential = get_potential('repel', images=images, device=torch.device('cpu'), dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=torch.device('cpu'), dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == torch.device('cpu')
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[943.86301799], [943.92821147], [943.86301843]], dtype=dtype),
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
        torch.tensor(
            [
                [
                    -4.51865894e-01,  2.60889008e-01,  1.27651820e+00,
                    -1.84400051e-07,  9.63495950e-08,  1.81069026e-06,
                    -4.51865808e-01,  2.60888958e-01, -1.27657624e+00,
                     4.58695987e-01, -7.94485590e-01, -1.02033937e-06,
                    -1.67574399e-07, -5.21800139e-01,  1.27651820e+00,
                     7.66717167e-02, -1.32799610e-01,  1.81074711e-06,
                    -1.67517555e-07, -5.21800040e-01, -1.27657624e+00,
                    -4.58696817e-01,  7.94437673e-01, -1.02039621e-06,
                     1.29365550e-01,  7.46932274e-02,  1.03808842e+00,
                    -7.66725335e-02,  1.32791062e-01,  1.81069026e-06,
                     1.29365551e-01,  7.46932279e-02, -1.03813575e+00,
                    -9.17392468e-01, -5.42854650e-08, -1.02033937e-06,
                     8.82323548e-07, -1.49380911e-01,  1.03808841e+00,
                    -1.53343030e-01, -5.42854650e-08,  1.81074711e-06,
                     8.82323548e-07, -1.49380912e-01, -1.03813575e+00,
                    -4.58696227e-01, -7.94485560e-01, -1.02039621e-06,
                    -2.15482812e-01, -1.24415763e-01,  1.14532671e+00,
                    -7.66719562e-02, -1.32799580e-01,  1.81080395e-06,
                    -2.15482773e-01, -1.24415740e-01, -1.14537884e+00,
                    -5.48538992e-08, -9.84641702e-07, -1.02033937e-06,
                    -1.29365871e-01,  7.46932709e-02,  1.03808842e+00,
                    -5.47970558e-08, -7.87736099e-07,  1.81069026e-06,
                    -1.29365872e-01,  7.46932715e-02, -1.03813575e+00,
                     9.17338571e-01, -3.63741037e-07, -1.02033937e-06,
                     4.51893160e-01,  2.60888699e-01,  1.27651832e+00,
                     1.53334573e-01, -3.63627350e-07,  1.81069026e-06,
                     4.51893074e-01,  2.60888650e-01, -1.27657636e+00,
                     4.81293227e-07,  4.31441549e-08, -1.02033937e-06,
                     2.15470390e-01, -1.24416083e-01,  1.14532671e+00,
                     4.45652404e-07, -1.89402272e-07,  1.81074711e-06,
                     2.15470351e-01, -1.24416060e-01, -1.14537884e+00,
                     4.58696222e-01,  7.94437415e-01, -1.02039621e-06,
                     3.11672466e-07,  2.48832872e-01,  1.14532676e+00,
                     7.66719381e-02,  1.32790804e-01,  1.81074711e-06,
                     3.11615622e-07,  2.48832827e-01, -1.14537890e+00
                ],
                [
                     1.84357933e-01,  1.06440538e-01,  1.11004303e+00,
                    -2.01294187e-01,  3.48647102e-01,  1.81074711e-06,
                     1.84357932e-01,  1.06440537e-01, -1.11009359e+00,
                    -4.92264007e-08, -6.28688213e-08, -1.02033937e-06,
                    -1.32048953e-01, -1.55121854e-02,  1.47083814e+00,
                    -1.39436906e-07, -3.39923645e-08,  1.81074711e-06,
                    -1.32048938e-01, -1.55121269e-02, -1.47090486e+00,
                     3.99690769e-01,  2.30761046e-01, -8.83915163e-07,
                    -7.94577745e-02, -1.06606857e-01,  1.47083813e+00,
                     2.01290782e-01, -3.48626764e-01,  1.81074711e-06,
                    -7.94577164e-02, -1.06606873e-01, -1.47090485e+00,
                    -3.48183079e-01, -2.01024622e-01, -1.02028253e-06,
                    -2.98005318e-04, -5.08296251e-01,  1.10038339e+00,
                     4.48733007e-01,  2.59075679e-01,  1.81074711e-06,
                    -2.97996792e-04, -5.08296228e-01, -1.10043351e+00,
                    -3.10538127e-01, -4.72690962e-01, -1.02033937e-06,
                    -4.40339946e-01,  2.53901557e-01,  1.10038341e+00,
                    -4.26383064e-01,  1.62303451e-01,  1.81069026e-06,
                    -4.40339921e-01,  2.53901552e-01, -1.10043353e+00,
                    -5.64639117e-01, -3.25873366e-02, -1.02033937e-06,
                    -2.08012133e-01, -1.20103351e-01,  1.04054923e+00,
                    -7.26335293e-02, -4.50384112e-01,  1.81074711e-06,
                    -2.08012119e-01, -1.20103343e-01, -1.04059667e+00,
                     5.64598060e-01,  3.25883386e-02, -1.02033937e-06,
                    -6.77306655e-02,  4.00736223e-01,  1.48514070e+00,
                     7.26281741e-02,  4.50410003e-01,  1.81074711e-06,
                    -6.77307062e-02,  4.00736175e-01, -1.48520803e+00,
                     3.10538406e-01,  4.72690624e-01, -1.02039621e-06,
                     2.30489507e-01,  1.33089829e-01,  1.26861540e+00,
                     4.26383410e-01, -1.62303747e-01,  1.81074711e-06,
                     2.30489473e-01,  1.33089810e-01, -1.26867305e+00,
                     3.48183762e-01,  2.01011813e-01, -1.02033937e-06,
                     3.13179665e-01, -2.59034415e-01,  1.48514070e+00,
                    -4.48732256e-01, -2.59061097e-01,  1.81074711e-06,
                     3.13179604e-01, -2.59034427e-01, -1.48520804e+00
                ],
                [
                     2.27998953e-07, -5.21777731e-01,  1.27651826e+00,
                     7.66727042e-02, -1.32799470e-01,  1.81069026e-06,
                     2.28055796e-07, -5.21777632e-01, -1.27657631e+00,
                    -4.58696189e-01,  7.94485524e-01, -1.02033937e-06,
                     1.29367255e-01,  7.46947821e-02,  1.03808842e+00,
                    -7.66720202e-02,  1.32799544e-01,  1.81069026e-06,
                     1.29367256e-01,  7.46947828e-02, -1.03813575e+00,
                     4.58701912e-01, -7.94485522e-01, -1.02033937e-06,
                    -4.51866291e-01,  2.60900129e-01,  1.27651823e+00,
                    -6.14022611e-07, -5.89466254e-07,  1.81069026e-06,
                    -4.51866205e-01,  2.60900080e-01, -1.27657628e+00,
                    -4.58695532e-01, -7.94485655e-01, -1.02033937e-06,
                    -2.15482566e-01, -1.24410917e-01,  1.14532670e+00,
                    -7.66712413e-02, -1.32799621e-01,  1.81074711e-06,
                    -2.15482527e-01, -1.24410894e-01, -1.14537883e+00,
                    -5.82645043e-08, -3.38047812e-07, -1.02039621e-06,
                    -1.29365241e-01,  7.46945351e-02,  1.03808842e+00,
                    -5.82645043e-08, -1.05501385e-07,  1.81074711e-06,
                    -1.29365242e-01,  7.46945358e-02, -1.03813575e+00,
                    -9.17405186e-01, -6.73935574e-07, -1.02033937e-06,
                    -1.85309545e-07, -1.49388069e-01,  1.03808841e+00,
                    -1.53345773e-01, -6.73935574e-07,  1.81074711e-06,
                    -1.85252702e-07, -1.49388070e-01, -1.03813575e+00,
                    -7.67954589e-08,  1.02318154e-08, -1.02033937e-06,
                     2.15496317e-01, -1.24411253e-01,  1.14532669e+00,
                    -3.56976670e-08, -2.16516582e-07,  1.81069026e-06,
                     2.15496278e-01, -1.24411231e-01, -1.14537882e+00,
                     4.58696418e-01,  7.94485293e-01, -1.02033937e-06,
                    -1.27045041e-07,  2.48834309e-01,  1.14532668e+00,
                     7.66722518e-02,  1.32799285e-01,  1.81074711e-06,
                    -1.26988198e-07,  2.48834264e-01, -1.14537881e+00,
                     9.17393469e-01, -8.47421688e-07, -1.02033937e-06,
                     4.51866457e-01,  2.60899717e-01,  1.27651823e+00,
                     1.53343828e-01, -8.47421688e-07,  1.81074711e-06,
                     4.51866371e-01,  2.60899667e-01, -1.27657628e+00
                ]
            ],
            dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device == torch.device('cpu')
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None