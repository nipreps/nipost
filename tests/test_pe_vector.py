import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nibabel.spatialimages import SpatialImage
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter

from nipost import pe_vector, prepare_epi, reconstruct_fieldmap
from tests.test_fieldmap_reconstruct import shift_warp

RAS_2MM = np.diag([2.0, 2.0, 2.0, 1.0])
# Same voxel centres as RAS_2MM, with the i axis stored reversed
LAS_2MM = np.array(
    [
        [-2.0, 0.0, 0.0, 8.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def bold(affine):
    return nb.Nifti1Image(np.zeros((5, 5, 5, 3), dtype='f4'), affine)


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def test_pe_vector_is_one_pe_voxel_times_readout_time():
    v = pe_vector(bold(RAS_2MM), [(1, 0.05)] * 3, nt.Affine())
    assert_allclose(v, [0.0, 0.1, 0.0])


def test_pe_vector_follows_boldref_and_transform_rotations():
    # boldref rotated +30 deg about z, xfm rotates -30 deg: back on the fieldmap x axis
    affine = rot_z(np.pi / 6) @ RAS_2MM
    v = pe_vector(bold(affine), [(0, 0.05)] * 3, nt.Affine(rot_z(-np.pi / 6)))
    assert_allclose(v, [0.1, 0.0, 0.0], atol=1e-6)


def test_pe_vector_is_independent_of_storage_orientation():
    # RAS stored with PE 'i-' and LAS stored with PE 'i' are the same acquisition
    meta = {'TotalReadoutTime': 0.05}
    ras, ras_pe = prepare_epi(bold(RAS_2MM), {**meta, 'PhaseEncodingDirection': 'i-'})
    las, las_pe = prepare_epi(bold(LAS_2MM), {**meta, 'PhaseEncodingDirection': 'i'})
    assert_allclose(pe_vector(ras, ras_pe, nt.Affine()), pe_vector(las, las_pe, nt.Affine()))
    assert_allclose(pe_vector(ras, ras_pe, nt.Affine()), [-0.1, 0.0, 0.0])


def test_pe_vector_rejects_mixed_pe_info():
    with pytest.raises(ValueError):
        pe_vector(bold(RAS_2MM), [(0, 0.05), (0, -0.05), (0, 0.05)], nt.Affine())


def test_pe_vector_rejects_nonlinear_transform():
    with pytest.raises(ValueError):
        pe_vector(bold(RAS_2MM), [(0, 0.05)] * 3, shift_warp(0.0))


@pytest.mark.parametrize(
    ('code', 'flipped'),
    [
        pytest.param('j', False, id='j'),
        pytest.param('j-', False, id='j-'),
        pytest.param('i', True, id='i, stored flipped'),
        pytest.param('i-', True, id='i-, stored flipped'),
    ],
)
def test_jacobian_matches_finite_difference_of_field(code, flipped):
    rng = np.random.default_rng(0)
    # Oblique 10 mm lattice holding a smooth field, and a boldref->fieldmap rigid transform
    lattice = rot_z(0.3) @ nb.affines.from_matvec(np.diag([10.0, 9.0, 11.0]), [-70, -70, -70])
    smooth = gaussian_filter(rng.normal(size=(15, 16, 14)), 1.0) * 400
    coefficients: list[SpatialImage] = [nb.Nifti1Image(smooth.astype('f4'), lattice)]
    xfm_matrix = rot_z(-0.1)
    xfm_matrix[:3, 3] = [3.0, -2.0, 1.5]
    xfm = nt.Affine(xfm_matrix)

    affine = rot_z(0.25) @ nb.affines.from_matvec(np.eye(3), [-15, -15, -12])
    if flipped:
        affine = affine @ nb.affines.from_matvec(np.diag([-1, 1, 1]), [29, 0, 0])
    img = nb.Nifti1Image(np.zeros((30, 30, 24, 2), dtype='f4'), affine)
    boldref, pe_info = prepare_epi(img, {'PhaseEncodingDirection': code, 'TotalReadoutTime': 0.05})
    axis, trt = pe_info[0]

    v = pe_vector(boldref, pe_info, xfm)
    field, jac = reconstruct_fieldmap(coefficients, boldref.slicer[..., 0], xfm, pe_vector=v)

    expected = 1 + np.gradient(field.get_fdata() * trt, axis=axis)
    interior = (slice(3, -3),) * 3
    assert_allclose(jac.get_fdata()[interior], expected[interior], atol=5e-3)
