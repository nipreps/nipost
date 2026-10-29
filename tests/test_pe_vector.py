import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nipost import pe_vector, prepare_epi
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
