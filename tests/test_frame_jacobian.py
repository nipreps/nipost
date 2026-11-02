import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from nibabel.spatialimages import SpatialImage
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from nipost import prepare_epi, reconstruct_fieldmap, reconstruct_gradient, resample_image
from tests.test_fieldmap_reconstruct import ramp_coefficients, rot_z

_fmap2epi = rot_z(-0.1)
_fmap2epi[:3, 3] = [3.0, -2.0, 1.5]
FMAP2EPI = nt.Affine(_fmap2epi)

# A RAS 2 mm grid centered on 0 mm, where the ramp's gradient is 2 Hz per voxel along i
RAS_2MM = nb.affines.from_matvec(np.eye(3) * 2.0, [-9.0, -9.0, -9.0])


def smooth_coefficients() -> list[SpatialImage]:
    """#7's smooth random field on an oblique 10 mm lattice (seed 0, x400)."""
    rng = np.random.default_rng(0)
    lattice = rot_z(0.3) @ nb.affines.from_matvec(np.diag([10.0, 9.0, 11.0]), [-70, -70, -70])
    smooth = gaussian_filter(rng.normal(size=(15, 16, 14)), 1.0) * 400
    return [nb.Nifti1Image(smooth.astype('f4'), lattice)]


def epi(affine, shape, code='j', frames=2):
    """A constant series through prepare_epi, TotalReadoutTime 0.05.

    prepare_epi needs 4-D input: for a 3-D source, use ``bold.slicer[..., 0]`` and
    ``pe_info[:1]``.
    """
    img = nb.Nifti1Image(np.ones((*shape, frames), dtype='f4'), affine)
    return prepare_epi(img, {'PhaseEncodingDirection': code, 'TotalReadoutTime': 0.05})


def frame_jacobians(bold, pe_info, coefficients, fmap2epi, motion=None):
    """Resample a constant ``bold`` onto its first frame's grid; returns (J_t data, field).

    ``motion``: per-frame world 4x4s, passed as a LinearTransformsMapping; None for no HMC.
    Uses reconstruct_fieldmap and reconstruct_gradient with fmap2target = fmap2epi
    (the target is the boldref grid), order=1, mode='nearest'.
    """
    boldref = bold.slicer[..., 0] if bold.ndim == 4 else bold
    field = reconstruct_fieldmap(coefficients, boldref, fmap2epi)
    gradient = reconstruct_gradient(coefficients, boldref, fmap2epi, fmap2epi)
    transforms = (
        nt.Affine()
        if motion is None
        else nt.linear.LinearTransformsMapping([nt.Affine(m) for m in motion])
    )
    out = resample_image(
        source=bold,
        target=boldref,
        transforms=transforms,
        fieldmap=field,
        pe_info=pe_info,
        fieldmap_gradient=gradient,
        order=1,
        mode='nearest',
    )
    return np.asanyarray(out.dataobj), field.get_fdata()


def oblique_epi(code='j', flipped=False):
    """#7's 1 mm boldref, rotated about z, optionally stored with i reversed."""
    affine = rot_z(0.25) @ nb.affines.from_matvec(np.eye(3), [-15, -15, -12])
    if flipped:
        affine = affine @ nb.affines.from_matvec(np.diag([-1, 1, 1]), [29, 0, 0])
    return epi(affine, (30, 30, 24), code)


@pytest.mark.parametrize(
    ('code', 'flipped'),
    [
        pytest.param('j', False, id='j'),
        pytest.param('j-', False, id='j-'),
        pytest.param('i', True, id='i, stored flipped'),
        pytest.param('i-', True, id='i-, stored flipped'),
    ],
)
def test_frame_jacobian_without_motion(code, flipped):
    bold, pe_info = oblique_epi(code, flipped)
    axis, trt = pe_info[0]

    jac, field = frame_jacobians(bold, pe_info, smooth_coefficients(), FMAP2EPI)

    expected = 1 + np.gradient(field * trt, axis=axis)
    interior = (slice(3, -3),) * 3
    for frame in np.moveaxis(jac, -1, 0):
        assert_allclose(frame[interior], expected[interior], atol=5e-3)


def test_frame_jacobian_ignores_translation():
    bold, pe_info = oblique_epi()
    shift = nb.affines.from_matvec(np.eye(3), [1.5, -2.0, 0.7])

    jac, _ = frame_jacobians(bold, pe_info, smooth_coefficients(), FMAP2EPI, [np.eye(4), shift])

    assert_allclose(jac[..., 1], jac[..., 0], atol=1e-5)


def test_frame_jacobian_ignores_rotation_about_pe_axis():
    bold, pe_info = oblique_epi()
    pe_axis = bold.affine[:3, pe_info[0][0]]
    rotvec = np.deg2rad(5) * pe_axis / np.linalg.norm(pe_axis)
    turn = nb.affines.from_matvec(Rotation.from_rotvec(rotvec).as_matrix(), np.zeros(3))

    jac, _ = frame_jacobians(bold, pe_info, smooth_coefficients(), FMAP2EPI, [np.eye(4), turn])

    assert_allclose(jac[..., 1], jac[..., 0], atol=1e-5)


def test_frame_jacobian_matches_determinant_of_frame_map():
    # Oblique anisotropic voxels and a tilted fmap2epi, so the gradient has a component
    # along the 4 mm axis; about z alone, the transpose variant would coincide.
    affine = rot_z(0.4) @ nb.affines.from_matvec(np.diag([2.0, 2.0, 4.0]), [-10, -10, -12])
    bold, pe_info = epi(affine, (10, 10, 6), 'i')
    axis, trt = pe_info[0]
    M = nb.affines.from_matvec(
        Rotation.from_euler('xyz', [0, 30, 20], degrees=True).as_matrix(), [2, -1, 0]
    )
    H = nb.affines.from_matvec(
        Rotation.from_euler('xyz', [20, -15, 10], degrees=True).as_matrix(), [1, 2, -1]
    )

    jac, _ = frame_jacobians(bold, pe_info, ramp_coefficients(), nt.Affine(M), [np.eye(4), H])

    A = bold.affine
    Ht = (np.linalg.inv(A) @ H @ A)[:3, :3]
    g = A[:3, :3].T @ M[:3, :3].T @ [1.0, 0.0, 0.0]  # Hz per source voxel
    e_axis = np.eye(3)[axis]
    expected = np.linalg.det(Ht + trt * np.outer(e_axis, g)) / np.linalg.det(Ht)
    assert_allclose(jac[..., 1], expected, atol=1e-5)
    # The geometry separates the correct formula from the static and transposed ones
    assert abs(expected - (1 + trt * g[axis])) > 1e-3
    assert abs(expected - (1 + trt * g @ Ht.T[:, axis])) > 1e-3


def test_frame_jacobian_per_frame_pe_info():
    bold, _ = epi(RAS_2MM, (10, 10, 10), frames=3)
    pe_info = [(0, 0.05), (0, -0.05), (1, 0.05)]

    jac, _ = frame_jacobians(bold, pe_info, ramp_coefficients(), nt.Affine())

    assert_allclose(jac, np.broadcast_to([1.1, 0.9, 1.0], jac.shape), atol=1e-5)


def test_frame_jacobian_fold_over():
    bold, _ = epi(RAS_2MM, (10, 10, 10))
    pe_info = [(0, 0.05), (0, -0.75)]  # J = 1 - 0.75 * 2 = -0.5 in the second frame

    jac, _ = frame_jacobians(bold, pe_info, ramp_coefficients(), nt.Affine())

    assert_allclose(jac, np.broadcast_to([1.1, 0.0], jac.shape), atol=1e-5)


def test_resample_image_single_volume():
    bold, pe_info = epi(RAS_2MM, (10, 10, 10), 'i')
    trt = pe_info[0][1]

    jac, _ = frame_jacobians(bold.slicer[..., 0], pe_info[:1], ramp_coefficients(), nt.Affine())

    assert jac.shape == (10, 10, 10)
    assert_allclose(jac, 1 + 2 * trt, atol=1e-5)


def test_resample_image_without_gradient():
    bold, pe_info = epi(RAS_2MM, (10, 10, 10), 'i')
    boldref = bold.slicer[..., 0]

    out = resample_image(
        source=bold,
        target=boldref,
        transforms=nt.Affine(),
        fieldmap=reconstruct_fieldmap(ramp_coefficients(), boldref, nt.Affine()),
        pe_info=pe_info,
        order=1,
        mode='nearest',
    )

    assert_allclose(np.asanyarray(out.dataobj), 1.0, atol=1e-6)


def test_resample_image_rejects_gradient_without_fieldmap():
    bold, pe_info = epi(RAS_2MM, (10, 10, 10), 'i')
    boldref = bold.slicer[..., 0]
    gradient = reconstruct_gradient(ramp_coefficients(), boldref, nt.Affine(), nt.Affine())

    with pytest.raises(ValueError, match='fieldmap_gradient requires fieldmap'):
        resample_image(
            source=bold,
            target=boldref,
            transforms=nt.Affine(),
            fieldmap=None,
            pe_info=pe_info,
            fieldmap_gradient=gradient,
        )
