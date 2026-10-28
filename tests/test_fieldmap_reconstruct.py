# tests/test_fieldmap_reconstruct.py
import nibabel as nb
import nitransforms as nt
import numpy as np
from numpy.testing import assert_allclose

from nipost.fieldmap import aligned, as_affine, reconstruct_fieldmap


def test_aligned_true_for_same_orientation():
    assert aligned(np.eye(4), np.diag([2.0, 2.0, 2.0, 1.0]))


def test_aligned_false_for_rotated():
    rot = np.eye(4)
    rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert not aligned(np.eye(4), rot)


def test_as_affine_collapses_identity_chain():
    # nt.TransformChain is in nt.manip; nt.base.TransformChain does not exist in 25.x
    chain = nt.TransformChain([nt.Affine(), nt.Affine()])
    assert isinstance(as_affine(chain), nt.Affine)


def test_reconstruct_fieldmap_direct_grid(tmp_path):
    # target aligned with coefficient grid -> direct reconstruction path
    target = nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4))
    fmapref = nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4))
    coeff = nb.Nifti1Image(np.ones((5, 5, 5), dtype='f4'), np.diag([2.0, 2.0, 2.0, 1.0]))

    field = reconstruct_fieldmap(
        coefficients=[coeff],
        fmap_reference=fmapref,
        target=target,
        transforms=nt.TransformChain([nt.Affine()]),
    )
    assert field.shape == target.shape
    assert np.isfinite(np.asanyarray(field.dataobj)).all()


# Cubic B-splines form a partition of unity, so all-ones coefficients reconstruct to
# exactly 1.0 Hz over the coefficient grid's interior -- -36..+36mm for the 21 knots of
# 4mm below. That contains every target here plus the intermediate grid built around it,
# margin included, which is what makes 1.0 Hz a hand-derived expectation.
def ones_coefficients():
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -40.0
    return [nb.Nifti1Image(np.ones((21, 21, 21), dtype='f4'), affine)]


def grid(span, shift=0.0, zoom=2.0):
    """An isotropic grid spanning -span..+span mm, optionally shifted off the knots."""
    affine = np.diag([zoom, zoom, zoom, 1.0])
    affine[:3, 3] = -span + shift
    return nb.Nifti1Image(np.zeros((int(2 * span / zoom) + 1,) * 3, dtype='f4'), affine)


def test_reconstruct_fieldmap_is_zero_past_the_coefficient_support():
    # All-ones coefficients creates a field with a max of 1Hz, falling off to 0Hz
    # beyond two knot spacings past the outermost knot.
    target = grid(span=60.0)

    field = reconstruct_fieldmap(
        coefficients=ones_coefficients(),
        fmap_reference=grid(span=10.0),
        target=target,
        # An identity transform avoids the reconstruct-then-resample path
        transforms=nt.TransformChain([nt.Affine()]),
    )

    data = np.asanyarray(field.dataobj)
    assert data.min() >= -1e-6, f'field undershoots the coefficients: {data.min()}'
    assert data.max() <= 1 + 1e-6, f'field overshoots the coefficients: {data.max()}'

    ijk = np.indices(target.shape[:3], dtype=float).reshape(3, -1).T
    ras = nb.affines.apply_affine(target.affine, ijk)
    # 48mm is 2 knot spacings beyond the outermost knot of the 21 knots spanning -40..+40mm
    outside = (np.abs(ras) > 48.0).any(axis=1).reshape(target.shape[:3])
    assert_allclose(data[outside], 0.0, atol=1e-6)
