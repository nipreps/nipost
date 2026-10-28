# tests/test_fieldmap_reconstruct.py
from itertools import product

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nipost.fieldmap import aligned, as_affine, reconstruct_fieldmap, target_bbox_grid


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


def ramp_coefficients():
    """Coefficients set to each knot's RAS x coordinate, over the same grid.

    Cubic B-splines reproduce cubics, so the field is exactly ``f(x, y, z) = x``.
    A constant field would be blind to boundary error, which reproduces it exactly.
    """
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -40.0
    knots = nb.Nifti1Image(np.zeros((21, 21, 21), dtype='f4'), affine)
    return [nb.Nifti1Image(ras_x(knots).astype('f4'), affine)]


def ras_x(img):
    """The RAS x coordinate of every voxel of an image."""
    ijk = np.indices(img.shape[:3], dtype=float).reshape(3, -1).T
    return nb.affines.apply_affine(img.affine, ijk)[:, 0].reshape(img.shape[:3])


def grid(span, shift=0.0, zoom=2.0):
    """An isotropic grid spanning -span..+span mm, optionally shifted off the knots."""
    affine = np.diag([zoom, zoom, zoom, 1.0])
    affine[:3, 3] = -span + shift
    return nb.Nifti1Image(np.zeros((int(2 * span / zoom) + 1,) * 3, dtype='f4'), affine)


def identity_warp():
    """A zero-displacement dense field: maps identity, but is not an Affine.

    ``as_affine`` therefore returns None and the non-direct path is taken.
    """
    affine = np.diag([4.0, 4.0, 4.0, 1.0])
    affine[:3, 3] = -40.0
    field = nb.Nifti1Image(np.zeros((21, 21, 21, 3), dtype='f4'), affine)
    return nt.TransformChain([nt.DenseFieldTransform(field, is_deltas=True)])


def rotated_grid(span, angle, zoom=2.0):
    """A grid spanning -span..+span mm, rotated ``angle`` radians about the k axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    affine = np.eye(4)
    affine[:3, :3] = rotation * zoom
    affine[:3, 3] = rotation @ np.full(3, -span)
    return nb.Nifti1Image(np.zeros((int(2 * span / zoom) + 1,) * 3, dtype='f4'), affine)


def corners(img):
    """The eight voxel-centre corners of an image, in RAS."""
    ijk = np.array(list(product(*([0, n - 1] for n in img.shape[:3]))), dtype=float)
    return nb.affines.apply_affine(img.affine, ijk)


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


def test_reconstruct_fieldmap_target_edges_survive_interpolation():
    # The margin past the bounding box keeps nt.resampling.apply's boundary extension
    # away from the target's samples. The bounding box alone already guarantees no
    # sample falls outside the intermediate grid, so cval never applies; what the
    # margin buys is distance from the spline prefilter's boundary conditions, whose
    # error is signal-dependent and decays inwards. Shift the target half a voxel off
    # the fieldmap grid so resampling interpolates rather than copies.
    target = grid(span=20.0, shift=1.0)

    field = reconstruct_fieldmap(
        coefficients=ramp_coefficients(),
        fmap_reference=grid(span=10.0),
        target=target,
        transforms=identity_warp(),
    )

    assert_allclose(np.asanyarray(field.dataobj), ras_x(target), atol=0.05)


@pytest.mark.parametrize(
    ('description', 'target'),
    [
        ('aligned with the fieldmap grid', grid(span=20.0)),
        ('shifted half a voxel', grid(span=20.0, shift=1.0)),
        ('finer than the fieldmap grid', grid(span=20.0, zoom=1.0)),
        ('rotated 45 degrees', rotated_grid(span=20.0, angle=np.pi / 4)),
    ],
)
def test_target_bbox_grid_reserves_margin_around_mapped_target(description, target):
    fmap_reference = grid(span=10.0)

    bbox = target_bbox_grid(fmap_reference, target, identity_warp(), margin=4)

    ijk = nb.affines.apply_affine(np.linalg.inv(bbox.affine), corners(target))
    assert (ijk >= 4 - 1e-3).all(), f'target {description} is short of margin at the low edge'
    upper = np.array(bbox.shape[:3]) - 5
    assert (ijk <= upper + 1e-3).all(), f'target {description} is short of margin at the top'


def test_target_bbox_grid_keeps_fieldmap_reference_resolution_and_orientation():
    fmap_reference = grid(span=10.0)

    bbox = target_bbox_grid(fmap_reference, grid(span=20.0), identity_warp(), margin=4)

    assert_allclose(bbox.affine[:3, :3], fmap_reference.affine[:3, :3])


def test_target_bbox_grid_shrinks_for_target_inside_fmap_fov():
    # A partial-FOV target must not drag in the whole fieldmap reference FOV: 5 target
    # voxels plus a 4-voxel margin on each side is 13, not the reference's 21.
    bbox = target_bbox_grid(grid(span=20.0), grid(span=4.0), identity_warp(), margin=4)

    assert bbox.shape[:3] == (13, 13, 13)


def test_reconstruct_fieldmap_rejects_reference_misaligned_with_coefficients():
    # The intermediate grid inherits the reference's direction cosines, so a reference
    # rotated relative to the coefficient grids would silently break the separable
    # tensor product grid_bspline_weights() builds.
    with pytest.raises(ValueError, match='not aligned with spline grids'):
        reconstruct_fieldmap(
            coefficients=ones_coefficients(),
            fmap_reference=rotated_grid(span=10.0, angle=np.pi / 6),
            target=grid(span=20.0),
            transforms=identity_warp(),
        )


def test_reconstruct_fieldmap_canonicalizes_permuted_reference():
    # Coefficients are RAS; a reference whose voxel axes are permuted is reoriented to
    # match before its orientation is handed to the intermediate grid. A permutation is
    # what the reorientation is actually for -- a mere axis flip reads as aligned, since
    # antiparallel columns have a zero cross product, and ensure_positive_cosines()
    # would settle it regardless.
    affine = np.zeros((4, 4))
    affine[[0, 1, 2], [2, 0, 1]] = 2.0
    affine[:3, 3] = -10.0
    affine[3, 3] = 1.0
    fmap_reference = nb.Nifti1Image(np.zeros((11, 11, 11), dtype='f4'), affine)
    assert nb.aff2axcodes(affine) == ('A', 'S', 'R')

    field = reconstruct_fieldmap(
        coefficients=ones_coefficients(),
        fmap_reference=fmap_reference,
        target=grid(span=20.0),
        transforms=identity_warp(),
    )

    assert_allclose(np.asanyarray(field.dataobj), 1.0, atol=1e-4)


def test_target_bbox_grid_follows_the_transform_into_fieldmap_space():
    # nitransforms samples the moving image at transforms.map(target coordinates), so
    # the grid has to follow the transform rather than oppose it. Every other test here
    # uses a zero-displacement warp, which cannot tell the two directions apart.
    shift = np.eye(4)
    shift[0, 3] = 30.0

    bbox = target_bbox_grid(grid(span=10.0), grid(span=4.0), nt.Affine(shift), margin=2)

    lo = nb.affines.apply_affine(bbox.affine, [0, 0, 0])
    hi = nb.affines.apply_affine(bbox.affine, np.array(bbox.shape[:3]) - 1)
    # The target spans -4..+4mm; shifted +30 and padded two 2mm voxels either side.
    assert_allclose([lo[0], hi[0]], [22.0, 38.0])


def test_target_bbox_grid_pads_an_anisotropic_reference_per_axis():
    # The margin counts voxels, so an anisotropic reference pads a different number of
    # millimetres per axis. The target's extents and its origin both differ per axis:
    # equal origins would make the lower bound [0, 0, 0], which survives an axis
    # permutation unchanged and so could not detect one.
    reference = nb.Nifti1Image(np.zeros((5, 5, 5), dtype='f4'), np.diag([1.0, 2.0, 5.0, 1.0]))
    target_affine = np.diag([10.0, 20.0, 40.0, 1.0])
    target_affine[:3, 3] = [3.0, -7.0, 11.0]
    target = nb.Nifti1Image(np.zeros((2, 2, 2), dtype='f4'), target_affine)

    bbox = target_bbox_grid(reference, target, nt.Affine(), margin=1)

    assert_allclose(nb.affines.voxel_sizes(bbox.affine), [1.0, 2.0, 5.0])
    assert_allclose(nb.affines.apply_affine(bbox.affine, [0, 0, 0]), [2.0, -10.0, 5.0])
    assert_allclose(
        nb.affines.apply_affine(bbox.affine, np.array(bbox.shape[:3]) - 1),
        [14.0, 16.0, 60.0],
    )
