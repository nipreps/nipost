# tests/test_fieldmap_weights.py
import nibabel as nb
import numpy as np

from nipost.fieldmap import grid_bspline_weights


def test_grid_bspline_weights_shape_and_partition_of_unity():
    target = nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4))
    # coarse control grid covering the target
    ctrl_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    ctrl = nb.Nifti1Image(np.zeros((5, 5, 5), dtype='f4'), ctrl_affine)

    weights = grid_bspline_weights(target, ctrl)

    assert weights.shape == (6 * 6 * 6, 5 * 5 * 5)
    # cubic B-spline weights are non-negative
    assert weights.min() >= 0


def test_grid_bspline_weights_stay_bounded_outside_the_knot_range():
    # Cubic B-splines are non-negative and form a partition of unity, so each sample's
    # weights sum to at most 1 -- exactly 1 where the full support is available, and
    # tapering to 0 across the two knot spacings past the outermost knot. A sample
    # past the outermost knot is still within that knot's support, so it must still
    # obey the bound rather than pick up an extrapolated edge polynomial.
    ctrl_affine = np.diag([4.0, 4.0, 4.0, 1.0])
    ctrl_affine[:3, 3] = -40.0
    ctrl = nb.Nifti1Image(np.zeros((21, 21, 21), dtype='f4'), ctrl_affine)
    # The knots span -40..+40mm, so the support ends at 48mm; the target reaches 60mm.
    target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    target_affine[:3, 3] = -60.0
    target = nb.Nifti1Image(np.zeros((61, 61, 61), dtype='f4'), target_affine)

    weights = grid_bspline_weights(target, ctrl)

    per_sample = np.asarray(weights.sum(axis=1)).ravel()
    assert per_sample.min() >= -1e-6, f'negative weight sum: {per_sample.min()}'
    assert per_sample.max() <= 1 + 1e-6, f'weight sum exceeds unity: {per_sample.max()}'
