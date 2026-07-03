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
