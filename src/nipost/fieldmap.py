"""Fieldmap reconstruction (ported from fMRIPrep + SDCFlows)."""

from warnings import warn

import nibabel as nb
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import kron, lil_array


def grid_bspline_weights(target_nii, ctrl_nii, dtype='float32'):
    r"""
    Evaluate tensor-product B-Spline weights on a grid.

    .. _bspline-tensor:

    For each of the *N* input samples :math:`(s_1, s_2, s_3)` and *K* control
    points or *knots* :math:`\mathbf{k} =(k_1, k_2, k_3)`, the tensor-product
    cubic B-Spline kernel weights are calculated:

    .. math::

        \Psi^3(\mathbf{k}, \mathbf{s}) =
        \beta^3(s_1 - k_1) \cdot \beta^3(s_2 - k_2) \cdot \beta^3(s_3 - k_3),
        \label{eq:2}\tag{2}

    where each :math:`\beta^3` represents the cubic B-Spline for one dimension.
    The 1D B-Spline kernel implementation uses :obj:`numpy.piecewise`, and is based on the
    closed-form given by Eq. (6) of [Unser1999]_.

    By iterating over dimensions, the data samples that fall outside of the compact
    support of the tensor-product kernel associated to each control point can be filtered
    out and dismissed to lighten computation.

    Finally, the resulting weights matrix :math:`\Psi^3(\mathbf{k}, \mathbf{s})` can easily be
    identified in `Eq. (1) <sdcflows.interfaces.bspline.html#bspline-interpolation>`_,
    and used as the design matrix for approximation of data.

    Parameters
    ----------
    target_nii :  :obj:`nibabel.spatialimages`
        An spatial image object (typically, a :obj:`~nibabel.nifti1.Nifti1Image`)
        embedding the target EPI image to be corrected.
        Provides the location of the *N* samples (total number of voxels) in the space.
    ctrl_nii : :obj:`nibabel.spatialimages`
        An spatial image object (typically, a :obj:`~nibabel.nifti1.Nifti1Image`)
        embedding the location of the control points of the B-Spline grid.
        The data array should contain a total of :math:`K` knots (control points).

    Returns
    -------
    weights : :obj:`numpy.ndarray` (:math:`N \times K`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* voxels of the target EPI, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """
    sample_shape = target_nii.shape[:3]
    knots_shape = ctrl_nii.shape[:3]

    # Ensure the cross-product of affines is near zero (i.e., both coordinate systems are aligned)
    if not np.allclose(
        np.linalg.norm(
            np.cross(ctrl_nii.affine[:-1, :-1].T, target_nii.affine[:-1, :-1].T),
            axis=1,
        ),
        0,
        atol=1e-3,
    ):
        warn("Image's and B-Spline's grids are not aligned.", stacklevel=2)

    target_to_grid = np.linalg.inv(ctrl_nii.affine) @ target_nii.affine
    wd = []
    for axis in range(3):
        # 3D ijk coordinates of current axis
        coords = np.zeros((3, sample_shape[axis]), dtype=dtype)
        coords[axis] = np.arange(sample_shape[axis], dtype=dtype)

        # Calculate the index component of samples w.r.t. B-Spline knots along current axis
        # Size of locations is L
        locs = nb.affines.apply_affine(target_to_grid, coords.T)[:, axis]

        # Size of knots is K + 6 so that all locations are fully covered by basis
        knots = np.arange(-3, knots_shape[axis] + 3, dtype=dtype)

        bspl = BSpline(knots, np.eye(len(knots) - 3 - 1), 3)

        # Construct a sparse design matrix (L, K)
        distance = np.abs(locs[..., np.newaxis] - knots[np.newaxis, 3:-3])
        within_support = distance < 2.0

        colloc_ax = lil_array(distance.shape, dtype=dtype)
        colloc_ax[within_support] = bspl(locs)[:, 1:-1][within_support]

        # Convert to CSR for efficient multiplication
        wd.append(colloc_ax.tocsr())

    # Calculate the tensor product of the three design matrices
    return kron(kron(wd[0], wd[1]), wd[2])
