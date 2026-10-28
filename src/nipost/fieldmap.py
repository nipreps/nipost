"""Fieldmap reconstruction (ported from fMRIPrep + SDCFlows)."""

from warnings import warn

import nibabel as nb
import nitransforms as nt
import numpy as np
from nibabel.spatialimages import SpatialImage
from scipy.interpolate import BSpline
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import kron, lil_array

from nipost.epi import ensure_positive_cosines

# Tolerance, in voxels, for snapping a bounding box bound to a voxel center
SNAP_TOL = 1e-3


def grid_bspline_weights(target_nii: SpatialImage, ctrl_nii: SpatialImage, dtype='float32'):
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
    target_nii :  :obj:`nibabel.spatialimages.SpatialImage`
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

        # Size of knots is K + 10, so that the spline's base interval -- the middle
        # of the knot vector, [knots[3], knots[-4]] -- spans -2..K+1 and thus covers
        # the whole compact support of the K basis functions centred on 0..K-1.
        knots = np.arange(-5, knots_shape[axis] + 5, dtype=dtype)

        # We are calculating weights exactly; extrapolation would do the wrong thing
        # and we'd rather see NaNs if we get this wrong.
        bspl = BSpline(knots, np.eye(len(knots) - 3 - 1), 3, extrapolate=False)

        # Construct a sparse design matrix (L, K)
        distance = np.abs(locs[..., np.newaxis] - knots[np.newaxis, 5:-5])
        within_support = distance < 2.0

        colloc_ax = lil_array(distance.shape, dtype=dtype)
        colloc_ax[within_support] = bspl(locs)[:, 3:-3][within_support]

        # Convert to CSR for efficient multiplication
        wd.append(colloc_ax.tocsr())

    # Calculate the tensor product of the three design matrices
    return kron(kron(wd[0], wd[1]), wd[2])


def aligned(aff1: np.ndarray, aff2: np.ndarray) -> bool:
    """Determine if two affines have aligned grids"""
    return np.allclose(
        np.linalg.norm(np.cross(aff1[:-1, :-1].T, aff2[:-1, :-1].T), axis=1),
        0,
        atol=1e-3,
    )


def as_affine(xfm: nt.base.TransformBase) -> nt.Affine | None:
    # Identity transform
    if type(xfm) is nt.base.TransformBase:
        return nt.Affine()

    if isinstance(xfm, nt.Affine):
        return xfm

    if isinstance(xfm, nt.TransformChain) and all(isinstance(x, nt.Affine) for x in xfm):
        return xfm.asaffine()

    return None


def target_bbox_grid(
    fmap_reference: SpatialImage,
    target: SpatialImage,
    transforms: nt.base.TransformBase,
    margin: int = 4,
) -> nb.Nifti1Image:
    """Build a grid covering the target's bounding box, aligned with the fieldmap.

    Keeps ``fmap_reference``'s resolution and direction cosines, so the grid stays
    aligned with the coefficient grids, and takes its origin and shape from
    ``target`` mapped through ``transforms``.

    Assumes a homeomorphic transform and only maps faces, for efficiency.

    Parameters
    ----------
    fmap_reference
        Donates the grid's resolution and orientation.
    target
        The space whose extent the grid must cover.
    transforms
        Maps coordinates in the target space into the fieldmap space.
    margin
        Voxels to leave on every side, keeping the resampling prefilter's boundary
        conditions away from the target's samples.

    Returns
    -------
    grid
        A zero-filled image; only its shape and affine are meaningful.

    Examples
    --------
    A target five 2mm voxels across, with two voxels of margin either side:

    >>> ref = nb.Nifti1Image(np.zeros((11, 11, 11)), np.diag([2.0, 2.0, 2.0, 1.0]))
    >>> target = nb.Nifti1Image(np.zeros((5, 5, 5)), np.diag([2.0, 2.0, 2.0, 1.0]))
    >>> target_bbox_grid(ref, target, nt.Affine(), margin=2).shape
    (9, 9, 9)
    """
    # Get the edge voxels of the target
    mask = np.ones(target.shape[:3], dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = False

    vox2vox = nt.TransformChain(
        [nt.Affine(target.affine), transforms, ~nt.Affine(fmap_reference.affine)]
    )
    ijk = vox2vox.map(np.argwhere(mask))

    # Round outwards, snapping near-integer bounds: a co-aligned target lands on exact
    # indices, where numerical noise would otherwise flip floor/ceil by a whole voxel.
    start = np.floor(ijk.min(axis=0) + SNAP_TOL).astype(int) - margin
    end = np.ceil(ijk.max(axis=0) - SNAP_TOL).astype(int) + margin

    affine = fmap_reference.affine @ nb.affines.from_matvec(np.eye(3), start)
    return nb.Nifti1Image(np.zeros(end - start + 1, dtype='u1'), affine)


def reconstruct_fieldmap(
    coefficients: list[SpatialImage],
    fmap_reference: SpatialImage,
    target: SpatialImage,
    transforms: nt.TransformChain,
) -> nb.Nifti1Image:
    """Resample a fieldmap from B-Spline coefficients into a target space

    If the coefficients and target are aligned, the field is reconstructed
    directly in the target space.
    If not, then the field is reconstructed to the ``fmap_reference``
    resolution, and then resampled according to transforms.

    The former method only applies if the transform chain can be
    collapsed to a single affine transform.

    Parameters
    ----------
    coefficients
        list of B-spline coefficient files. The affine matrices are used
        to reconstruct the knot locations.
    fmap_reference
        The intermediate reference to reconstruct the fieldmap in, if
        it cannot be reconstructed directly in the target space.
    target
        The target space to to resample the fieldmap into.
    transforms
        A nitransforms TransformChain that maps images from the fieldmap
        space into the target space.

    Returns
    -------
    fieldmap
        The fieldmap encoded in ``coefficients``, resampled in the same
        space as ``target``
    """

    direct = False
    affine_xfm = as_affine(transforms)
    if affine_xfm is not None:
        # Transforms maps RAS coordinates in the target to RAS coordinates in
        # the fieldmap space. Composed with target.affine, we have a target voxel
        # to fieldmap RAS affine. Hence, this is projected into fieldmap space.
        projected_affine = affine_xfm.matrix @ target.affine
        # If the coordinates have the same rotation from voxels, we can construct
        # bspline weights efficiently.
        direct = aligned(projected_affine, coefficients[-1].affine)

    if direct:
        reference, _ = ensure_positive_cosines(
            target.__class__(target.dataobj, projected_affine, target.header),
        )
    else:
        # Hack. Sometimes the reference array is rotated relative to the fieldmap
        # and coefficient grids. As far as I know, coefficients are always RAS,
        # but good to check before doing this.
        if (
            nb.aff2axcodes(coefficients[-1].affine)
            == ('R', 'A', 'S')
            != nb.aff2axcodes(fmap_reference.affine)
        ):
            fmap_reference = nb.as_closest_canonical(fmap_reference)
        if not aligned(fmap_reference.affine, coefficients[-1].affine):
            raise ValueError('Reference passed is not aligned with spline grids')
        reference, _ = ensure_positive_cosines(fmap_reference)

    # Generate tensor-product B-Spline weights
    colmat = sparse_hstack(
        [grid_bspline_weights(reference, level) for level in coefficients]
    ).tocsr()
    coeff_data: np.ndarray = np.hstack(
        [level.get_fdata(dtype='float32').reshape(-1) for level in coefficients]
    )

    # Reconstruct the fieldmap (in Hz) from coefficients
    fmap_img = nb.Nifti1Image(
        np.reshape(colmat @ coeff_data, reference.shape[:3]),
        reference.affine,
    )

    if not direct:
        fmap_img = nt.resampling.apply(transforms, fmap_img, reference=target)

    fmap_img.header.set_intent('estimate', name='fieldmap Hz')
    fmap_img.header.set_data_dtype('float32')
    fmap_img.header['cal_max'] = max((abs(fmap_img.dataobj.min()), fmap_img.dataobj.max()))  # type: ignore[attr-defined]
    fmap_img.header['cal_min'] = -fmap_img.header['cal_max']

    return fmap_img
