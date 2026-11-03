"""Fieldmap reconstruction (ported from fMRIPrep + SDCFlows)."""

from collections.abc import Iterator

import nibabel as nb
import nitransforms as nt
import numpy as np
from nibabel.spatialimages import SpatialImage
from scipy.interpolate import NdBSpline

from nipost.transforms import collapse_transforms


def as_affine(xfm: nt.base.TransformBase) -> nt.Affine | None:
    """Reduce a transform to a single Affine, or None if it is not one"""
    xfm = collapse_transforms(xfm)
    return xfm if type(xfm) is nt.Affine else None


def _level_spline(level: SpatialImage) -> NdBSpline:
    """Build the cubic tensor-product spline of one coefficient level.

    The spline is a function of the level's lattice (voxel index) coordinates.
    """
    # Four zero coefficients per side make the outermost pieces of the spline
    # identically zero, so the field is exactly 0 beyond the support of the lattice,
    # whether or not extrapolation is attempted.
    coeffs = np.pad(level.get_fdata().reshape(level.shape[:3]), 4)
    knots = tuple(np.arange(-6, n + 6, dtype='f8') for n in level.shape[:3])
    return NdBSpline(knots, coeffs, 3)


def _level_points(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    transforms: nt.base.TransformBase,
) -> Iterator[tuple[SpatialImage, np.ndarray, np.ndarray]]:
    """Yield, per level, the target voxels it covers and their lattice coordinates.

    Yields ``(level, inside, lattice[inside])``: ``inside`` is a boolean mask over the
    target voxels, and the coordinates are in the level's voxel index space.
    """
    shape = target.shape[:3]
    targetvox2fmap = collapse_transforms(
        nt.TransformChain([nt.Affine(target.affine, reference=target), transforms])
    )
    fmap_xyz = targetvox2fmap.map(np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)

    for level in coefficients:
        lattice = nb.affines.apply_affine(np.linalg.inv(level.affine), fmap_xyz)
        # The field is exactly 0 from two knots outside the lattice
        inside = ((lattice > -2) & (lattice < np.array(level.shape[:3]) + 1)).all(axis=-1)
        yield level, inside, lattice[inside]


def reconstruct_fieldmap(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    transforms: nt.base.TransformBase,
) -> nb.Nifti1Image:
    """Evaluate a B-Spline fieldmap at the voxels of a target space

    Each coefficient level is a cubic B-Spline over its own lattice. The levels
    are evaluated exactly at the center of every target voxel, and summed.
    There is no intermediate grid and no resampling, so the transform may be
    anything nitransforms can map points with, and the target's orientation
    and resolution are unconstrained.

    Parameters
    ----------
    coefficients
        list of B-spline coefficient images. The affine matrices are used
        to reconstruct the knot locations. Levels may have different
        spacings and orientations, and their order is immaterial.
    target
        The space to evaluate the fieldmap in. Only the first three axes are
        used, so a time series is a valid target.
    transforms
        A nitransforms transform that maps points in the target space
        into the fieldmap space, i.e., the one that resamples fieldmap images
        into the target space.

    Returns
    -------
    fieldmap
        The fieldmap encoded in ``coefficients``, in Hz, with the shape
        (first three axes) and affine of ``target``. It is 0 beyond the support
        of the coefficients.
    """
    field = np.zeros(target.shape[:3])
    for level, inside, lattice in _level_points(coefficients, target, transforms):
        field[inside] += _level_spline(level)(lattice)

    fmap_img = nb.Nifti1Image(field.astype('float32'), target.affine)
    fmap_img.header.set_intent('estimate', name='fieldmap Hz')
    max_abs = max(abs(field.min()), field.max())
    fmap_img.header['cal_max'] = max_abs
    fmap_img.header['cal_min'] = -max_abs

    return fmap_img


def reconstruct_gradient(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    fmap2target: nt.base.TransformBase,
    fmap2epi: nt.base.TransformBase,
) -> nb.Nifti1Image:
    """Evaluate the analytic gradient of a B-Spline fieldmap at the voxels of a target

    Evaluates the row vector :math:`\\nabla_F f\\,M`, where :math:`f` is the fieldmap
    summed over the coefficient levels, :math:`\\nabla_F f` is its gradient in fieldmap
    space, and :math:`M` is the linear part of ``fmap2epi``. The result is the gradient
    of the field along the EPI world axes, in Hz/mm. The Jacobian of ``fmap2target``
    itself is not applied.

    Parameters
    ----------
    coefficients
        list of B-spline coefficient images, as for :func:`reconstruct_fieldmap`.
    target
        The space to evaluate the gradient in. Only the first three axes are used.
    fmap2target
        A nitransforms transform that maps points in the target space into the
        fieldmap space, i.e., the one that resamples fieldmap images into the target space.
    fmap2epi
        The transform that resamples fieldmap images into the EPI (boldref) space,
        i.e., mapping EPI points into the fieldmap space. It must reduce to an affine.

    Returns
    -------
    gradient
        The gradient in Hz/mm, as a float32 vector image of shape
        ``(*target.shape[:3], 1, 3)`` with the affine of ``target``. It is 0 beyond
        the support of the coefficients.

    Raises
    ------
    ValueError
        If ``fmap2epi`` does not reduce to an affine.
    """
    if (fmap2epi := as_affine(fmap2epi)) is None:
        raise ValueError('fmap2epi must reduce to an affine')

    grad = np.zeros((*target.shape[:3], 1, 3))
    for level, inside, lattice in _level_points(coefficients, target, fmap2target):
        spline = _level_spline(level)
        # The gradient is a row vector at each lattice point
        partials = np.stack([spline(lattice, nu=nu) for nu in np.eye(3, dtype=int)], axis=1)
        # Right-applied to gradients in lattice space to get gradients in EPI world space
        lattice2epi = np.linalg.inv(level.affine[:3, :3]) @ fmap2epi.matrix[:3, :3]
        # The sum of gradients is the gradient of the sum
        grad[inside, 0] += partials @ lattice2epi

    img = nb.Nifti1Image(grad.astype('float32'), target.affine)
    img.header.set_intent('vector', name='fmap grad Hz/mm')
    return img
