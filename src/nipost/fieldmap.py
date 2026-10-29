"""Fieldmap reconstruction (ported from fMRIPrep + SDCFlows)."""

from typing import overload

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


def pe_vector(
    boldref: SpatialImage,
    pe_info: list[tuple[int, float]],
    xfm: nt.base.TransformBase,
) -> np.ndarray:
    r"""Phase-encoding voxel step times signed readout time, in fieldmap world space.

    With :math:`\nabla\phi` in Hz/mm, the susceptibility Jacobian is
    :math:`J = 1 + \nabla\phi \cdot v`.

    Parameters
    ----------
    boldref
        BOLD reference, as returned by :func:`~nipost.epi.prepare_epi`. Its voxel
        axes must be those of the ``source`` given to :func:`~nipost.resample_image`,
        as ``pe_info`` indexes them.
    pe_info
        Per-volume ``(axis-index, signed-readout-time)`` from
        :func:`~nipost.epi.prepare_epi`. All entries must agree.
    xfm
        Same convention as ``transforms`` in :func:`reconstruct_fieldmap`:
        maps images from fieldmap into boldref space, so its matrix maps
        boldref points to fieldmap points. Must reduce to an affine.

    Returns
    -------
    v : :class:`~numpy.ndarray`
        Shape ``(3,)``, in mm·s. The sign encodes polarity.

    Raises
    ------
    ValueError
        If ``pe_info`` entries differ or ``xfm`` is not affine.
    """
    if len(set(pe_info)) != 1:
        raise ValueError('pe_info entries must agree')
    axis, trt = pe_info[0]

    affine = as_affine(xfm)
    if affine is None:
        raise ValueError('xfm must reduce to an affine')

    return affine.matrix[:3, :3] @ boldref.affine[:3, axis] * trt


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


@overload
def reconstruct_fieldmap(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    transforms: nt.base.TransformBase,
    *,
    pe_vector: None = None,
) -> tuple[nb.Nifti1Image, None]: ...
@overload
def reconstruct_fieldmap(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    transforms: nt.base.TransformBase,
    *,
    pe_vector: np.ndarray,
) -> tuple[nb.Nifti1Image, nb.Nifti1Image]: ...
def reconstruct_fieldmap(
    coefficients: list[SpatialImage],
    target: SpatialImage,
    transforms: nt.base.TransformBase,
    *,
    pe_vector: np.ndarray | None = None,
) -> tuple[nb.Nifti1Image, nb.Nifti1Image | None]:
    r"""Evaluate a B-Spline fieldmap at the voxels of a target space

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
    pe_vector
        A vector from :func:`pe_vector`. If given, the Jacobian is also returned.

    Returns
    -------
    fieldmap
        The fieldmap encoded in ``coefficients``, in Hz, with the shape
        (first three axes) and affine of ``target``. It is 0 beyond the support
        of the coefficients.
    jacobian
        Only if ``pe_vector`` is given: :math:`J = 1 + \nabla\phi \cdot v`, with
        :math:`v` the ``pe_vector``, unclipped, on the grid of ``target``.
        It is 1 beyond the support of the coefficients.
    """
    if pe_vector is not None:
        raise NotImplementedError
    shape = target.shape[:3]
    ijk = np.indices(shape).reshape(3, -1).T
    # Map through any warp once; each level then needs only a float64 affine
    fmap_xyz = collapse_transforms(
        nt.TransformChain([nt.Affine(target.affine, reference=target), transforms])
    ).map(ijk)

    field = np.zeros(len(ijk))
    for level in coefficients:
        lattice = nb.affines.apply_affine(np.linalg.inv(level.affine), fmap_xyz)
        # The field is exactly 0 from two knots outside the lattice
        inside = ((lattice > -2) & (lattice < np.array(level.shape[:3]) + 1)).all(axis=1)
        field[inside] += _level_spline(level)(lattice[inside])
    field = field.reshape(shape)

    fmap_img = nb.Nifti1Image(field.astype('float32'), target.affine)
    fmap_img.header.set_intent('estimate', name='fieldmap Hz')
    max_abs = max(abs(field.min()), field.max())
    fmap_img.header['cal_max'] = max_abs
    fmap_img.header['cal_min'] = -max_abs

    return fmap_img
