"""In-process resampling of BOLD series (ported from fMRIPrep)."""

from __future__ import annotations

import asyncio
import os
from functools import partial

import nibabel as nb
import nitransforms as nt
import numpy as np
from scipy import ndimage as ndi

from nipost._async import worker

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Literal

    from nibabel.spatialimages import SpatialImage

    InterpolationOrder = Literal[0, 1, 2, 3, 4, 5]
    InterpolationMode = Literal[
        'reflect',
        'grid-mirror',
        'constant',
        'grid-constant',
        'nearest',
        'mirror',
        'wrap',
        'grid-wrap',
    ]


def resample_vol(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: tuple[int, float],
    fieldmap_gradient: np.ndarray | None,
    hmc_xfm: np.ndarray | None,
    fmap_hz: np.ndarray,
    output: np.dtype | np.ndarray | None = None,
    order: InterpolationOrder = 3,
    mode: InterpolationMode = 'grid-constant',
    cval: float = 0.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Resample a volume at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``
        The first dimension should have length ``data.ndim``. The further
        dimensions have the shape of the target array.
    pe_info
        The readout vector in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    fieldmap_gradient
        The gradient of the fieldmap with respect to the source voxel index, in Hz
        per voxel, sampled at ``coordinates``, with shape ``(*coordinates.shape[1:], 3)``.
        Used to calculate the Jacobian determinant of the susceptibility distortion
        in the given volume.
        If ``None``, Jacobian modulation is not applied.
    hmc_xfm
        Affine transformation accounting for head motion from the individual
        volume into the BOLD reference space. This affine must be in VOX2VOX
        form.
    fmap_hz
        The fieldmap, sampled to the target space (at ``coordinates``), in Hz
    output
        The dtype or a pre-allocated array for sampling into the target space.
        If pre-allocated, ``output.shape == coordinates.shape[1:]``.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:]``.
    """
    if hmc_xfm is not None:
        # Move image with the head
        coords_shape = coordinates.shape
        coordinates = nb.affines.apply_affine(
            hmc_xfm, coordinates.reshape(coords_shape[0], -1).T
        ).T.reshape(coords_shape)
    else:
        # Copy coordinates to avoid interfering with other calls
        coordinates = coordinates.copy()

    vsm = fmap_hz * pe_info[1]
    coordinates[pe_info[0], ...] += vsm

    result = ndi.map_coordinates(
        data,
        coordinates,
        output=output,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    if fieldmap_gradient is not None:
        axis, trt = pe_info
        # Pull the PE step back into the reference pose: J = 1 + trt * g . inv(H)[:, axis]
        step = np.eye(3)[axis] if hmc_xfm is None else np.linalg.inv(hmc_xfm[:3, :3])[:, axis]
        jac = 1 + trt * (fieldmap_gradient @ step.astype('f4'))
        result *= np.clip(jac, 0, None)

    return result


async def resample_series_async(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: list[tuple[int, float]],
    fieldmap_gradient: np.ndarray | None,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
    output_dtype: np.dtype | str | None = None,
    order: InterpolationOrder = 3,
    mode: InterpolationMode = 'grid-constant',
    cval: float = 0.0,
    prefilter: bool = True,
    max_concurrent: int = min(os.cpu_count(), 12),  # type: ignore[type-var,assignment]
) -> np.ndarray:
    """Resample a 4D time series at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``.
        The first dimension should have length 3.
        The further dimensions determine the shape of the target array.
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    fieldmap_gradient
        The gradient of the fieldmap with respect to the source voxel index, in Hz
        per voxel, with shape ``(*coordinates.shape[1:], 3)``. If given, each volume
        is multiplied by its own Jacobian; see :func:`resample_vol`.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.
    max_concurrent
        Maximum number of volumes to resample concurrently

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:] + (N,)``,
        where N is the number of volumes in ``data``.
    """
    output: np.dtype | None = np.dtype(output_dtype) if output_dtype is not None else None
    if data.ndim == 3:
        return resample_vol(
            data,
            coordinates,
            pe_info[0],
            fieldmap_gradient,
            hmc_xfms[0] if hmc_xfms else None,
            fmap_hz,
            output,
            order,
            mode,
            cval,
            prefilter,
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    # Order F ensures individual volumes are contiguous in memory
    # Also matches NIfTI, making final save more efficient
    out_array = np.zeros(coordinates.shape[1:] + data.shape[-1:], dtype=output, order='F')

    tasks = [
        asyncio.create_task(
            worker(
                partial(
                    resample_vol,
                    data=volume,
                    coordinates=coordinates,
                    pe_info=pe_info[volid],
                    fieldmap_gradient=fieldmap_gradient,
                    hmc_xfm=hmc_xfms[volid] if hmc_xfms else None,
                    fmap_hz=fmap_hz,
                    output=out_array[..., volid],
                    order=order,
                    mode=mode,
                    cval=cval,
                    prefilter=prefilter,
                ),
                semaphore,
            )
        )
        for volid, volume in enumerate(np.rollaxis(data, -1, 0))
    ]

    await asyncio.gather(*tasks)

    return out_array


def resample_series(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: list[tuple[int, float]],
    fieldmap_gradient: np.ndarray | None,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
    output_dtype: np.dtype | str | None = None,
    order: InterpolationOrder = 3,
    mode: InterpolationMode = 'grid-constant',
    cval: float = 0.0,
    prefilter: bool = True,
    nthreads: int = 1,
) -> np.ndarray:
    """Resample a 4D time series at specified coordinates

    This function implements simultaneous head-motion correction and
    susceptibility-distortion correction. It accepts coordinates in
    the source voxel space. It is the responsibility of the caller to
    transform coordinates from any other target space.

    Parameters
    ----------
    data
        The data array to resample
    coordinates
        The first-approximation voxel coordinates to sample from ``data``.
        The first dimension should have length 3.
        The further dimensions determine the shape of the target array.
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    fieldmap_gradient
        The gradient of the fieldmap with respect to the source voxel index, in Hz
        per voxel, with shape ``(*coordinates.shape[1:], 3)``. If given, each volume
        is multiplied by its own Jacobian; see :func:`resample_vol`.
    hmc_xfm
        A sequence of affine transformations accounting for head motion from
        the individual volume into the BOLD reference space.
        These affines must be in VOX2VOX form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.
    nthreads
        Number of threads to use for parallel resampling

    Returns
    -------
    resampled_array
        The resampled array, with shape ``coordinates.shape[1:] + (N,)``,
        where N is the number of volumes in ``data``.
    """
    return asyncio.run(
        resample_series_async(
            data=data,
            coordinates=coordinates,
            pe_info=pe_info,
            fieldmap_gradient=fieldmap_gradient,
            hmc_xfms=hmc_xfms,
            fmap_hz=fmap_hz,
            output_dtype=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
            max_concurrent=nthreads,
        )
    )


def resample_image[T: SpatialImage](
    source: SpatialImage,
    target: T,
    transforms: nt.TransformChain,
    fieldmap: SpatialImage | None,
    pe_info: list[tuple[int, float]] | None,
    fieldmap_gradient: SpatialImage | None = None,
    nthreads: int = 1,
    output_dtype: np.dtype | str | None = 'f4',
    order: InterpolationOrder = 3,
    mode: InterpolationMode = 'grid-constant',
    cval: float = 0.0,
    prefilter: bool = True,
) -> T:
    r"""Resample a 3- or 4D image into a target space, applying head-motion
    and susceptibility-distortion correction simultaneously.

    Parameters
    ----------
    source
        The 3D bold image or 4D bold series to resample.
    target
        An image sampled in the target space.
    transforms
        A nitransforms TransformChain that maps images from the individual
        BOLD volume space into the target space.
    fieldmap
        The fieldmap, in Hz, sampled in the target space
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    fieldmap_gradient
        The gradient of the fieldmap in Hz/mm, sampled in the target space, from
        :func:`~nipost.fieldmap.reconstruct_gradient`, whose ``fmap2epi`` maps
        ``source`` world points to fieldmap points.
        Each volume is multiplied by its own Jacobian,
        :math:`J_t = 1 + \tau_t\,\mathbf g\cdot(\tilde H_t^{-1})_{[:,p_t]}`,
        clipped at 0, where :math:`\mathbf g` is the gradient with respect to the
        source voxel index, :math:`\tilde H_t` the volume's head motion in voxel
        coordinates, and :math:`(p_t, \tau_t)` its ``pe_info``.
        If ``None``, no modulation is applied. Requires ``fieldmap``.
    nthreads
        Number of threads to use for parallel resampling
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.

    Returns
    -------
    resampled_bold
        The BOLD series resampled into the target space
    """
    if fieldmap_gradient is not None and fieldmap is None:
        raise ValueError('fieldmap_gradient requires fieldmap')

    if not isinstance(transforms, nt.TransformChain):
        transforms = nt.TransformChain([transforms])
    if isinstance(transforms[-1], nt.linear.LinearTransformsMapping):
        transform_list, hmc = transforms[:-1], transforms[-1]
    else:
        if any(isinstance(xfm, nt.linear.LinearTransformsMapping) for xfm in transforms):
            classes = [xfm.__class__.__name__ for xfm in transforms]
            raise ValueError(f'HMC transforms must come last. Found sequence: {classes}')
        transform_list: list = transforms.transforms  # type: ignore[no-redef]
        hmc = []

    # Retrieve the RAS coordinates of the target space
    coordinates = nt.base.SpatialReference.factory(target).ndcoords.astype('f4')

    # We will operate in voxel space, so get the source affine
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)
    # Transform RAS2RAS head motion transforms to VOX2VOX
    hmc_xfms = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc]

    # After removing the head-motion transforms, add a mapping from boldref
    # world space to voxels. This new transform maps from world coordinates
    # in the target space to voxel coordinates in the source space.
    ref2vox = nt.TransformChain(transform_list + [nt.Affine(ras2vox)])
    mapped_coordinates = ref2vox.map(coordinates)

    # Some identities to reduce special casing downstream
    if fieldmap is None:
        fieldmap = nb.Nifti1Image(np.zeros(target.shape[:3], dtype='f4'), target.affine)
    if pe_info is None:
        pe_info = [(0, 0) for _ in range(source.shape[-1])]
    # Hz/mm along the source world axes -> Hz per source voxel
    gradient = None
    if fieldmap_gradient is not None:
        g_mm = fieldmap_gradient.get_fdata(dtype='f4')[..., 0, :]
        gradient = g_mm @ vox2ras[:3, :3].astype('f4')

    resampled_data = resample_series(
        data=source.get_fdata(dtype='f4'),
        coordinates=mapped_coordinates.T.reshape((3, *target.shape[:3])),
        pe_info=pe_info,
        fieldmap_gradient=gradient,
        hmc_xfms=hmc_xfms,
        fmap_hz=fieldmap.get_fdata(dtype='f4'),
        output_dtype=output_dtype,
        nthreads=nthreads,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )
    resampled_img = target.__class__(resampled_data, target.affine, target.header)
    resampled_img.set_data_dtype('f4')
    # Preserve zooms of additional dimensions
    resampled_img.header.set_zooms(target.header.get_zooms()[:3] + source.header.get_zooms()[3:])

    return resampled_img
