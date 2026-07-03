"""In-process resampling of BOLD series (ported from fMRIPrep)."""

import asyncio
import os
from functools import partial

import nibabel as nb
import numpy as np
from scipy import ndimage as ndi

from nipost._async import worker


def resample_vol(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: tuple[int, float],
    jacobian: bool,
    hmc_xfm: np.ndarray | None,
    fmap_hz: np.ndarray,
    output: np.dtype | np.ndarray | None = None,
    order: int = 3,
    mode: str = 'constant',
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
    hmc_xfm
        Affine transformation accounting for head motion from the individual
        volume into the BOLD reference space. This affine must be in VOX2VOX
        form.
    fmap_hz
        The fieldmap, sampled to the target space, in Hz
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

    if jacobian:
        result *= 1 + np.gradient(vsm, axis=pe_info[0])

    return result


async def resample_series_async(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: list[tuple[int, float]],
    jacobian: bool,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
    output_dtype: np.dtype | None = None,
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
    max_concurrent: int = min(os.cpu_count() or 1, 12),
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
    if data.ndim == 3:
        return resample_vol(
            data,
            coordinates,
            pe_info[0],
            jacobian,
            hmc_xfms[0] if hmc_xfms else None,
            fmap_hz,
            output_dtype,
            order,
            mode,
            cval,
            prefilter,
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    # Order F ensures individual volumes are contiguous in memory
    # Also matches NIfTI, making final save more efficient
    out_array = np.zeros(coordinates.shape[1:] + data.shape[-1:], dtype=output_dtype, order='F')

    tasks = [
        asyncio.create_task(
            worker(
                partial(
                    resample_vol,
                    data=volume,
                    coordinates=coordinates,
                    pe_info=pe_info[volid],
                    jacobian=jacobian,
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
    jacobian: bool,
    hmc_xfms: list[np.ndarray] | None,
    fmap_hz: np.ndarray,
    output_dtype: np.dtype | None = None,
    order: int = 3,
    mode: str = 'constant',
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
            jacobian=jacobian,
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
