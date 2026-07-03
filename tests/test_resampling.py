import numpy as np

from nipost.resampling import resample_series, resample_vol


def test_resample_vol_identity_recovers_data(ramp_volume):
    data = np.asanyarray(ramp_volume.dataobj)
    # target coordinates = source voxel grid (identity mapping)
    # coordinates must have shape (3, *target_shape) so shape[1:] matches fmap_hz.shape
    coordinates = np.indices(data.shape).astype('f4')  # shape (3, 5, 5, 5)
    fmap_hz = np.zeros(data.shape, dtype='f4')

    out = resample_vol(
        data=data,
        coordinates=coordinates,
        pe_info=(0, 0.0),
        jacobian=False,
        hmc_xfm=None,
        fmap_hz=fmap_hz,
        order=1,
    )
    assert np.allclose(out, data, atol=1e-4)


def test_resample_series_identity_recovers_series(ramp_volume):
    vol = np.asanyarray(ramp_volume.dataobj)
    series = np.stack([vol, vol + 1], axis=-1)  # (5,5,5,2)
    # coordinates shape (3, 5, 5, 5) so shape[1:] matches fmap_hz.shape
    coordinates = np.indices(vol.shape).astype('f4')  # shape (3, 5, 5, 5)
    fmap_hz = np.zeros(vol.shape, dtype='f4')

    out = resample_series(
        data=series,
        coordinates=coordinates,
        pe_info=[(0, 0.0), (0, 0.0)],
        jacobian=False,
        hmc_xfms=None,
        fmap_hz=fmap_hz,
        order=1,
        nthreads=1,
    )
    assert out.shape == series.shape
    assert np.allclose(out, series, atol=1e-4)
