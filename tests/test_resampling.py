import nibabel as nb
import nitransforms as nt
import numpy as np
from numpy.testing import assert_allclose

from nipost.resampling import resample_image, resample_series, resample_vol


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
        jacobian=None,
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
        jacobian=None,
        hmc_xfms=None,
        fmap_hz=fmap_hz,
        order=1,
        nthreads=1,
    )
    assert out.shape == series.shape
    assert np.allclose(out, series, atol=1e-4)


def test_resample_image_identity_to_same_grid(ramp_volume):
    vol = np.asanyarray(ramp_volume.dataobj)
    series = np.stack([vol, vol], axis=-1)
    source = nb.Nifti1Image(series, np.eye(4))
    target = nb.Nifti1Image(np.zeros((5, 5, 5), dtype='f4'), np.eye(4))

    out = resample_image(
        source=source,
        target=target,
        transforms=nt.TransformChain([nt.Affine()]),
        fieldmap=None,
        pe_info=None,
        order=1,
    )
    assert out.shape == series.shape
    assert np.allclose(np.asanyarray(out.dataobj), series, atol=1e-4)


def test_resample_image_applies_jacobian(ramp_volume):
    vol = np.asanyarray(ramp_volume.dataobj)
    source = nb.Nifti1Image(np.stack([vol, vol], axis=-1), np.eye(4))
    target = nb.Nifti1Image(np.zeros((5, 5, 5), dtype='f4'), np.eye(4))
    jac = nb.Nifti1Image(np.linspace(0.5, 1.5, 125, dtype='f4').reshape(5, 5, 5), np.eye(4))
    identity = nt.TransformChain([nt.Affine()])

    plain = resample_image(source, target, identity, None, None, order=1)
    scaled = resample_image(source, target, identity, None, None, jacobian=jac, order=1)

    assert_allclose(
        np.asanyarray(scaled.dataobj),
        np.asanyarray(plain.dataobj) * np.asanyarray(jac.dataobj)[..., None],
        rtol=1e-5,
    )


def test_resample_image_zeroes_fold_over(ramp_volume):
    # J <= 0 marks fold-over: signal there is lost, so it must not turn negative
    vol = np.asanyarray(ramp_volume.dataobj)
    source = nb.Nifti1Image(np.stack([vol, vol], axis=-1), np.eye(4))
    target = nb.Nifti1Image(np.zeros((5, 5, 5), dtype='f4'), np.eye(4))
    jac_data = np.linspace(-0.5, 1.5, 125, dtype='f4').reshape(5, 5, 5)
    jac = nb.Nifti1Image(jac_data, np.eye(4))
    identity = nt.TransformChain([nt.Affine()])

    plain = resample_image(source, target, identity, None, None, order=1)
    scaled = resample_image(source, target, identity, None, None, jacobian=jac, order=1)

    assert_allclose(
        np.asanyarray(scaled.dataobj),
        np.asanyarray(plain.dataobj) * np.clip(jac_data, 0, None)[..., None],
        rtol=1e-5,
    )
