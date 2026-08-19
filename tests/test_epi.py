# tests/test_epi.py
import nibabel as nb
import numpy as np
import pytest

from nipost.epi import ensure_positive_cosines, get_trt, prepare_epi


def _img(affine):
    return nb.Nifti1Image(np.zeros((4, 4, 4, 4), dtype='f4'), affine)


def test_ensure_positive_cosines_flips_negative_axes():
    # LAS-ish affine: negative i cosine
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    img = _img(affine)
    reoriented, axcodes = ensure_positive_cosines(img)
    ornt = nb.io_orientation(reoriented.affine)
    assert np.all(ornt[:, 1] == 1)  # all positive cosines
    assert axcodes == nb.orientations.aff2axcodes(affine)


def test_get_trt_direct_total_readout_time():
    assert get_trt({'TotalReadoutTime': 0.05}) == 0.05


def test_get_trt_effective_echo_spacing(tmp_path):
    img = _img(np.eye(4))  # shape (4,4,4); PE axis j -> npe = 4
    p = tmp_path / 'bold.nii.gz'
    img.to_filename(p)
    meta = {'EffectiveEchoSpacing': 0.001, 'PhaseEncodingDirection': 'j'}
    # trt = ees * (npe - 1) = 0.001 * 3
    assert get_trt(meta, str(p)) == pytest.approx(0.003)


def test_get_trt_fallback():
    assert get_trt({}, fallback=0.1) == 0.1


def test_get_trt_unknown_raises():
    with pytest.raises(ValueError):
        get_trt({})


def test_prepare_epi():
    affine = np.array([[0.0, 2, 0, 0], [-2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])
    img = _img(affine)
    metadata = {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.003}

    assert nb.aff2axcodes(img.affine) == ('P', 'R', 'S')

    reoriented_img, pe_info = prepare_epi(img, metadata)

    assert nb.aff2axcodes(reoriented_img.affine) == ('A', 'R', 'S')
    # PE direction is L->R, so positive TRT
    assert pe_info == [(1, 0.003)] * 4

    metadata = {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.003}
    _, pe_info = prepare_epi(img, metadata)
    # PE direction is R->L, so negative TRT
    assert pe_info == [(1, -0.003)] * 4

    metadata = {'PhaseEncodingDirection': 'i', 'TotalReadoutTime': 0.003}
    _, pe_info = prepare_epi(img, metadata)
    # PE direction is A->P, so negative TRT
    assert pe_info == [(0, -0.003)] * 4

    metadata = {'PhaseEncodingDirection': 'i-', 'TotalReadoutTime': 0.003}
    _, pe_info = prepare_epi(img, metadata)
    # PE direction is P->A, so positive TRT
    assert pe_info == [(0, 0.003)] * 4
