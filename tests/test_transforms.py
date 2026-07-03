import nitransforms as nt
import numpy as np

from nipost.transforms import load_transforms


def test_empty_list_returns_identity():
    xfm = load_transforms([], [])
    assert np.allclose(xfm.matrix, np.eye(4))


def test_single_affine_roundtrip(tmp_path):
    aff = nt.Affine(np.diag([2.0, 2.0, 2.0, 1.0]))
    p = tmp_path / 'xfm.txt'
    aff.to_filename(p, fmt='itk')

    loaded = load_transforms([p], [False])
    assert np.allclose(loaded.matrix, aff.matrix)


def test_inverse_flag_inverts(tmp_path):
    aff = nt.Affine(np.diag([2.0, 2.0, 2.0, 1.0]))
    p = tmp_path / 'xfm.txt'
    aff.to_filename(p, fmt='itk')

    loaded = load_transforms([p], [True])
    assert np.allclose(loaded.matrix, np.linalg.inv(aff.matrix))


def test_mismatched_inverse_length_raises(tmp_path):
    import pytest

    p = tmp_path / 'xfm.txt'
    nt.Affine().to_filename(p, fmt='itk')
    with pytest.raises(ValueError, match='Mismatched'):
        load_transforms([p, p], [True, False, True])
