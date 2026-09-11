import pytest

pytest.importorskip('bids')  # skip whole module if [bids] extra not installed


def test_get_layout_is_cached(tmp_path, deriv_dataset):
    from nipost.bids._layout import get_layout

    deriv_dataset(tmp_path)
    a = get_layout(tmp_path)
    b = get_layout(tmp_path)
    assert a is b  # memoized
