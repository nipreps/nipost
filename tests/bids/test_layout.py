import pytest

pytest.importorskip('bids')  # skip whole module if [bids] extra not installed


def test_get_layout_is_cached(tmp_path):
    from nipost.bids._layout import get_layout

    (tmp_path / 'dataset_description.json').write_text(
        '{"Name": "x", "BIDSVersion": "1.8.0", "DatasetType": "derivative", '
        '"GeneratedBy": [{"Name": "nipost"}]}'
    )
    a = get_layout(tmp_path)
    b = get_layout(tmp_path)
    assert a is b  # memoized
