# tests/bids/test_collect_fieldmaps.py
import json

import pytest

pytest.importorskip('bids')


@pytest.fixture
def fmap_deriv(tmp_path):
    root = tmp_path / 'deriv'
    (root).mkdir()
    (root / 'dataset_description.json').write_text(
        json.dumps(
            {
                'Name': 'x',
                'BIDSVersion': '1.8.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'nipost'}],
            }
        )
    )
    fmap = root / 'sub-01' / 'fmap'
    fmap.mkdir(parents=True)
    for name in (
        'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-epi_fieldmap.nii.gz',
    ):
        (fmap / name).write_text('')
    return root


def test_collect_fieldmaps_groups_by_id(fmap_deriv):
    from nipost.bids.collect import collect_fieldmaps

    out = collect_fieldmaps(fmap_deriv, {'subject': '01'})
    assert 'auto00000' in out
    # coeffs is always a list: one entry per B-spline level
    assert out['auto00000']['coeffs'] == [
        str(fmap_deriv / 'sub-01' / 'fmap' / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz')
    ]


@pytest.fixture
def multilevel_fmap_deriv(tmp_path):
    """A fieldmap written with two B-spline levels, as SDCFlows does."""
    root = tmp_path / 'deriv'
    root.mkdir()
    (root / 'dataset_description.json').write_text(
        json.dumps(
            {
                'Name': 'x',
                'BIDSVersion': '1.8.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'nipost'}],
            }
        )
    )
    fmap = root / 'sub-01' / 'fmap'
    fmap.mkdir(parents=True)
    for name in (
        'sub-01_fmapid-auto00000_desc-coeff0_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-coeff1_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-epi_fieldmap.nii.gz',
    ):
        (fmap / name).write_text('')
    return root


def test_collect_fieldmaps_returns_every_bspline_level(multilevel_fmap_deriv):
    """Multiple B-spline levels are a valid dataset, not an ambiguous one."""
    from nipost.bids.collect import collect_fieldmaps

    out = collect_fieldmaps(multilevel_fmap_deriv, {'subject': '01'})

    coeffs = out['auto00000']['coeffs']
    assert [p.rsplit('desc-', 1)[1] for p in coeffs] == [
        'coeff0_fieldmap.nii.gz',
        'coeff1_fieldmap.nii.gz',
    ]
    # the other two items stay scalar
    assert isinstance(out['auto00000']['fieldmap'], str)
    assert isinstance(out['auto00000']['magnitude'], str)


def test_collect_fieldmaps_ignores_json_sidecars(tmp_path):
    """A preproc fieldmap ships a sidecar; only the image may be collected."""
    import json as _json

    from nipost.bids.collect import collect_fieldmaps

    root = tmp_path / 'deriv'
    root.mkdir()
    (root / 'dataset_description.json').write_text(
        _json.dumps(
            {
                'Name': 'x',
                'BIDSVersion': '1.8.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'nipost'}],
            }
        )
    )
    fmap = root / 'sub-01' / 'fmap'
    fmap.mkdir(parents=True)
    for name in (
        'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-epi_fieldmap.nii.gz',
    ):
        (fmap / name).write_text('')
    (fmap / 'sub-01_fmapid-auto00000_desc-preproc_fieldmap.json').write_text('{}')

    out = collect_fieldmaps(root, {'subject': '01'})

    assert out['auto00000']['fieldmap'].endswith('desc-preproc_fieldmap.nii.gz')
    assert out['auto00000']['coeffs'] == [
        str(fmap / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz')
    ]
