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
    # coeffs is a scalar path (Task 0), not a list
    assert isinstance(out['auto00000']['coeffs'], str)
    assert out['auto00000']['coeffs'].endswith('desc-coeff_fieldmap.nii.gz')
