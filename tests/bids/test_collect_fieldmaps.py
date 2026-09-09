# tests/bids/test_collect_fieldmaps.py
import pytest

pytest.importorskip('bids')


@pytest.fixture
def fmap_deriv(tmp_path, deriv_dataset):
    root = deriv_dataset(tmp_path / 'deriv')
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

    out = collect_fieldmaps(fmap_deriv, entities={'subject': '01'})
    assert 'auto00000' in out['fieldmaps']
    # coeffs is always a list: one entry per B-spline level
    assert out['fieldmaps']['auto00000']['coeffs'] == [
        str(fmap_deriv / 'sub-01' / 'fmap' / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz')
    ]


def test_collect_fieldmaps_uses_the_passed_spec_not_the_bundled_one(fmap_deriv):
    """The `spec=` override must actually be read, not merely accepted.

    The group name deliberately is not `fieldmaps` -- the bundled name -- so
    passing this test proves the wrapper nests results under the spec's own
    group name rather than assuming the bundled one.
    """
    from nipost.bids.collect import collect_fieldmaps
    from nipost.bids.spec import Group, Query

    spec = {
        'custom_fmaps': Group(
            over='fmapid',
            queries={
                'preproc': Query([{'datatype': 'fmap', 'fmapid': '{fmapid}', 'desc': 'preproc'}])
            },
        )
    }

    out = collect_fieldmaps(fmap_deriv, entities={'subject': '01'}, spec=spec)

    assert 'fieldmaps' not in out
    preproc = out['custom_fmaps']['auto00000']['preproc']
    assert isinstance(preproc, str)
    assert preproc.endswith('fmapid-auto00000_desc-preproc_fieldmap.nii.gz')


@pytest.fixture
def multilevel_fmap_deriv(tmp_path, deriv_dataset):
    """A fieldmap written with two B-spline levels, as SDCFlows does."""
    root = deriv_dataset(tmp_path / 'deriv')
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

    out = collect_fieldmaps(multilevel_fmap_deriv, entities={'subject': '01'})

    coeffs = out['fieldmaps']['auto00000']['coeffs']
    assert [p.rsplit('desc-', 1)[1] for p in coeffs] == [
        'coeff0_fieldmap.nii.gz',
        'coeff1_fieldmap.nii.gz',
    ]
    # the other two items stay scalar
    assert isinstance(out['fieldmaps']['auto00000']['fieldmap'], str)
    assert isinstance(out['fieldmaps']['auto00000']['magnitude'], str)


def test_collect_fieldmaps_ignores_json_sidecars(tmp_path, deriv_dataset):
    """A preproc fieldmap ships a sidecar; only the image may be collected."""
    from nipost.bids.collect import collect_fieldmaps

    root = deriv_dataset(tmp_path / 'deriv')
    fmap = root / 'sub-01' / 'fmap'
    fmap.mkdir(parents=True)
    for name in (
        'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz',
        'sub-01_fmapid-auto00000_desc-epi_fieldmap.nii.gz',
    ):
        (fmap / name).write_text('')
    (fmap / 'sub-01_fmapid-auto00000_desc-preproc_fieldmap.json').write_text('{}')

    out = collect_fieldmaps(root, entities={'subject': '01'})

    fieldmap = out['fieldmaps']['auto00000']['fieldmap']
    assert isinstance(fieldmap, str)
    assert fieldmap.endswith('desc-preproc_fieldmap.nii.gz')
    assert out['fieldmaps']['auto00000']['coeffs'] == [
        str(fmap / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz')
    ]


def test_collect_fieldmaps_orders_bspline_levels(multilevel_fmap_deriv):
    """The end-to-end shape, on a real layout."""
    from nipost.bids.collect import collect_fieldmaps

    coeffs = collect_fieldmaps(multilevel_fmap_deriv, entities={'subject': '01'})['fieldmaps'][
        'auto00000'
    ]['coeffs']

    assert [p.rsplit('desc-', 1)[1] for p in coeffs] == [
        'coeff0_fieldmap.nii.gz',
        'coeff1_fieldmap.nii.gz',
    ]


@pytest.fixture
def two_fieldmap_root(tmp_path, deriv_dataset):
    """A subject with two fieldmaps -- the case that broke the flat fmap spec."""
    root = deriv_dataset(tmp_path / 'fmap_deriv')
    fmap = root / 'sub-01' / 'fmap'
    fmap.mkdir(parents=True)
    for fmapid in ('auto00000', 'auto00001'):
        for desc in ('preproc', 'epi', 'coeff'):
            (fmap / f'sub-01_fmapid-{fmapid}_desc-{desc}_fieldmap.nii.gz').write_bytes(b'')
    return root


def test_collect_derivatives_runs_the_fmap_spec_directly(two_fieldmap_root):
    """The general collector must handle the fmap spec.

    Before groups, this raised "expected at most one match": the fmap
    queries landed flat with no fmapid indexing, so two fieldmaps collided
    on a scalar key.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import load_spec

    out = collect_derivatives(
        two_fieldmap_root,
        spec=load_spec('fmap'),
        entities={'subject': '01'},
        params={'fmapid': ['auto00000', 'auto00001']},
    )

    assert set(out['fieldmaps']) == {'auto00000', 'auto00001'}
    assert out['fieldmaps']['auto00000']['fieldmap'].endswith(
        'fmapid-auto00000_desc-preproc_fieldmap.nii.gz'
    )


def test_collect_fieldmaps_takes_entities_by_keyword_only(fmap_deriv):
    """Matches collect_derivatives, which is keyword-only past the dataset
    root. Several downstream projects are about to hard-code both, so the two
    signatures agree before that happens rather than after."""
    from nipost.bids.collect import collect_fieldmaps

    with pytest.raises(TypeError, match='positional'):
        collect_fieldmaps(fmap_deriv, {'subject': '01'})  # type: ignore

    out = collect_fieldmaps(fmap_deriv, entities={'subject': '01'})

    assert 'auto00000' in out['fieldmaps']
