# tests/bids/test_collect_derivatives.py
import os
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip('bids')
pytest.importorskip('niworkflows')

from bids.layout import Query as BIDSQuery

import nipost.bids._layout
from nipost.bids.collect import _resolve, collect_derivatives, collect_fieldmaps
from nipost.bids.spec import Group, Ordered, Query, load_spec, sanitize_space


@pytest.fixture(scope='module', autouse=True)
def disable_layout_cache():
    # Allow for modifying directories mid-test
    with mock.patch('nipost.bids.collect.get_layout', nipost.bids._layout.get_layout.__wrapped__):
        yield


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')


@pytest.fixture
def deriv_root(tmp_path, deriv_dataset) -> Path:
    # Plausible-ish derivative dataset
    # JSON files are excluded to avoid specifying extensions in all tests
    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    func = root / 'sub-01' / 'func'
    fmap = root / 'sub-01' / 'fmap'
    # dual T1w + T2w preproc (space-qualified case)
    _write(anat / 'sub-01_desc-preproc_T1w.nii.gz')
    _write(anat / 'sub-01_desc-preproc_T2w.nii.gz')
    # ordered TPMs - include 'Other' as a distrator; no query should find it
    for label in ('GM', 'WM', 'CSF', 'Other'):
        _write(anat / f'sub-01_label-{label}_probseg.nii.gz')
    # surface pair, ordered by hemi
    _write(anat / 'sub-01_hemi-L_white.surf.gii')
    _write(anat / 'sub-01_hemi-R_white.surf.gii')
    # single masks
    _write(anat / 'sub-01_desc-ribbon_mask.nii.gz')
    _write(anat / 'sub-01_desc-brain_mask.nii.gz')
    # coreg + normalization transforms
    _write(anat / 'sub-01_from-T1w_to-T2w_mode-image_xfm.txt')
    _write(anat / 'sub-01_from-anat_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
    # boldrefs
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_space-orig_desc-hmc_boldref.nii.gz')
    # affines
    _write(func / 'sub-01_task-rest_from-run_to-anat_mode-image_desc-coreg_xfm.txt')
    _write(func / 'sub-01_task-rest_from-orig_to-run_mode-image_desc-hmc_xfm.txt')
    _write(func / 'sub-01_task-rest_from-run_to-auto00000_mode-image_xfm.txt')
    _write(func / 'sub-01_task-rest_from-run_to-auto00001_mode-image_xfm.txt')
    # single-level fieldmap
    _write(fmap / 'sub-01_fmapid-auto00000_desc-coeff_fieldmap.nii.gz')
    _write(fmap / 'sub-01_fmapid-auto00000_desc-magnitude_fieldmap.nii.gz')
    _write(fmap / 'sub-01_fmapid-auto00000_desc-preproc_fieldmap.nii.gz')
    # multi-level fieldmap (coeff indices selected so natural sort is not lexical)
    _write(fmap / 'sub-01_fmapid-auto00001_desc-coeff2_fieldmap.nii.gz')
    _write(fmap / 'sub-01_fmapid-auto00001_desc-coeff10_fieldmap.nii.gz')
    _write(fmap / 'sub-01_fmapid-auto00001_desc-magnitude_fieldmap.nii.gz')
    _write(fmap / 'sub-01_fmapid-auto00001_desc-preproc_fieldmap.nii.gz')
    return root


@pytest.fixture
def empty_root(tmp_path, deriv_dataset):
    """A valid but empty derivative dataset, for tests that add their own files."""
    return deriv_dataset(tmp_path / 'empty')


def test_collect_covers_case_catalog(deriv_root):
    spec = {
        'images': Group(
            queries={
                't1w_preproc': Query([{'suffix': 'T1w', 'desc': 'preproc'}]),
                't2w_preproc': Query([{'suffix': 'T2w', 'desc': 'preproc'}]),
                'tpms': Query(
                    [{'suffix': 'probseg', 'label': ['GM', 'WM', 'CSF']}],
                    multi=Ordered(order='label'),
                ),
                'white': Query(
                    [{'suffix': 'white', 'extension': '.surf.gii', 'hemi': ['L', 'R']}],
                    multi=Ordered(order='hemi'),
                ),
                'ribbon': Query([{'desc': 'ribbon', 'suffix': 'mask'}]),
                't1w2t2w': Query([{'from': 'T1w', 'to': 'T2w', 'suffix': 'xfm'}]),
            }
        ),
        'transforms': Group(
            per='space',
            queries={
                'forward': Query([{'from': 'anat', 'to': '{space}', 'suffix': 'xfm'}]),
            },
        ),
        'empty': Group(queries={'none': Query([{'suffix': 'madeup'}])}),
        'fieldmaps': Group(
            per='fmapid',
            queries={
                'coeffs': Query(
                    [
                        {
                            'datatype': 'fmap',
                            'fmapid': '{fmapid}',
                            'desc': ['coeff', 'coeff10', 'coeff2'],
                            'suffix': 'fieldmap',
                        }
                    ],
                    multi=True,
                )
            },
        ),
    }

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        # Include BOLD suffix to ensure that spec entities override caller entities
        entities={'subject': '01', 'suffix': 'bold'},
        params={'space': ['MNI152NLin2009cAsym'], 'fmapid': ['auto00000', 'auto00001']},
    )

    # Top-level keys are always populated, empty or not
    images = out['images']
    transforms = out['transforms']
    empty = out['empty']
    fieldmaps = out['fieldmaps']

    # Images are not parameterized, so are flat
    assert images['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert images['t2w_preproc'].endswith('desc-preproc_T2w.nii.gz')

    # Groups ordered by an entity are sorted by those values as listed
    assert [p.split('label-')[1][:2] for p in images['tpms']] == ['GM', 'WM', 'CS']
    assert [p.split('hemi-')[1][0] for p in images['white']] == ['L', 'R']
    assert isinstance(images['ribbon'], str)
    assert images['t1w2t2w'].endswith('from-T1w_to-T2w_mode-image_xfm.txt')

    # Transforms are parameterized, so nest under the parameter value
    assert transforms['MNI152NLin2009cAsym']['forward'].endswith('_xfm.h5')

    # Fieldmaps are parameterized
    auto0 = fieldmaps['auto00000']
    auto1 = fieldmaps['auto00001']
    assert len(auto0['coeffs']) == 1
    assert len(auto1['coeffs']) == 2
    # Sorting is natural (10 > 2), not lexical ("2" > "10")
    assert 'coeff2' in auto1['coeffs'][0]
    assert 'coeff10' in auto1['coeffs'][1]

    # Empty groups are empty
    assert empty == {}

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01'},
        params={'space': None, 'fmapid': None},
    )
    assert out['transforms'] == {}  # no iterations
    assert out['fieldmaps'] == {}  # no iterations


def test_uniterated_group_with_placeholders(deriv_root):
    spec = {
        'transforms': Group(
            queries={
                'run2anat': Query([{'from': 'run', 'to': 'anat', 'suffix': 'xfm'}]),
                'run2fmap': Query(
                    [{'from': 'run', 'to': '{fmapid}', 'desc': None, 'suffix': 'xfm'}],
                    multi=True,
                ),
            }
        ),
    }

    def collect(fmapid):
        out = collect_derivatives(
            deriv_root,
            spec=spec,
            entities={'subject': '01', 'task': 'rest'},
            params={'fmapid': fmapid},
        )
        assert 'desc-coreg' in out['transforms']['run2anat']
        return out['transforms'].get('run2fmap')

    # The wildcard is explicit
    both = collect(BIDSQuery.ANY)
    assert [p.rsplit('to-', 1)[1][:9] for p in both] == ['auto00000', 'auto00001']
    [one] = collect('auto00000')
    assert one.endswith('to-auto00000_mode-image_xfm.txt')
    # Every transform has `to`, so requiring its absence finds nothing
    assert collect(None) is None


def test_list_valued_entities(deriv_root):
    spec = {
        'images': Group(
            queries={'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}], multi=True)}
        )
    }
    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})
    assert len(out['images']['preproc']) == 2  # matches both T1w and T2w

    # If multi is false, result is ambiguous
    spec = {
        'images': Group(
            queries={'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}])}
        )
    }
    with pytest.raises(ValueError, match='preproc'):
        collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})


def test_ordered_query(empty_root):
    anat = empty_root / 'sub-01' / 'anat'

    spec = {
        'images': Group(
            queries={
                'tpms': Query(
                    [{'suffix': 'probseg', 'label': ['GM', 'WM', 'CSF']}],
                    multi=Ordered(order='label'),
                )
            }
        )
    }
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert out == {'images': {}}

    # GM and WM only -- CSF is missing, so the tuple is incomplete.
    for label in ('GM', 'WM'):
        _write(anat / f'sub-01_label-{label}_probseg.nii.gz')

    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert out == {'images': {}}

    _write(anat / 'sub-01_label-CSF_probseg.nii.gz')
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert 'tpms' in out['images']
    assert len(out['images']['tpms']) == 3


def test_ordered_query_with_none(empty_root):
    """``None`` is a legal member of an ordering value list: it means "the file
    lacking this entity sorts here". One probseg carries `label-GM`, the other
    carries no `label` entity at all, and both sort into the declared order.
    """
    anat = empty_root / 'sub-01' / 'anat'
    gm = anat / 'sub-01_label-GM_probseg.nii.gz'
    probseg = anat / 'sub-01_probseg.nii.gz'

    anat.mkdir(parents=True)
    _write(gm)
    _write(probseg)

    spec = {
        'images': Group(
            queries={
                'tpms': Query(
                    [{'suffix': 'probseg', 'label': ['GM', None]}],
                    multi=Ordered(order='label'),
                )
            }
        )
    }

    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert out['images']['tpms'] == [str(gm), str(probseg)]


def test_ordered_query_with_ambiguity(empty_root: Path) -> None:
    anat = empty_root / 'sub-01' / 'anat'
    _write(anat / 'sub-01_acq-A_label-GM_probseg.nii.gz')
    _write(anat / 'sub-01_acq-A_label-WM_probseg.nii.gz')
    _write(anat / 'sub-01_acq-B_label-GM_probseg.nii.gz')
    _write(anat / 'sub-01_acq-B_label-WM_probseg.nii.gz')

    # Ordered field with multiple matches for a single value
    spec = {
        'images': Group(
            queries={
                'tpms': Query(
                    entities=[{'label': ['GM', 'WM'], 'suffix': 'probseg'}],
                    multi=Ordered(order='label'),
                )
            },
        )
    }
    with pytest.raises(ValueError, match=r'^images/tpms in .*with label='):
        collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})


def test_ambiguity_error_names_its_location(deriv_root):
    spec = {
        'images': Group(
            queries={
                'preproc': Query(
                    [{'suffix': 'madeup'}, {'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}]
                )
            }
        )
    }
    with pytest.raises(ValueError) as excinfo:
        collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})
    message = str(excinfo.value)
    assert message.startswith('images/preproc in ')
    assert deriv_root.name in message
    assert '(alternative 1): expected at most one match for ' in message
    assert "'suffix': ['T1w', 'T2w']" in message


def test_ambiguity_error_names_the_iteration(empty_root):
    fmap = empty_root / 'sub-01' / 'fmap'
    for run in ('01', '02'):
        _write(fmap / f'sub-01_run-{run}_fmapid-auto00000_desc-preproc_fieldmap.nii.gz')
    with pytest.raises(ValueError, match=r'^fieldmaps\[auto00000\]/fieldmap in '):
        collect_fieldmaps(empty_root, entities={'subject': '01'})


FORWARD = {
    'transforms': Group(
        per='space',
        queries={'forward': Query([{'from': 'anat', 'to': '{space}', 'suffix': 'xfm'}])},
    )
}


def _forward(root, space):
    out = collect_derivatives(
        root, spec=FORWARD, entities={'subject': '01'}, params={'space': space}
    )
    return out['transforms']


def test_indexed_group_takes_a_scalar_none_or_sequence(deriv_root):
    one = _forward(deriv_root, 'MNI152NLin2009cAsym')
    assert list(one) == ['MNI152NLin2009cAsym']
    assert one['MNI152NLin2009cAsym']['forward'].endswith(
        'to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    )
    assert _forward(deriv_root, ['MNI152NLin2009cAsym']) == one
    assert _forward(deriv_root, ('MNI152NLin2009cAsym',)) == one
    assert _forward(deriv_root, None) == {}
    assert _forward(deriv_root, []) == {}
    # A non-string scalar is one value too
    assert _forward(deriv_root, 7) == {7: {}}
    # Requested values are keys even when nothing is found
    assert _forward(deriv_root, ['MNI152NLin2009cAsym', 'fsnative'])['fsnative'] == {}


def test_indexed_group_requires_its_parameter(deriv_root):
    with pytest.raises(TypeError, match=r"'transforms'.*'space'"):
        collect_derivatives(deriv_root, spec=FORWARD, entities={'subject': '01'})


@pytest.mark.parametrize(
    'space',
    [BIDSQuery.ANY, [BIDSQuery.ANY], ['MNI152NLin2009cAsym', BIDSQuery.OPTIONAL]],
)
def test_indexed_group_rejects_pybids_queries(deriv_root, space):
    with pytest.raises(TypeError, match='space'):
        _forward(deriv_root, space)


def test_indexed_group_none_member_requires_absence(deriv_root):
    spec = {
        'boldrefs': Group(
            per='desc',
            queries={'boldref': Query([{'suffix': 'boldref', 'desc': '{desc}'}])},
        )
    }
    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01', 'task': 'rest'},
        params={'desc': [None, 'hmc']},
    )
    assert out['boldrefs'][None]['boldref'].endswith('task-rest_space-run_boldref.nii.gz')
    assert out['boldrefs']['hmc']['boldref'].endswith('desc-hmc_boldref.nii.gz')


def test_first_matching_alternative_wins(empty_root):
    """Alternatives are tried in order; the first with any match is used alone."""
    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_desc-coreg_boldref.nii.gz')

    spec = {
        'images': Group(
            queries={
                'run_boldref': Query(
                    [
                        {'space': 'run', 'suffix': 'boldref'},
                        {'desc': 'coreg', 'suffix': 'boldref'},
                    ]
                )
            }
        )
    }
    # Second alternative hit when first misses
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01', 'task': 'rest'})
    assert out['images']['run_boldref'].endswith('desc-coreg_boldref.nii.gz')

    # If first hit, accept first
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01', 'task': 'rest'})
    assert out['images']['run_boldref'].endswith('space-run_boldref.nii.gz')


def test_scoped_queries(empty_root):
    """Scopes act as pass-lists, and empty is distinct from absent"""
    func_boldref = empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz'
    # Include invalid anat distractor to ensure datatype constraint is respected
    anat_boldref = empty_root / 'sub-01' / 'anat' / 'sub-01_space-subject_boldref.nii.gz'

    _write(func_boldref)
    _write(anat_boldref)

    entity_alts = [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}]
    spec = {
        'images': Group(
            queries={
                'sub_scoped': Query(entity_alts, scope=['subject']),
                'root_scoped': Query(entity_alts, scope=[]),
                'unscoped': Query(entity_alts),
            }
        )
    }
    out = collect_derivatives(
        empty_root,
        spec=spec,
        entities={'subject': '01', 'task': 'rest', 'run': '01'},
    )
    # task and run are dropped, subject is found
    assert out['images']['sub_scoped'] == str(func_boldref)
    # task, run and subject are dropped, boldref still found
    assert out['images']['root_scoped'] == str(func_boldref)
    # task and run not dropped, failed match
    assert 'unscoped' not in out['images']

    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '02', 'run': '01'})
    # task, run and subject are dropped, boldref from sub-01 found
    assert out['images']['root_scoped'] == str(func_boldref)
    # subject not dropped, failed match
    assert 'sub_scoped' not in out['images']
    assert 'unscoped' not in out['images']


def test_none_requires_entity_to_be_absent(deriv_root):
    """`None` is PyBIDS Query.NONE: the entity must be ABSENT, not unconstrained."""
    spec = {
        'images': Group(queries={'coreg_boldref': Query([{'suffix': 'boldref', 'desc': None}])})
    }
    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})

    assert 'desc-' not in out['images']['coreg_boldref']

    spec = {
        'images': Group(
            queries={'boldrefs': Query([{'desc': ['hmc', None], 'suffix': 'boldref'}], multi=True)}
        )
    }
    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01', 'task': 'rest'})

    names = {os.path.basename(p) for p in out['images']['boldrefs']}
    assert names == {
        'sub-01_task-rest_space-run_boldref.nii.gz',
        'sub-01_task-rest_space-orig_desc-hmc_boldref.nii.gz',
    }


def test_resolver_does_not_normalize_param_values():
    """Values pass through verbatim; normalizing is the caller's job.

    Asserted with a cohort space, whose sanitized form differs from its raw
    form -- the existing group-level test uses a space that sanitizes to
    itself, so it cannot detect a resolver that normalizes.
    """
    out = _resolve({'to': '{space}'}, {}, None, {'space': 'MNI152NLin6Asym:cohort-1'})

    assert out['to'] == ['MNI152NLin6Asym:cohort-1']


def test_resolve_substitutes_placeholder_inside_a_list():
    resolved = _resolve(
        {'to': ['{fmapid}', 'auto00000']}, base={}, scope=None, params={'fmapid': 'auto00001'}
    )

    assert resolved['to'] == ['auto00001', 'auto00000']


def test_resolve_keeps_a_none_member_of_a_list():
    resolved = _resolve(
        {'to': ['{fmapid}', 'auto00000']}, base={}, scope=None, params={'fmapid': None}
    )
    assert resolved['to'] == [None, 'auto00000']


def test_resolve_keeps_a_lone_none_member():
    resolved = _resolve({'to': ['{fmapid}']}, base={}, scope=None, params={'fmapid': None})
    assert resolved == {'to': [None]}


def test_resolve_substitutes_none_verbatim():
    resolved = _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': None})
    assert resolved == {'to': [None]}


def test_unbound_placeholder_raises(deriv_root):
    spec = {
        'transforms': Group(
            queries={'run2fmap': Query([{'from': 'run', 'to': '{fmapid}', 'suffix': 'xfm'}])}
        )
    }
    with pytest.raises(TypeError, match=r"transforms/run2fmap \(alternative 0\).*'fmapid'"):
        collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})


def test_resolve_raises_when_a_list_param_binds_a_query_level_placeholder():
    # Multiple values for parameters would be a feature in search of a use case.
    # If this is eventually supported, it will be more complicated than just removing
    # the check, so error for now.

    # Check against `{fmapid}` to ensure the placeholder is not interpolated in the message
    with pytest.raises(TypeError, match=r'\{fmapid\}'):
        _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': ['a', 'b']})


def test_resolve_substitutes_a_scalar_param_into_a_scalar_placeholder():
    resolved = _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': 'auto00001'})

    assert resolved == {'to': ['auto00001']}


def test_space_placeholder_substitutes(deriv_root):
    spec = {
        'transforms': Group(
            per='space',
            queries={
                'forward': Query([{'from': 'anat', 'to': '{space}', 'suffix': 'xfm'}]),
            },
        )
    }
    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01'},
        params={'space': ['MNI152NLin2009cAsym']},
    )

    assert out['transforms']['MNI152NLin2009cAsym']['forward'].endswith(
        'to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    )


def test_space_cohort_two_step(empty_root):
    _write(empty_root / 'sub-01' / 'anat' / 'sub-01_from-T1w_to-MNIInfant+1_mode-image_xfm.h5')

    spec = {
        'transforms': Group(
            per='space',
            queries={
                'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}]),
            },
        )
    }

    # Niworkflows specification-style is not automatically cleaned
    space = 'MNIInfant:cohort-1'
    out = collect_derivatives(
        empty_root, spec=spec, entities={'subject': '01'}, params={'space': [space]}
    )
    assert out['transforms'] == {space: {}}

    # The caller should call sanitize_space, and the transforms dict uses the passed key
    space = sanitize_space(space)
    out = collect_derivatives(
        empty_root, spec=spec, entities={'subject': '01'}, params={'space': [space]}
    )
    assert out['transforms'][space]['forward'].endswith('to-MNIInfant+1_mode-image_xfm.h5')


@pytest.fixture
def anat_with_std_space_dupes(tmp_path, deriv_dataset):
    """fMRIPrep's default output: native anat files plus standard-space copies.

    Regression fixture for the anat spec not constraining ``space``: without
    ``space: null``, t1w_preproc/t2w_preproc/mask/dseg each match 2 files
    (native + std-space) and raise, while ``tpms`` (ordered by label) silently
    returns the wrong-resolution std-space probsegs instead.
    """
    root = deriv_dataset(tmp_path / 'deriv', generated_by='fMRIPrep')
    anat = root / 'sub-01' / 'anat'
    for space_suffix in ('', 'space-MNI152NLin2009cAsym_res-2_'):
        _write(anat / f'sub-01_{space_suffix}desc-preproc_T1w.nii.gz')
        _write(anat / f'sub-01_{space_suffix}desc-preproc_T2w.nii.gz')
        _write(anat / f'sub-01_{space_suffix}desc-brain_mask.nii.gz')
        _write(anat / f'sub-01_{space_suffix}dseg.nii.gz')
        for label in ('GM', 'WM', 'CSF'):
            _write(anat / f'sub-01_{space_suffix}label-{label}_probseg.nii.gz')
    return root


def test_anat_spec_prefers_native_over_std_space_dupes(anat_with_std_space_dupes):
    """The shipped anat spec must resolve to native files, not std-space copies."""
    out = collect_derivatives(
        anat_with_std_space_dupes,
        spec=load_spec('anat'),
        entities={'subject': '01'},
        params={'space': None},
    )

    images = out['images']
    for key in ('t1w_preproc', 't2w_preproc', 'mask', 'dseg'):
        assert 'space-' not in images[key], f'{key}: expected native file, got {images[key]!r}'
    assert all('space-' not in p for p in images['tpms']), (
        f'tpms: expected native files, got {images["tpms"]!r}'
    )
    assert [p.rsplit('label-', 1)[1][:2] for p in images['tpms']] == ['GM', 'WM', 'CS']


def test_image_queries_ignore_json_sidecars(empty_root):
    """PyBIDS indexes a derivative's sidecar as its own file with the same entities."""
    anat = empty_root / 'sub-01' / 'anat'
    for name in (
        'sub-01_desc-preproc_T1w.nii.gz',
        'sub-01_desc-brain_mask.nii.gz',
    ):
        _write(anat / name)
    for name in (
        'sub-01_desc-preproc_T1w.json',
        'sub-01_desc-brain_mask.json',
    ):
        anat.mkdir(parents=True, exist_ok=True)
        (anat / name).write_text('{}')

    out = collect_derivatives(
        empty_root, spec=load_spec('anat'), entities={'subject': '01'}, params={'space': None}
    )

    assert out['images']['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert out['images']['mask'].endswith('desc-brain_mask.nii.gz')


def test_int_typed_entities_accept_strings(empty_root):
    """PyBIDS coerces the filter, so the axis has to coerce identically.

    The value list is read twice -- once as a filter, once as the ordering
    axis -- and PyBIDS parses ``run`` as a ``PaddedInt``. A declared ``'1'``
    matched files as a filter but missed in the axis lookup, because dict
    lookup hashes before it compares and ``hash('1')`` is not ``hash(1)``.
    """
    func = empty_root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    for run in ('00', '01', '02', '03'):
        _write(func / f'sub-01_task-rest_run-{run}_desc-preproc_bold.nii.gz')

    for run in (
        ['1', '2'],
        ['01', '02'],
        [1, 2],
        [2, 1],
        ['02', '01'],
        # Check falsy values
        ['0', '1'],
        [0, 1],
    ):
        spec = {
            'bolds': Group(
                queries={
                    'runs': Query(
                        entities=[
                            {
                                'datatype': 'func',
                                'suffix': 'bold',
                                'desc': 'preproc',
                                'run': run,
                            }
                        ],
                        multi=Ordered(order='run'),
                    )
                }
            )
        }
        out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

        # All invocations find both runs
        assert len(out['bolds']['runs']) == 2

        # Order is respected for ints and strs
        for i, run_idx in enumerate(run):
            assert f'{int(run_idx):02d}' in out['bolds']['runs'][i]

    # Non-coercible values are left as-is, and just miss
    spec = {
        'bolds': Group(
            queries={
                'runs': Query(
                    entities=[
                        {
                            'datatype': 'func',
                            'suffix': 'bold',
                            'desc': 'preproc',
                            'run': ['01', 'GM'],
                        }
                    ],
                    multi=Ordered(order='run'),
                )
            }
        )
    }
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert 'runs' not in out['bolds']


def test_empty_unordered_multi_omits_its_key(empty_root):
    spec = {
        'images': Group(
            queries={'preproc': Query([{'suffix': 'T1w', 'desc': 'preproc'}], multi=True)}
        )
    }
    assert collect_derivatives(empty_root, spec=spec, entities={'subject': '01'}) == {'images': {}}


def test_unbound_placeholder_in_a_later_alternative_raises(empty_root):
    _write(empty_root / 'sub-01' / 'func' / 'sub-01_task-rest_from-run_to-T1w_mode-image_xfm.txt')
    spec = {
        'transforms': Group(
            queries={
                'xfm': Query(
                    [
                        {'from': 'run', 'suffix': 'xfm'},
                        {'from': 'boldref', 'to': '{fmapid}', 'suffix': 'xfm'},
                    ]
                )
            }
        )
    }
    with pytest.raises(TypeError, match=r'xfm \(alternative 1\)'):
        collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})


def test_unbound_placeholder_raises_without_iterations(empty_root):
    spec = {
        'fieldmaps': Group(
            per='fmapid',
            queries={
                'coeffs': Query([{'fmapid': '{fmapid}', 'desc': '{level}', 'suffix': 'fieldmap'}])
            },
        )
    }
    with pytest.raises(TypeError, match='level'):
        collect_derivatives(
            empty_root, spec=spec, entities={'subject': '01'}, params={'fmapid': None}
        )


@pytest.mark.parametrize('listed', [False, True], ids=['scalar', 'list'])
def test_bound_values_follow_pybids_semantics(empty_root, listed):
    func = empty_root / 'sub-01' / 'func'
    plain = 'sub-01_task-rest_boldref.nii.gz'
    hmc = 'sub-01_task-rest_desc-hmc_boldref.nii.gz'
    coreg = 'sub-01_task-rest_desc-coreg_boldref.nii.gz'
    for name in (plain, hmc, coreg):
        _write(func / name)
    desc = ['{desc}', 'coreg'] if listed else '{desc}'
    spec = {
        'images': Group(
            queries={'boldrefs': Query([{'suffix': 'boldref', 'desc': desc}], multi=True)}
        )
    }

    def found(value):
        out = collect_derivatives(
            empty_root,
            spec=spec,
            entities={'subject': '01', 'task': 'rest'},
            params={'desc': value},
        )
        names = {os.path.basename(p) for p in out['images'].get('boldrefs', [])}
        if listed:
            # The literal member always matches
            assert coreg in names
            names -= {coreg}
        return names

    assert found('hmc') == {hmc}
    assert found(None) == {plain}
    assert found(BIDSQuery.NONE) == {plain}
    assert found(BIDSQuery.ANY) == ({hmc} if listed else {hmc, coreg})
    assert found(BIDSQuery.OPTIONAL) == ({plain, hmc} if listed else {plain, hmc, coreg})
