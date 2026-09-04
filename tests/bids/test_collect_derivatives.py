# tests/bids/test_collect_derivatives.py
import json

import pytest

pytest.importorskip('bids')


def _write(path, **entities):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')
    return path


@pytest.fixture
def deriv_root(tmp_path):
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
    anat = root / 'sub-01' / 'anat'
    # dual T1w + T2w preproc (space-qualified case)
    _write(anat / 'sub-01_desc-preproc_T1w.nii.gz')
    _write(anat / 'sub-01_desc-preproc_T2w.nii.gz')
    # ordered TPMs
    for label in ('GM', 'WM', 'CSF'):
        _write(anat / f'sub-01_label-{label}_probseg.nii.gz')
    # surface pair
    _write(anat / 'sub-01_hemi-L_white.surf.gii')
    _write(anat / 'sub-01_hemi-R_white.surf.gii')
    # single mask
    _write(anat / 'sub-01_desc-ribbon_mask.nii.gz')
    # coreg + normalization transforms
    _write(anat / 'sub-01_from-T1w_to-T2w_mode-image_xfm.txt')
    _write(anat / 'sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
    return root


def test_collect_covers_case_catalog(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(
        items={
            't1w_preproc': Query([{'suffix': 'T1w', 'desc': 'preproc'}], 'single'),
            't2w_preproc': Query([{'suffix': 'T2w', 'desc': 'preproc'}], 'single'),
            'tpms': Query([{'suffix': 'probseg'}], 'ordered', labels=['GM', 'WM', 'CSF']),
            'white': Query([{'suffix': 'white', 'extension': '.surf.gii'}], 'pair'),
            'ribbon': Query([{'desc': 'ribbon', 'suffix': 'mask'}], 'single'),
            't1w2t2w': Query([{'from': 'T1w', 'to': 'T2w', 'suffix': 'xfm'}], 'single'),
        },
        space_transforms={
            'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}], 'single'),
        },
    )

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        subject_id='01',
        std_spaces=['MNI152NLin2009cAsym'],
    )

    assert out['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert out['t2w_preproc'].endswith('desc-preproc_T2w.nii.gz')
    assert [p.split('label-')[1][:2] for p in out['tpms']] == ['GM', 'WM', 'CS']
    assert len(out['white']) == 2  # sorted L, R
    assert isinstance(out['ribbon'], str)
    assert out['t1w2t2w'].endswith('from-T1w_to-T2w_mode-image_xfm.txt')
    # space-NESTED transform dict (anat case)
    assert out['transforms']['MNI152NLin2009cAsym']['forward'].endswith(
        'to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    )


@pytest.fixture
def func_root(tmp_path):
    root = tmp_path / 'fderiv'
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
    func = root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_desc-hmc_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    _write(func / 'sub-01_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    _write(func / 'sub-01_task-rest_from-boldref_to-auto00000_mode-image_xfm.txt')
    return root


def test_func_flat_transforms_and_boldref2fmap_list(func_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(
        items={'hmc_boldref': Query([{'desc': 'hmc', 'suffix': 'boldref'}], 'single')},
        transforms={
            'hmc': Query([{'from': 'orig', 'to': 'boldref', 'suffix': 'xfm'}], 'single'),
            'boldref2anat': Query([{'from': 'boldref', 'to': 'T1w', 'suffix': 'xfm'}], 'single'),
            'boldref2fmap': Query(
                [{'from': 'boldref', 'to': '{fieldmap_id}', 'desc': None, 'suffix': 'xfm'}],
                'list',
            ),
        },
    )
    # Called WITHOUT fieldmap_id, exactly as the notebook does. boldref2fmap must
    # return only the fmap transform (desc absent), NOT the desc-coreg file.
    out = collect_derivatives(func_root, spec=spec, subject_id='01', entities={'task': 'rest'})

    # FLAT transform dict (no std_spaces): keys sit directly under 'transforms'
    assert isinstance(out['transforms']['hmc'], str)
    assert out['transforms']['hmc'].endswith('desc-hmc_xfm.txt')
    assert isinstance(out['transforms']['boldref2anat'], str)
    assert 'desc-coreg' in out['transforms']['boldref2anat']
    # boldref2fmap is a LIST (notebook indexes [0]); the desc-coreg file is excluded
    assert isinstance(out['transforms']['boldref2fmap'], list)
    assert len(out['transforms']['boldref2fmap']) == 1
    assert out['transforms']['boldref2fmap'][0].endswith('to-auto00000_mode-image_xfm.txt')
    assert 'desc-coreg' not in out['transforms']['boldref2fmap'][0]


def test_list_valued_entities(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(items={'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}], 'list')})
    out = collect_derivatives(deriv_root, spec=spec, subject_id='01')
    assert len(out['preproc']) == 2  # matches both T1w and T2w


def test_cohort_substitution():
    from nipost.bids.spec import substitute_space

    assert substitute_space('MNIInfant:cohort-1') == 'MNIInfant+1'


def test_fieldmap_id_sanitized():
    from nipost.bids.spec import sanitize_fieldmap_id

    assert sanitize_fieldmap_id('auto_00000') == 'auto00000'


def test_boldref2fmap_list_zero_matches(func_root):
    """list cardinality with zero matches must return [] not None (key must be present)."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    # Spec with only boldref2fmap (list cardinality).
    # The func_root has a desc-coreg boldref->T1w file but NO no-desc fmap transform
    # when desc=None (must be absent) is applied and fieldmap_id is a specific value
    # that does not exist in the tree → zero matches expected.
    spec = Spec(
        transforms={
            'boldref2fmap': Query(
                [
                    {
                        'from': 'boldref',
                        'to': '{fieldmap_id}',
                        'desc': None,
                        'suffix': 'xfm',
                    }
                ],
                'list',
            ),
        },
    )
    # Pass a fieldmap_id that does not exist in the tree (no file has to=nonexistent)
    out = collect_derivatives(
        func_root,
        spec=spec,
        subject_id='01',
        entities={'task': 'rest'},
        fieldmap_id='nonexistent',
    )

    # Key must be present even with zero matches
    assert 'boldref2fmap' in out['transforms'], (
        'boldref2fmap key was dropped (got None instead of [])'
    )
    assert out['transforms']['boldref2fmap'] == [], (
        f'Expected [], got {out["transforms"]["boldref2fmap"]!r}'
    )


@pytest.fixture
def empty_root(tmp_path):
    """A valid but empty derivative dataset, for tests that add their own files."""
    root = tmp_path / 'empty'
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
    return root


def test_first_matching_alternative_wins(empty_root):
    """Alternatives are tried in order; the first with any match is used alone."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_desc-coreg_boldref.nii.gz')

    spec = Spec(
        items={
            'run_boldref': Query(
                [
                    {'space': 'run', 'suffix': 'boldref'},
                    {'desc': 'coreg', 'suffix': 'boldref'},
                ],
                'single',
            )
        }
    )
    out = collect_derivatives(empty_root, spec=spec, subject_id='01', entities={'task': 'rest'})

    # Not a union across alternatives: only the first alternative's match.
    assert out['run_boldref'].endswith('space-run_boldref.nii.gz')


def test_later_alternative_used_when_earlier_misses(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_desc-coreg_boldref.nii.gz')

    spec = Spec(
        items={
            'run_boldref': Query(
                [
                    {'space': 'run', 'suffix': 'boldref'},
                    {'desc': 'coreg', 'suffix': 'boldref'},
                ],
                'single',
            )
        }
    )
    out = collect_derivatives(empty_root, spec=spec, subject_id='01', entities={'task': 'rest'})

    assert out['run_boldref'].endswith('desc-coreg_boldref.nii.gz')


def test_scope_drops_caller_entities(empty_root):
    """A subject-level file is only reachable when run-level entities are dropped."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz')

    alternatives = [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}]
    spec = Spec(
        items={
            'scoped': Query(alternatives, 'single', scope=['subject']),
            'unscoped': Query(alternatives, 'single'),
        }
    )
    out = collect_derivatives(
        empty_root,
        spec=spec,
        subject_id='01',
        entities={'task': 'rest', 'run': '01'},
    )

    assert out['scoped'].endswith('space-subject_boldref.nii.gz')
    assert 'unscoped' not in out


def test_scope_keeps_listed_entities(empty_root):
    """`scope` is an allowlist, not a blanket drop: listed entities still filter."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz')

    spec = Spec(
        items={
            'other_subject': Query(
                [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}],
                'single',
                scope=['subject'],
            )
        }
    )
    out = collect_derivatives(empty_root, spec=spec, subject_id='02', entities={'task': 'rest'})

    assert 'other_subject' not in out


def test_spec_entities_override_caller_entities(empty_root):
    """A caller may pass raw source-file entities without stripping them first."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_task-rest_space-run_boldref.nii.gz')

    spec = Spec(
        items={
            'run_boldref': Query(
                [
                    {
                        'datatype': 'func',
                        'space': 'run',
                        'suffix': 'boldref',
                        'extension': ['.nii.gz', '.nii'],
                    }
                ],
                'single',
            )
        }
    )
    # Entities as they come off a raw BOLD BIDSFile: suffix/extension describe
    # the *source*, not the derivative being looked up.
    out = collect_derivatives(
        empty_root,
        spec=spec,
        subject_id='01',
        entities={
            'task': 'rest',
            'datatype': 'func',
            'suffix': 'bold',
            'extension': '.nii.gz',
        },
    )

    assert out['run_boldref'].endswith('space-run_boldref.nii.gz')


def test_none_requires_entity_to_be_absent(empty_root):
    """`None` is PyBIDS Query.NONE: the entity must be ABSENT, not unconstrained."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    anat = empty_root / 'sub-01' / 'anat'
    _write(anat / 'sub-01_dseg.nii.gz')
    _write(anat / 'sub-01_desc-aseg_dseg.nii.gz')

    spec = Spec(items={'dseg': Query([{'suffix': 'dseg', 'desc': None}], 'single')})
    out = collect_derivatives(empty_root, spec=spec, subject_id='01')

    assert out['dseg'].endswith('sub-01_dseg.nii.gz')


def test_none_inside_a_value_list_means_absent(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_space-session_boldref.nii.gz')

    spec = Spec(
        items={
            'either': Query([{'space': ['run', None], 'suffix': 'boldref'}], 'list'),
        }
    )
    out = collect_derivatives(empty_root, spec=spec, subject_id='01', entities={'task': 'rest'})

    names = sorted(p.rsplit('/', 1)[-1] for p in out['either'])
    assert names == [
        'sub-01_task-rest_boldref.nii.gz',
        'sub-01_task-rest_space-run_boldref.nii.gz',
    ]


def test_resolve_converts_none_list_members_to_query_none():
    """PyBIDS already treats a raw `None` inside a value list as absence, so an
    end-to-end test cannot distinguish whether `_resolve` performs this
    conversion itself. Test `_resolve` directly instead.
    """
    from bids.layout import Query as PyBIDSQuery

    from nipost.bids.collect import _resolve

    resolved = _resolve({'space': ['run', None]}, base={}, scope=None, fieldmap_id=None)

    assert resolved['space'] == ['run', PyBIDSQuery.NONE]


def test_space_placeholder_substitutes(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(
        space_transforms={
            'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}], 'single'),
        }
    )
    out = collect_derivatives(
        deriv_root, spec=spec, subject_id='01', std_spaces=['MNI152NLin2009cAsym']
    )

    assert out['transforms']['MNI152NLin2009cAsym']['forward'].endswith(
        'to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    )


def test_space_placeholder_substitutes_cohort(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    _write(empty_root / 'sub-01' / 'anat' / 'sub-01_from-T1w_to-MNIInfant+1_mode-image_xfm.h5')

    spec = Spec(
        space_transforms={
            'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}], 'single'),
        }
    )
    out = collect_derivatives(
        empty_root, spec=spec, subject_id='01', std_spaces=['MNIInfant:cohort-1']
    )

    assert out['transforms']['MNIInfant:cohort-1']['forward'].endswith(
        'to-MNIInfant+1_mode-image_xfm.h5'
    )


def test_space_placeholder_outside_space_transforms_raises(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(items={'bad': Query([{'space': '{space}', 'suffix': 'boldref'}], 'single')})

    with pytest.raises(ValueError, match='space_transforms'):
        collect_derivatives(empty_root, spec=spec, subject_id='01')


def test_single_cardinality_raises_on_ambiguity(deriv_root):
    """Two matches for a scalar item is a malformed dataset, not a list result."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    # deriv_root has both desc-preproc_T1w and desc-preproc_T2w
    spec = Spec(
        items={
            'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}], 'single'),
        }
    )

    with pytest.raises(ValueError, match='preproc'):
        collect_derivatives(deriv_root, spec=spec, subject_id='01')


def test_single_cardinality_returns_scalar(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(items={'t1w': Query([{'suffix': 'T1w', 'desc': 'preproc'}], 'single')})
    out = collect_derivatives(deriv_root, spec=spec, subject_id='01')

    assert isinstance(out['t1w'], str)


def test_optional_cardinality_is_rejected(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Query, Spec

    spec = Spec(items={'t1w': Query([{'suffix': 'T1w', 'desc': 'preproc'}], 'optional')})

    with pytest.raises(ValueError, match='Unknown cardinality'):
        collect_derivatives(deriv_root, spec=spec, subject_id='01')


def test_image_queries_ignore_json_sidecars(empty_root):
    """PyBIDS indexes a derivative's sidecar as its own file with the same entities."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import load_spec

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

    out = collect_derivatives(empty_root, spec=load_spec('anat'), subject_id='01')

    assert out['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert out['mask'].endswith('desc-brain_mask.nii.gz')
