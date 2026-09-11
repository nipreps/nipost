# tests/bids/test_collect_derivatives.py
import os
from pathlib import Path

import pytest

pytest.importorskip('bids')
pytest.importorskip('niworkflows')


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('')


@pytest.fixture
def deriv_root(tmp_path, deriv_dataset) -> Path:
    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    # dual T1w + T2w preproc (space-qualified case)
    _write(anat / 'sub-01_desc-preproc_T1w.nii.gz')
    _write(anat / 'sub-01_desc-preproc_T2w.nii.gz')
    # ordered TPMs
    for label in ('GM', 'WM', 'CSF'):
        _write(anat / f'sub-01_label-{label}_probseg.nii.gz')
    # surface pair, ordered by hemi
    _write(anat / 'sub-01_hemi-L_white.surf.gii')
    _write(anat / 'sub-01_hemi-R_white.surf.gii')
    # single masks
    _write(anat / 'sub-01_desc-ribbon_mask.nii.gz')
    _write(anat / 'sub-01_desc-brain_mask.nii.gz')
    # coreg + normalization transforms
    _write(anat / 'sub-01_from-T1w_to-T2w_mode-image_xfm.txt')
    _write(anat / 'sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
    return root


def test_collect_covers_case_catalog(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

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
                'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}]),
            },
        ),
    }

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01'},
        params={'space': ['MNI152NLin2009cAsym']},
    )

    images = out['images']
    assert images['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert images['t2w_preproc'].endswith('desc-preproc_T2w.nii.gz')
    assert [p.split('label-')[1][:2] for p in images['tpms']] == ['GM', 'WM', 'CS']
    # 'pair' sorted by path and got L before R alphabetically; order='hemi'
    # makes that explicit, so assert the order rather than just the length.
    assert [p.split('hemi-')[1][0] for p in images['white']] == ['L', 'R']
    assert isinstance(images['ribbon'], str)
    assert images['t1w2t2w'].endswith('from-T1w_to-T2w_mode-image_xfm.txt')
    assert out['transforms']['MNI152NLin2009cAsym']['forward'].endswith('_xfm.h5')


@pytest.fixture
def func_root(tmp_path, deriv_dataset):
    root = deriv_dataset(tmp_path / 'fderiv')
    func = root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_desc-hmc_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    _write(func / 'sub-01_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    _write(func / 'sub-01_task-rest_from-boldref_to-auto00000_mode-image_xfm.txt')
    return root


@pytest.fixture
def empty_root(tmp_path, deriv_dataset):
    """A valid but empty derivative dataset, for tests that add their own files."""
    return deriv_dataset(tmp_path / 'empty')


def test_func_flat_transforms_and_boldref2fmap_list(func_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'images': Group(queries={'hmc_boldref': Query([{'desc': 'hmc', 'suffix': 'boldref'}])}),
        'transforms': Group(
            queries={
                'hmc': Query([{'from': 'orig', 'to': 'boldref', 'suffix': 'xfm'}]),
                'boldref2anat': Query([{'from': 'boldref', 'to': 'T1w', 'suffix': 'xfm'}]),
                'boldref2fmap': Query(
                    [{'from': 'boldref', 'to': '{fmapid}', 'desc': None, 'suffix': 'xfm'}],
                    multi=True,
                ),
            }
        ),
    }
    # Called WITHOUT fmapid, exactly as the notebook does. boldref2fmap must
    # return only the fmap transform (desc absent), NOT the desc-coreg file.
    out = collect_derivatives(func_root, spec=spec, entities={'subject': '01', 'task': 'rest'})

    # FLAT transform dict: keys sit directly under 'transforms'
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
    from nipost.bids.spec import Group, Query

    spec = {
        'images': Group(
            queries={'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}], multi=True)}
        )
    }
    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})
    assert len(out['images']['preproc']) == 2  # matches both T1w and T2w


# --- groups and params ---


def test_plain_group_nests_results_under_its_name(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'images': Group(
            queries={'mask': Query([{'datatype': 'anat', 'suffix': 'mask', 'desc': 'brain'}])}
        )
    }

    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})

    assert set(out) == {'images'}
    assert out['images']['mask'].endswith('desc-brain_mask.nii.gz')


def test_declared_groups_appear_even_when_they_collect_nothing(empty_root):
    """Groups are structural -- declared by the spec, not by the data -- so a
    caller can walk the output without checking whether each key exists."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {'images': Group(queries={'nope': Query([{'suffix': 'Nonexistent'}])})}

    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert out == {'images': {}}


def test_incomplete_ordered_result_omits_its_key(tmp_path, deriv_dataset):
    """The reducer's omit-on-incomplete rule reaching the public output.

    Pinned end to end because the unit tests cover _reduce in isolation, and
    the key's absence from the group dict is what callers actually see.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    anat.mkdir(parents=True)
    # GM and WM only -- CSF is missing, so the tuple is incomplete.
    for label in ('GM', 'WM'):
        _write(anat / f'sub-01_label-{label}_probseg.nii.gz')

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

    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out == {'images': {}}


def test_ordered_query_matching_zero_files_omits_its_key(tmp_path, deriv_dataset):
    """The zero-match path through `_lookup`, distinct from the partial-match
    path `test_incomplete_ordered_result_omits_its_key` exercises above: no
    probseg files exist at all, so no alternative matches anything and
    `_lookup` never reaches `_reduce_ordered`."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    anat.mkdir(parents=True)
    # No probseg files at all -- the query matches nothing.

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

    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out == {'images': {}}


def test_ordered_result_accepts_a_null_member_for_the_label_less_file(tmp_path, deriv_dataset):
    """``None`` is a legal member of an ordering value list: it means "the file
    lacking this entity sorts here". One probseg carries `label-GM`, the other
    carries no `label` entity at all, and both sort into the declared order.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    anat.mkdir(parents=True)
    _write(anat / 'sub-01_label-GM_probseg.nii.gz')
    _write(anat / 'sub-01_probseg.nii.gz')

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

    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out['images']['tpms'] == [
        str(anat / 'sub-01_label-GM_probseg.nii.gz'),
        str(anat / 'sub-01_probseg.nii.gz'),
    ]


def test_indexed_group_nests_under_each_param_value(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}])},
        )
    }

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01'},
        params={'space': ['MNI152NLin2009cAsym']},
    )

    assert set(out['transforms']) == {'MNI152NLin2009cAsym'}
    assert out['transforms']['MNI152NLin2009cAsym']['forward'].endswith('_xfm.h5')


def test_indexed_group_with_no_param_values_is_empty(deriv_root):
    """This is what 'an unbound space is an empty list' means concretely."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}])},
        )
    }

    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})

    assert out == {'transforms': {}}


def test_indexed_group_rejects_a_bare_string_param(deriv_root):
    """Iterating a string's characters would silently query for 'M', 'N', 'I'..."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}])},
        )
    }

    with pytest.raises(TypeError, match='space'):
        collect_derivatives(
            deriv_root,
            spec=spec,
            entities={'subject': '01'},
            params={'space': 'MNI152NLin2009cAsym'},
        )


def test_indexed_group_rejects_a_non_iterable_param(empty_root):
    """A non-iterable slips past a bare `str` check and dies unhelpfully deep
    inside the dict comprehension -- the right exception type, but naming
    neither the group nor the parameter, which is the guard's whole purpose."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}])},
        )
    }

    with pytest.raises(TypeError, match='space'):
        collect_derivatives(
            empty_root,
            spec=spec,
            entities={'subject': '01'},
            params={'space': 7},
        )


def test_output_key_is_the_param_value_verbatim(deriv_root):
    """The resolver transforms nothing: what you pass is what you match and
    what you index. Cohort conversion is the caller's job."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}])},
        )
    }

    out = collect_derivatives(
        deriv_root,
        spec=spec,
        entities={'subject': '01'},
        params={'space': ['MNI152NLin2009cAsym']},
    )

    assert list(out['transforms']) == ['MNI152NLin2009cAsym']


def test_resolver_does_not_normalize_param_values():
    """Design 5.2: values pass through verbatim; normalizing is the caller's job.

    Asserted with a cohort space, whose sanitized form differs from its raw
    form -- the existing group-level test uses a space that sanitizes to
    itself, so it cannot detect a resolver that normalizes.
    """
    from nipost.bids.collect import _resolve

    out = _resolve({'to': '{space}'}, {}, None, {'space': 'MNI152NLin6Asym:cohort-1'})

    assert out['to'] == ['MNI152NLin6Asym:cohort-1']


def test_unbound_placeholder_drops_its_constraint(func_root):
    """run2fmap without a fmapid must match every run-to-fieldmap transform."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            queries={
                'run2fmap': Query(
                    [
                        {
                            'datatype': 'func',
                            'from': 'boldref',
                            'to': '{fmapid}',
                            'desc': None,
                            'suffix': 'xfm',
                            'extension': '.txt',
                        }
                    ],
                    multi=True,
                )
            }
        )
    }

    out = collect_derivatives(func_root, spec=spec, entities={'subject': '01'})

    assert out['transforms']['run2fmap'] != []


def test_bound_placeholder_filters(func_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            queries={
                'run2fmap': Query(
                    [
                        {
                            'datatype': 'func',
                            'from': 'boldref',
                            'to': '{fmapid}',
                            'desc': None,
                            'suffix': 'xfm',
                            'extension': '.txt',
                        }
                    ],
                    multi=True,
                )
            }
        )
    }

    out = collect_derivatives(
        func_root,
        spec=spec,
        entities={'subject': '01'},
        params={'fmapid': 'nonexistent'},
    )

    assert out['transforms']['run2fmap'] == []


def test_boldref2fmap_list_zero_matches(func_root):
    """A multi query with zero matches must return [] not None (key must be present)."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            queries={
                'boldref2fmap': Query(
                    [{'from': 'boldref', 'to': '{fmapid}', 'desc': None, 'suffix': 'xfm'}],
                    multi=True,
                )
            }
        )
    }
    # Pass a fmapid that does not exist in the tree (no file has to=nonexistent)
    out = collect_derivatives(
        func_root,
        spec=spec,
        entities={'subject': '01', 'task': 'rest'},
        params={'fmapid': 'nonexistent'},
    )

    # Key must be present even with zero matches
    assert 'boldref2fmap' in out['transforms'], (
        'boldref2fmap key was dropped (got None instead of [])'
    )
    assert out['transforms']['boldref2fmap'] == [], (
        f'Expected [], got {out["transforms"]["boldref2fmap"]!r}'
    )


def test_first_matching_alternative_wins(empty_root):
    """Alternatives are tried in order; the first with any match is used alone."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
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
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01', 'task': 'rest'})

    # Not a union across alternatives: only the first alternative's match.
    assert out['images']['run_boldref'].endswith('space-run_boldref.nii.gz')


def test_later_alternative_used_when_earlier_misses(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

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
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01', 'task': 'rest'})

    assert out['images']['run_boldref'].endswith('desc-coreg_boldref.nii.gz')


def test_scope_drops_caller_entities(empty_root):
    """A subject-level file is only reachable when run-level entities are dropped."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz')

    entity_alts = [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}]
    spec = {
        'images': Group(
            queries={
                'scoped': Query(entity_alts, scope=['subject']),
                'unscoped': Query(entity_alts),
            }
        )
    }
    out = collect_derivatives(
        empty_root,
        spec=spec,
        entities={'subject': '01', 'task': 'rest', 'run': '01'},
    )

    assert out['images']['scoped'].endswith('space-subject_boldref.nii.gz')
    assert 'unscoped' not in out['images']


def test_empty_scope_accepts_no_caller_entities(empty_root):
    """`scope: []` is an empty allowlist, not an absent one.

    The caller asks for `sub-02` and the only file is `sub-01`. An empty scope
    drops the subject constraint along with everything else, so the query
    matches on its spec entities alone and finds the file. Treating `[]` as
    `None` would keep the constraint and find nothing -- and would do it
    silently, absence being a legal outcome.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz')

    spec = {
        'images': Group(
            queries={
                'any_subject': Query(
                    [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}],
                    scope=[],
                )
            }
        )
    }
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '02', 'run': '01'})

    assert out['images']['any_subject'].endswith('space-subject_boldref.nii.gz')


def test_scope_keeps_listed_entities(empty_root):
    """`scope` is an allowlist, not a blanket drop: listed entities still filter."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_space-subject_boldref.nii.gz')

    spec = {
        'images': Group(
            queries={
                'other_subject': Query(
                    [{'datatype': 'func', 'space': 'subject', 'suffix': 'boldref'}],
                    scope=['subject'],
                )
            }
        )
    }
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '02', 'task': 'rest'})

    assert 'other_subject' not in out['images']


def test_spec_entities_override_caller_entities(empty_root):
    """A caller may pass raw source-file entities without stripping them first."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    _write(empty_root / 'sub-01' / 'func' / 'sub-01_task-rest_space-run_boldref.nii.gz')

    spec = {
        'images': Group(
            queries={
                'run_boldref': Query(
                    [
                        {
                            'datatype': 'func',
                            'space': 'run',
                            'suffix': 'boldref',
                            'extension': ['.nii.gz', '.nii'],
                        }
                    ]
                )
            }
        )
    }
    # Entities as they come off a raw BOLD BIDSFile: suffix/extension describe
    # the *source*, not the derivative being looked up.
    out = collect_derivatives(
        empty_root,
        spec=spec,
        entities={
            'subject': '01',
            'task': 'rest',
            'datatype': 'func',
            'suffix': 'bold',
            'extension': '.nii.gz',
        },
    )

    assert out['images']['run_boldref'].endswith('space-run_boldref.nii.gz')


def test_none_requires_entity_to_be_absent(empty_root):
    """`None` is PyBIDS Query.NONE: the entity must be ABSENT, not unconstrained."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    anat = empty_root / 'sub-01' / 'anat'
    _write(anat / 'sub-01_dseg.nii.gz')
    _write(anat / 'sub-01_desc-aseg_dseg.nii.gz')

    spec = {'images': Group(queries={'dseg': Query([{'suffix': 'dseg', 'desc': None}])})}
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01'})

    assert out['images']['dseg'].endswith('sub-01_dseg.nii.gz')


def test_none_inside_a_value_list_means_absent(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    func = empty_root / 'sub-01' / 'func'
    _write(func / 'sub-01_task-rest_space-run_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_boldref.nii.gz')
    _write(func / 'sub-01_task-rest_space-session_boldref.nii.gz')

    spec = {
        'images': Group(
            queries={'either': Query([{'space': ['run', None], 'suffix': 'boldref'}], multi=True)}
        )
    }
    out = collect_derivatives(empty_root, spec=spec, entities={'subject': '01', 'task': 'rest'})

    names = sorted(os.path.basename(p) for p in out['images']['either'])
    assert names == [
        'sub-01_task-rest_boldref.nii.gz',
        'sub-01_task-rest_space-run_boldref.nii.gz',
    ]


def test_resolve_substitutes_fieldmap_id_placeholder_inside_a_list():
    from nipost.bids.collect import _resolve

    resolved = _resolve(
        {'to': ['{fmapid}', 'auto00000']}, base={}, scope=None, params={'fmapid': 'auto00001'}
    )

    assert resolved['to'] == ['auto00001', 'auto00000']


def test_resolve_drops_fieldmap_id_placeholder_member_when_missing():
    """Dropping just the placeholder member is the consistent reading of a list."""
    from nipost.bids.collect import _resolve

    resolved = _resolve({'to': ['{fmapid}', 'auto00000']}, base={}, scope=None, params={})

    assert resolved['to'] == ['auto00000']


def test_resolve_drops_whole_constraint_when_list_placeholder_empties_it():
    """If dropping the placeholder member would empty the list, drop the whole
    constraint instead of passing `[]` to PyBIDS (which would match nothing)."""
    from nipost.bids.collect import _resolve

    resolved = _resolve({'to': ['{fmapid}']}, base={}, scope=None, params={})

    assert 'to' not in resolved


def test_resolve_raises_when_a_list_param_binds_a_query_level_placeholder():
    """A query-level placeholder (not bound by an enclosing `per`) must take
    `params[name]` as a scalar. Without this check, a list param silently
    becomes a *nested* list in the PyBIDS filter dict -- `{'to': [['a', 'b']]}`
    -- which PyBIDS matches nothing against and raises nothing for."""
    from nipost.bids.collect import _resolve

    with pytest.raises(TypeError, match='fmapid'):
        _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': ['a', 'b']})


def test_resolve_raise_names_the_placeholder_not_its_resolved_value():
    """The message has to point at the spec text the author must go edit.

    Interpolating the resolved value instead reads
    `params['fmapid'] must be a scalar for placeholder ['a', 'b']`, which is
    self-contradictory and names nothing findable in the spec.
    """
    from nipost.bids.collect import _resolve

    with pytest.raises(TypeError, match=r'\{fmapid\}'):
        _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': ['a', 'b']})


def test_resolve_substitutes_a_scalar_param_into_a_scalar_placeholder():
    from nipost.bids.collect import _resolve

    resolved = _resolve({'to': '{fmapid}'}, base={}, scope=None, params={'fmapid': 'auto00001'})

    assert resolved['to'] == ['auto00001']


def test_resolve_drops_an_unbound_scalar_placeholder():
    from nipost.bids.collect import _resolve

    resolved = _resolve({'to': '{fmapid}'}, base={}, scope=None, params={})

    assert 'to' not in resolved


def test_space_placeholder_substitutes(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {
        'transforms': Group(
            per='space',
            queries={
                'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}]),
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


def test_space_placeholder_substitutes_cohort(empty_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query, sanitize_space

    _write(empty_root / 'sub-01' / 'anat' / 'sub-01_from-T1w_to-MNIInfant+1_mode-image_xfm.h5')

    spec = {
        'transforms': Group(
            per='space',
            queries={
                'forward': Query([{'from': 'T1w', 'to': '{space}', 'suffix': 'xfm'}]),
            },
        )
    }
    # Cohort conversion is caller-side now: sanitize before passing to params.
    space = sanitize_space('MNIInfant:cohort-1')
    out = collect_derivatives(
        empty_root, spec=spec, entities={'subject': '01'}, params={'space': [space]}
    )

    assert out['transforms'][space]['forward'].endswith('to-MNIInfant+1_mode-image_xfm.h5')


def test_scalar_query_raises_on_ambiguity(deriv_root):
    """Two matches for a scalar item is a malformed dataset, not a list result."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    # deriv_root has both desc-preproc_T1w and desc-preproc_T2w
    spec = {
        'images': Group(
            queries={'preproc': Query([{'suffix': ['T1w', 'T2w'], 'desc': 'preproc'}])}
        )
    }

    with pytest.raises(ValueError, match='preproc'):
        collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})


def test_scalar_query_returns_a_path(deriv_root):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Query

    spec = {'images': Group(queries={'t1w': Query([{'suffix': 'T1w', 'desc': 'preproc'}])})}
    out = collect_derivatives(deriv_root, spec=spec, entities={'subject': '01'})

    assert isinstance(out['images']['t1w'], str)


def test_cardinality_field_is_rejected():
    """The field is gone entirely, so passing it is a constructor error now,
    not a bad enum member caught later inside the reducer."""
    from nipost.bids.spec import Query

    with pytest.raises(TypeError, match='cardinality'):
        Query([{'suffix': 'T1w', 'desc': 'preproc'}], cardinality='optional')  # type: ignore


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
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import load_spec

    out = collect_derivatives(
        anat_with_std_space_dupes, spec=load_spec('anat'), entities={'subject': '01'}
    )

    images = out['images']
    for key in ('t1w_preproc', 't2w_preproc', 'mask', 'dseg'):
        assert 'space-' not in images[key], f'{key}: expected native file, got {images[key]!r}'
    assert all('space-' not in p for p in images['tpms']), (
        f'tpms: expected native files, got {images["tpms"]!r}'
    )
    assert [p.rsplit('label-', 1)[1][:2] for p in images['tpms']] == ['GM', 'WM', 'CS']


class _StubFile:
    """A minimal stand-in for a PyBIDS BIDSFile, for testing `_reduce` directly.

    `_reduce` is a pure reducer over `(path, entities)` pairs, so its
    ambiguity and absence rules are pinned here rather than end to end.
    """

    def __init__(self, path, **entities):
        self.path = path
        self.entities = entities


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

    out = collect_derivatives(empty_root, spec=load_spec('anat'), entities={'subject': '01'})

    assert out['images']['t1w_preproc'].endswith('desc-preproc_T1w.nii.gz')
    assert out['images']['mask'].endswith('desc-brain_mask.nii.gz')


def test_reduce_scalar_returns_the_single_path():
    from nipost.bids.collect import _reduce

    result = _reduce('mask', [_StubFile('/x/sub-01_desc-brain_mask.nii.gz')], multi=False)

    assert result == '/x/sub-01_desc-brain_mask.nii.gz'


def test_reduce_scalar_returns_none_when_absent():
    """Absence is never an error: precomputed derivatives are whatever exists."""
    from nipost.bids.collect import _reduce

    assert _reduce('mask', [], multi=False) is None


def test_reduce_scalar_raises_on_ambiguity():
    """Two matches for a scalar item means a malformed dataset."""
    from nipost.bids.collect import _reduce

    files = [_StubFile('/x/a_mask.nii.gz'), _StubFile('/x/b_mask.nii.gz')]

    with pytest.raises(ValueError, match='mask'):
        _reduce('mask', files, multi=False)


def test_reduce_multi_returns_empty_list_when_absent():
    """A multi query's key is always present, so callers can index it."""
    from nipost.bids.collect import _reduce

    assert _reduce('coeffs', [], multi=True) == []


def test_reduce_multi_sorts_naturally_not_lexically():
    """reconstruct_fieldmap reads coeffs[-1] as the finest B-spline level, so
    coeff2 must precede coeff10 rather than following it."""
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/desc-coeff10_fieldmap.nii.gz'),
        _StubFile('/x/desc-coeff2_fieldmap.nii.gz'),
    ]

    result = _reduce('coeffs', files, multi=True)

    assert result == [
        '/x/desc-coeff2_fieldmap.nii.gz',
        '/x/desc-coeff10_fieldmap.nii.gz',
    ]


def test_reduce_ordered_permutes_to_declared_order():
    """Order follows the declared value sequence, not the filesystem or the alphabet."""
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/label-CSF_probseg.nii.gz', label='CSF'),
        _StubFile('/x/label-GM_probseg.nii.gz', label='GM'),
        _StubFile('/x/label-WM_probseg.nii.gz', label='WM'),
    ]

    result = _reduce('tpms', files, multi=True, order='label', order_values=['GM', 'WM', 'CSF'])

    assert result == [
        '/x/label-GM_probseg.nii.gz',
        '/x/label-WM_probseg.nii.gz',
        '/x/label-CSF_probseg.nii.gz',
    ]


def test_reduce_ordered_orders_by_any_entity_not_just_label():
    """'pair' existed only for hemi-keyed surfaces; ordering generalises it."""
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/hemi-R_white.surf.gii', hemi='R'),
        _StubFile('/x/hemi-L_white.surf.gii', hemi='L'),
    ]

    result = _reduce('white', files, multi=True, order='hemi', order_values=['L', 'R'])

    assert result == ['/x/hemi-L_white.surf.gii', '/x/hemi-R_white.surf.gii']


def test_reduce_ordered_raises_on_duplicate_value():
    """Two files sharing an ordering value is a malformed dataset, not a last-wins pick."""
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/run-1_label-GM_probseg.nii.gz', label='GM'),
        _StubFile('/x/run-2_label-GM_probseg.nii.gz', label='GM'),
        _StubFile('/x/label-WM_probseg.nii.gz', label='WM'),
    ]

    with pytest.raises(ValueError, match='tpms'):
        _reduce('tpms', files, multi=True, order='label', order_values=['GM', 'WM', 'CSF'])


def test_reduce_ordered_omits_key_when_a_value_is_missing():
    """An incomplete ordered tuple is the absence of the item, not a partial item.

    This is the bug fix. The previous 'ordered' cardinality returned
    ['...WM...', '...CSF...'] here, so a caller either indexed [2] and raised
    IndexError or silently read WM where it expected GM.
    """
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/label-CSF_probseg.nii.gz', label='CSF'),
        _StubFile('/x/label-WM_probseg.nii.gz', label='WM'),
    ]

    result = _reduce('tpms', files, multi=True, order='label', order_values=['GM', 'WM', 'CSF'])

    assert result is None


def test_reduce_ordered_omits_key_when_nothing_matched():
    from nipost.bids.collect import _reduce

    assert _reduce('tpms', [], multi=True, order='label', order_values=['GM', 'WM', 'CSF']) is None


def test_reduce_ordered_ignores_files_outside_the_declared_values():
    """The entity constraint filters; order only permutes what filtering left."""
    from nipost.bids.collect import _reduce

    files = [
        _StubFile('/x/label-WM_probseg.nii.gz', label='WM'),
        _StubFile('/x/label-GM_probseg.nii.gz', label='GM'),
        _StubFile('/x/label-Other_probseg.nii.gz', label='Other'),
    ]

    result = _reduce('tpms', files, multi=True, order='label', order_values=['GM', 'WM'])

    assert result == ['/x/label-GM_probseg.nii.gz', '/x/label-WM_probseg.nii.gz']


def test_ordered_axis_accepts_a_string_for_an_int_typed_entity(tmp_path, deriv_dataset):
    """PyBIDS coerces the filter, so the axis has to coerce identically.

    The value list is read twice -- once as a filter, once as the ordering
    axis -- and PyBIDS parses ``run`` as a ``PaddedInt``. A declared ``'1'``
    matched files as a filter but missed in the axis lookup, because dict
    lookup hashes before it compares and ``hash('1')`` is not ``hash(1)``.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    for run in ('01', '02'):
        (func / f'sub-01_task-rest_run-{run}_desc-preproc_bold.nii.gz').write_bytes(b'')

    spec = {
        'bolds': Group(
            queries={
                'runs': Query(
                    entities=[
                        {
                            'datatype': 'func',
                            'suffix': 'bold',
                            'desc': 'preproc',
                            'run': ['1', '2'],
                        }
                    ],
                    multi=Ordered(order='run'),
                )
            }
        )
    }
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out['bolds']['runs'] == [
        str(func / 'sub-01_task-rest_run-01_desc-preproc_bold.nii.gz'),
        str(func / 'sub-01_task-rest_run-02_desc-preproc_bold.nii.gz'),
    ]


def test_ordered_axis_accepts_the_zero_padded_string_form(tmp_path, deriv_dataset):
    """Writing the padding explicitly is the form most likely to be reached
    for, and it was the form that silently failed."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    for run in ('01', '02'):
        (func / f'sub-01_task-rest_run-{run}_desc-preproc_bold.nii.gz').write_bytes(b'')

    spec = {
        'bolds': Group(
            queries={
                'runs': Query(
                    entities=[
                        {
                            'datatype': 'func',
                            'suffix': 'bold',
                            'desc': 'preproc',
                            'run': ['02', '01'],
                        }
                    ],
                    multi=Ordered(order='run'),
                )
            }
        )
    }
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    # Declared order is 02 then 01, and the axis permutes to match it.
    assert out['bolds']['runs'] == [
        str(func / 'sub-01_task-rest_run-02_desc-preproc_bold.nii.gz'),
        str(func / 'sub-01_task-rest_run-01_desc-preproc_bold.nii.gz'),
    ]


def test_ordered_axis_leaves_an_int_declared_value_working(tmp_path, deriv_dataset):
    """The integer form already worked; coercion must not regress it."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    for run in ('01', '02'):
        (func / f'sub-01_task-rest_run-{run}_desc-preproc_bold.nii.gz').write_bytes(b'')

    spec = {
        'bolds': Group(
            queries={
                'runs': Query(
                    entities=[
                        {
                            'datatype': 'func',
                            'suffix': 'bold',
                            'desc': 'preproc',
                            'run': [1, 2],
                        }
                    ],
                    multi=Ordered(order='run'),
                )
            }
        )
    }
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert len(out['bolds']['runs']) == 2


def test_ordered_axis_coercion_does_not_disturb_a_null_member(tmp_path, deriv_dataset):
    """``None`` means "the file lacking this entity sorts here", so it is not a
    value to coerce -- ``str(None)`` would be the string ``'None'``."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    anat = root / 'sub-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-01_label-GM_probseg.nii.gz').write_bytes(b'')
    (anat / 'sub-01_probseg.nii.gz').write_bytes(b'')

    spec = {
        'anat': Group(
            queries={
                'tpms': Query(
                    entities=[{'suffix': 'probseg', 'label': ['GM', None]}],
                    multi=Ordered(order='label'),
                )
            }
        )
    }
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out['anat']['tpms'] == [
        str(anat / 'sub-01_label-GM_probseg.nii.gz'),
        str(anat / 'sub-01_probseg.nii.gz'),
    ]


def test_ordered_axis_coercion_falls_back_when_the_value_will_not_convert(tmp_path, deriv_dataset):
    """A declared value that cannot become the parsed type is left alone and
    then simply misses, which is the pre-existing omit-the-key outcome. It must
    not raise out of the collector."""
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    (func / 'sub-01_task-rest_run-01_desc-preproc_bold.nii.gz').write_bytes(b'')

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
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert 'runs' not in out['bolds']


def test_ordered_axis_coerces_when_the_parsed_sample_is_falsy(tmp_path, deriv_dataset):
    """`run-0` parses to a falsy `PaddedInt`, which must still name the type.

    Reading the target type off the first *truthy* parsed value rather than the
    first non-`None` one skips coercion entirely here, and the axis lookup then
    misses on every value -- the silent omit-the-key failure coercion exists to
    prevent.
    """
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import Group, Ordered, Query

    root = deriv_dataset(tmp_path / 'deriv')
    func = root / 'sub-01' / 'func'
    func.mkdir(parents=True)
    for run in ('0', '1'):
        (func / f'sub-01_task-rest_run-{run}_desc-preproc_bold.nii.gz').write_bytes(b'')

    spec = {
        'bolds': Group(
            queries={
                'runs': Query(
                    entities=[
                        {
                            'datatype': 'func',
                            'suffix': 'bold',
                            'desc': 'preproc',
                            'run': ['0', '1'],
                        }
                    ],
                    multi=Ordered(order='run'),
                )
            }
        )
    }
    out = collect_derivatives(root, spec=spec, entities={'subject': '01'})

    assert out['bolds']['runs'] == [
        str(func / 'sub-01_task-rest_run-0_desc-preproc_bold.nii.gz'),
        str(func / 'sub-01_task-rest_run-1_desc-preproc_bold.nii.gz'),
    ]
