# tests/bids/test_func_spec.py
"""The shipped func.json against fMRIPrep output at each --bold-coreg-level.

Each dataset below is a realistic slice of what fMRIPrep writes. Group-level
files (space-session / space-subject) carry no run-level entities, which is
what the queries' ``scope`` allowlists exist to handle.
"""

import json

import pytest

pytest.importorskip('bids')

# Entities as a consumer holds them: one BOLD run, unstripped.
RUN_ENTITIES = {
    'subject': '01',
    'session': 'A',
    'task': 'rest',
    'run': '01',
    'datatype': 'func',
    'suffix': 'bold',
    'extension': '.nii.gz',
}

CURRENT_RUN = (
    'sub-01_ses-A_task-rest_run-01_space-orig_desc-hmc_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_space-run_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_from-orig_to-run_mode-image_desc-hmc_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-run_to-T1w_mode-image_desc-coreg_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-run_to-auto00000_mode-image_desc-fmap_xfm.txt',
)

LEGACY_RUN = (
    'sub-01_ses-A_task-rest_run-01_desc-hmc_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_desc-coreg_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-boldref_to-auto00000_mode-image_xfm.txt',
)

# Per-run HMC plus a session template; no per-run boldref->anat transform.
SESSION_LEVEL = (
    'sub-01_ses-A_task-rest_run-01_space-orig_desc-hmc_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_space-run_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_from-orig_to-run_mode-image_desc-hmc_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-run_to-session_mode-image_desc-coreg_xfm.txt',
    'sub-01_ses-A_space-session_boldref.nii.gz',
    'sub-01_ses-A_from-session_to-anat_mode-image_desc-coreg_xfm.txt',
)

SUBJECT_LEVEL = (
    'sub-01_ses-A_task-rest_run-01_space-orig_desc-hmc_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_space-run_boldref.nii.gz',
    'sub-01_ses-A_task-rest_run-01_from-orig_to-run_mode-image_desc-hmc_xfm.txt',
    'sub-01_ses-A_task-rest_run-01_from-run_to-subject_mode-image_desc-coreg_xfm.txt',
    'sub-01_space-subject_boldref.nii.gz',
    'sub-01_from-subject_to-anat_mode-image_desc-coreg_xfm.txt',
)


def _dataset(root, *names):
    """Build a derivative dataset holding exactly ``names``.

    Files whose name contains ``_ses-A`` are placed under ``sub-01/ses-A/func``;
    the rest under ``sub-01/func``, mirroring how fMRIPrep writes subject-level
    group outputs.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / 'dataset_description.json').write_text(
        json.dumps(
            {
                'Name': 'x',
                'BIDSVersion': '1.8.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'fMRIPrep'}],
            }
        )
    )
    for name in names:
        parts = ['sub-01'] + (['ses-A'] if '_ses-A' in name else []) + ['func']
        path = root.joinpath(*parts, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    return root


def _collect(root, **kwargs):
    from nipost.bids.collect import collect_derivatives
    from nipost.bids.spec import load_spec

    return collect_derivatives(root, spec=load_spec('func'), entities=RUN_ENTITIES, **kwargs)


def test_current_naming_run_level(tmp_path):
    out = _collect(_dataset(tmp_path / 'd', *CURRENT_RUN))
    transforms = out['transforms']

    assert out['hmc_boldref'].endswith('space-orig_desc-hmc_boldref.nii.gz')
    assert out['run_boldref'].endswith('run-01_space-run_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-run_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-run_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert transforms['run2fmap'][0].endswith('from-run_to-auto00000_mode-image_desc-fmap_xfm.txt')
    # No template level in this dataset.
    assert 'session_boldref' not in out
    assert 'subject_boldref' not in out
    assert 'run2session' not in transforms
    assert 'session2anat' not in transforms


def test_legacy_naming_run_level(tmp_path):
    out = _collect(_dataset(tmp_path / 'd', *LEGACY_RUN))
    transforms = out['transforms']

    assert out['hmc_boldref'].endswith('run-01_desc-hmc_boldref.nii.gz')
    assert out['run_boldref'].endswith('run-01_desc-coreg_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert transforms['run2fmap'][0].endswith('to-auto00000_mode-image_xfm.txt')


def test_current_naming_wins_over_legacy(tmp_path):
    """Both namings present: the first alternative is used, not the union."""
    out = _collect(_dataset(tmp_path / 'd', *CURRENT_RUN, *LEGACY_RUN))
    transforms = out['transforms']

    assert out['hmc_boldref'].endswith('space-orig_desc-hmc_boldref.nii.gz')
    assert out['run_boldref'].endswith('run-01_space-run_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-run_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-run_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert 'desc-fmap' in transforms['run2fmap'][0]


def test_session_level(tmp_path):
    out = _collect(_dataset(tmp_path / 'd', *SESSION_LEVEL))
    transforms = out['transforms']

    # Reachable only because the queries drop task/run via `scope`.
    assert out['session_boldref'].endswith('sub-01_ses-A_space-session_boldref.nii.gz')
    assert transforms['session2anat'].endswith(
        'from-session_to-anat_mode-image_desc-coreg_xfm.txt'
    )
    # Per-run, so it keeps the full entity set.
    assert transforms['run2session'].endswith('from-run_to-session_mode-image_desc-coreg_xfm.txt')
    # Coregistration is not per-run at this level.
    assert 'run2anat' not in transforms
    assert 'subject_boldref' not in out
    assert 'subject2anat' not in transforms


def test_subject_level(tmp_path):
    out = _collect(_dataset(tmp_path / 'd', *SUBJECT_LEVEL))
    transforms = out['transforms']

    assert out['subject_boldref'].endswith('sub-01_space-subject_boldref.nii.gz')
    assert transforms['subject2anat'].endswith(
        'from-subject_to-anat_mode-image_desc-coreg_xfm.txt'
    )
    assert transforms['run2subject'].endswith('from-run_to-subject_mode-image_desc-coreg_xfm.txt')
    assert 'run2anat' not in transforms
    assert 'session_boldref' not in out
    assert 'session2anat' not in transforms


def test_half_written_level_yields_no_usable_pair(tmp_path):
    """run2session without session2anat must not look like a valid chain."""
    names = tuple(n for n in SESSION_LEVEL if 'from-session_to-anat' not in n)
    out = _collect(_dataset(tmp_path / 'd', *names))
    transforms = out['transforms']

    assert 'run2session' in transforms
    assert 'session2anat' not in transforms
    levels = [
        level
        for level in ('session', 'subject')
        if f'run2{level}' in transforms and f'{level}2anat' in transforms
    ]
    assert levels == []


def test_run2fmap_selects_by_fieldmap_id(tmp_path):
    names = (
        *CURRENT_RUN,
        'sub-01_ses-A_task-rest_run-01_from-run_to-auto00001_mode-image_desc-fmap_xfm.txt',
    )
    out = _collect(_dataset(tmp_path / 'd', *names), fieldmap_id='auto_00001')

    assert len(out['transforms']['run2fmap']) == 1
    assert out['transforms']['run2fmap'][0].endswith('to-auto00001_mode-image_desc-fmap_xfm.txt')


def test_run2fmap_never_returns_a_coreg_transform(tmp_path):
    """Called without a fieldmap_id, run2fmap must not pick up boldref->anat."""
    out = _collect(_dataset(tmp_path / 'd', *CURRENT_RUN))

    for path in out['transforms']['run2fmap']:
        assert 'desc-coreg' not in path
        assert 'to-T1w' not in path
