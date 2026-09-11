# tests/bids/test_func_spec.py
"""The shipped func.yml against fMRIPrep output at each --bold-coreg-level.

Each dataset below is a realistic slice of what fMRIPrep writes. Group-level
files (space-session / space-subject) carry no run-level entities, which is
what the queries' ``scope`` allowlists exist to handle.
"""

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


def _dataset(root, deriv_dataset, *names):
    """Build a derivative dataset holding exactly ``names``.

    Files whose name contains ``_ses-A`` are placed under ``sub-01/ses-A/func``;
    the rest under ``sub-01/func``, mirroring how fMRIPrep writes subject-level
    group outputs.
    """
    deriv_dataset(root, generated_by='fMRIPrep')
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


def test_current_naming_run_level(tmp_path, deriv_dataset):
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *CURRENT_RUN))
    boldrefs = out['boldrefs']
    transforms = out['transforms']

    assert boldrefs['hmc'].endswith('space-orig_desc-hmc_boldref.nii.gz')
    assert boldrefs['run'].endswith('run-01_space-run_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-run_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-run_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert transforms['run2fmap'][0].endswith('from-run_to-auto00000_mode-image_desc-fmap_xfm.txt')
    # No template level in this dataset.
    assert 'session' not in boldrefs
    assert 'subject' not in boldrefs
    assert 'run2session' not in transforms
    assert 'session2anat' not in transforms


def test_legacy_naming_run_level(tmp_path, deriv_dataset):
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *LEGACY_RUN))
    boldrefs = out['boldrefs']
    transforms = out['transforms']

    assert boldrefs['hmc'].endswith('run-01_desc-hmc_boldref.nii.gz')
    assert boldrefs['run'].endswith('run-01_desc-coreg_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-boldref_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert transforms['run2fmap'][0].endswith('to-auto00000_mode-image_xfm.txt')


def test_current_naming_wins_over_legacy(tmp_path, deriv_dataset):
    """Both namings present: the first alternative is used, not the union."""
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *CURRENT_RUN, *LEGACY_RUN))
    boldrefs = out['boldrefs']
    transforms = out['transforms']

    assert boldrefs['hmc'].endswith('space-orig_desc-hmc_boldref.nii.gz')
    assert boldrefs['run'].endswith('run-01_space-run_boldref.nii.gz')
    assert transforms['hmc'].endswith('from-orig_to-run_mode-image_desc-hmc_xfm.txt')
    assert transforms['run2anat'].endswith('from-run_to-T1w_mode-image_desc-coreg_xfm.txt')
    assert len(transforms['run2fmap']) == 1
    assert 'desc-fmap' in transforms['run2fmap'][0]


def test_session_level(tmp_path, deriv_dataset):
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *SESSION_LEVEL))
    boldrefs = out['boldrefs']
    transforms = out['transforms']

    # Reachable only because the queries drop task/run via `scope`.
    assert boldrefs['session'].endswith('sub-01_ses-A_space-session_boldref.nii.gz')
    assert transforms['session2anat'].endswith(
        'from-session_to-anat_mode-image_desc-coreg_xfm.txt'
    )
    # Per-run, so it keeps the full entity set.
    assert transforms['run2session'].endswith('from-run_to-session_mode-image_desc-coreg_xfm.txt')
    # Coregistration is not per-run at this level.
    assert 'run2anat' not in transforms
    assert 'subject' not in boldrefs
    assert 'subject2anat' not in transforms


def test_subject_level(tmp_path, deriv_dataset):
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *SUBJECT_LEVEL))
    boldrefs = out['boldrefs']
    transforms = out['transforms']

    assert boldrefs['subject'].endswith('sub-01_space-subject_boldref.nii.gz')
    assert transforms['subject2anat'].endswith(
        'from-subject_to-anat_mode-image_desc-coreg_xfm.txt'
    )
    assert transforms['run2subject'].endswith('from-run_to-subject_mode-image_desc-coreg_xfm.txt')
    assert 'run2anat' not in transforms
    assert 'session' not in boldrefs
    assert 'session2anat' not in transforms


def test_half_written_level_yields_no_usable_leg(tmp_path, deriv_dataset):
    """run2session without session2anat must not look like a valid chain."""
    names = tuple(n for n in SESSION_LEVEL if 'from-session_to-anat' not in n)
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *names))
    transforms = out['transforms']

    assert 'run2session' in transforms
    assert 'session2anat' not in transforms
    levels = [
        level
        for level in ('session', 'subject')
        if f'run2{level}' in transforms and f'{level}2anat' in transforms
    ]
    assert levels == []


def test_run2fmap_selects_by_fieldmap_id(tmp_path, deriv_dataset):
    names = (
        *CURRENT_RUN,
        'sub-01_ses-A_task-rest_run-01_from-run_to-auto00001_mode-image_desc-fmap_xfm.txt',
    )
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *names), params={'fmapid': 'auto00001'})

    assert len(out['transforms']['run2fmap']) == 1
    assert out['transforms']['run2fmap'][0].endswith('to-auto00001_mode-image_desc-fmap_xfm.txt')


@pytest.mark.parametrize(
    'dataset',
    [CURRENT_RUN, LEGACY_RUN],
    ids=['current-naming', 'legacy-naming'],
)
def test_run2fmap_never_returns_a_coreg_transform(tmp_path, deriv_dataset, dataset):
    """Called without a fieldmap_id, run2fmap must not pick up boldref->anat.

    Under current naming, the first alternative's own entities (desc-fmap,
    from-run) exclude the coreg transform. Under legacy naming, the first
    alternative doesn't match at all (no 'run' entity), so the second
    alternative's ``desc: null`` is what actually does the exclusion —
    without it, the unconstrained ``to`` (no fieldmap_id given) would let
    the boldref->T1w coreg transform through too.
    """
    out = _collect(_dataset(tmp_path / 'd', deriv_dataset, *dataset))

    for path in out['transforms']['run2fmap']:
        assert 'desc-coreg' not in path
        assert 'to-T1w' not in path
