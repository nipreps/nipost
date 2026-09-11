# tests/bids/conftest.py
"""Shared fixtures for the nipost.bids tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def deriv_dataset():
    """Return a factory that makes a valid, empty derivative dataset root.

    PyBIDS refuses to index a directory without a well-formed
    ``dataset_description.json``, so every test that builds a dataset on disk
    needs one, and eleven copies of the same four keys had accumulated. The
    factory takes the root rather than deriving it from ``tmp_path``, because
    several tests build two datasets in one ``tmp_path`` and one writes into
    ``tmp_path`` itself.

    ``generated_by`` is a parameter because the ``func`` spec's tests assert
    against fMRIPrep-shaped derivatives, and the pipeline name is part of what
    makes the fixture realistic there.
    """

    def _make(root: Path, *, generated_by: str = 'nipost') -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / 'dataset_description.json').write_text(
            json.dumps(
                {
                    'Name': 'x',
                    'BIDSVersion': '1.8.0',
                    'DatasetType': 'derivative',
                    'GeneratedBy': [{'Name': generated_by}],
                }
            )
        )
        return root

    return _make
