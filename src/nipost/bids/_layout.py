"""Cached BIDSLayout factory for derivative discovery."""

from functools import cache
from pathlib import Path

from bids.layout import BIDSLayout


@cache
def get_layout(derivatives_dir: Path) -> BIDSLayout:
    import niworkflows.data

    return BIDSLayout(
        derivatives_dir,
        config=[niworkflows.data.load('nipreps.json')],
        validate=False,
    )
