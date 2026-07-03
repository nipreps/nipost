# src/nipost/bids/spec.py
"""nipost's derivative-query spec schema.

Not a port: designed to be expressive enough to cover the query cases in the
fMRIPrep, sMRIPrep, and nibabies io_spec files, without matching them verbatim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

Cardinality = str  # one of: 'single', 'optional', 'list', 'pair', 'ordered'


@dataclass
class Query:
    entities: dict
    cardinality: Cardinality = 'single'
    labels: list[str] | None = None


@dataclass
class Spec:
    items: dict[str, Query] = field(default_factory=dict)
    transforms: dict[str, Query] = field(default_factory=dict)  # flat -> out['transforms'][key]
    space_transforms: dict[str, Query] = field(
        default_factory=dict
    )  # per std_space -> out['transforms'][space][key]


def substitute_space(space: str) -> str:
    """Convert TemplateFlow cohort syntax to a BIDS filename entity value."""
    return space.replace(':cohort-', '+')


def sanitize_fieldmap_id(fieldmap_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', fieldmap_id)


def _query_from_dict(raw: dict) -> Query:
    return Query(
        entities=raw['entities'],
        cardinality=raw.get('cardinality', 'single'),
        labels=raw.get('labels'),
    )


def load_spec(name_or_path: str | Path) -> Spec:
    """Load a bundled spec by name (``anat``/``func``) or from a JSON path."""
    text: str
    if isinstance(name_or_path, str) and name_or_path in ('anat', 'func'):
        text = (files('nipost.bids.data') / f'{name_or_path}.json').read_text()
    else:
        text = Path(name_or_path).read_text()
    raw = json.loads(text)
    return Spec(
        items={k: _query_from_dict(v) for k, v in raw.get('items', {}).items()},
        transforms={k: _query_from_dict(v) for k, v in raw.get('transforms', {}).items()},
        space_transforms={
            k: _query_from_dict(v) for k, v in raw.get('space_transforms', {}).items()
        },
    )
