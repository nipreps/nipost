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
from typing import Literal, get_args

Cardinality = Literal['single', 'list', 'pair', 'ordered']
_CARDINALITIES = get_args(Cardinality)


@dataclass
class Query:
    """A single named lookup in a spec.

    Parameters
    ----------
    alternatives
        Ordered entity dicts describing the same logical item under different
        naming schemes. The interpreter selects the **first** alternative that
        matches anything; cardinality is applied to that alternative's matches
        alone, never to a union across alternatives. Put current naming first
        and legacy naming after it.
    cardinality
        The shape to reduce matches to. See :func:`nipost.bids.collect._cardinality`.
    labels
        For ``cardinality='ordered'``, the ``label`` entity values in the
        required output order.
    scope
        Allowlist of caller-supplied entity names this query accepts;
        everything else the caller passed is dropped for this query. ``None``
        accepts all of them. Group-level derivatives — written once per session
        or subject, with run-level entities dismissed — use this to become
        reachable from a run-level call.
    """

    alternatives: list[dict]
    cardinality: Cardinality = 'single'
    labels: list[str] | None = None
    scope: list[str] | None = None


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
    """Build a :class:`Query` from its JSON form.

    Accepts either ``{"entities": {...}}`` (sugar for a single alternative) or
    ``{"alternatives": [{...}, ...]}``, but not both and not neither. An empty
    ``alternatives`` list is rejected too: it would otherwise silently mean
    "this key is never present" (``_lookup`` loops zero times over it).

    ``cardinality`` is validated against the permitted set here, at load
    time, rather than left to surface as a ``ValueError`` from deep inside a
    layout query the first time that key is actually reached.
    """
    entities = raw.get('entities')
    alternatives = raw.get('alternatives')
    if (entities is None) == (alternatives is None):
        raise ValueError("A query needs exactly one of 'entities' or 'alternatives'")
    if alternatives is not None and not alternatives:
        raise ValueError("A query's 'alternatives' list must not be empty")
    cardinality = raw.get('cardinality', 'single')
    if cardinality not in _CARDINALITIES:
        raise ValueError(f'Unknown cardinality {cardinality!r}; must be one of {_CARDINALITIES}')
    if entities is not None:
        alts = [entities]
    else:
        assert alternatives is not None
        alts = alternatives
    return Query(
        alternatives=alts,
        cardinality=cardinality,
        labels=raw.get('labels'),
        scope=raw.get('scope'),
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
