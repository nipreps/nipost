# src/nipost/bids/spec.py
"""nipost's derivative-query spec schema.

Not a port: designed to be expressive enough to cover the query cases in the
fMRIPrep, sMRIPrep, and nibabies io_spec files, without matching them verbatim.
"""

from __future__ import annotations

import enum
import re
from importlib.resources import files
from pathlib import Path

from msgspec import Struct, field, yaml


class Cardinality(enum.StrEnum):
    SINGLE = 'single'
    LIST = 'list'
    PAIR = 'pair'
    ORDERED = 'ordered'


class Query(Struct):
    """A single named lookup in a spec.

    Parameters
    ----------
    entities
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

    entities: list[dict]
    cardinality: Cardinality
    labels: list[str] | None = None
    scope: list[str] | None = None

    def __post_init__(self):
        if not self.entities:
            raise ValueError("Query must have at least one alternative in 'entities'")
        if self.cardinality == 'ordered' and not self.labels:
            raise ValueError("Query with cardinality='ordered' must have non-empty labels")


class Spec(Struct):
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


def load_spec(name_or_path: str | Path) -> Spec:
    """Load a bundled spec by name (``anat``/``func``) or from a JSON path."""
    text: str
    if isinstance(name_or_path, str) and name_or_path in ('anat', 'func', 'fmap'):
        text = (files('nipost.bids.data') / f'{name_or_path}.yml').read_text()
    else:
        text = Path(name_or_path).read_text()
    return yaml.decode(text, type=Spec)
