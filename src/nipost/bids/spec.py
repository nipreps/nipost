# src/nipost/bids/spec.py
"""nipost's derivative-query spec schema.

A spec is a mapping of author-chosen group names to groups of named queries.
``Group`` and ``Query`` forbid unknown fields, so a misspelled key -- ``scop``
for ``scope``, ``querys`` for ``queries`` -- generates an error at load time.
Group *names* are free-form, accessed as dictionary entries.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Annotated

from msgspec import Meta, Struct, yaml

__all__ = [
    'Group',
    'Ordered',
    'Query',
    'Spec',
    'load_spec',
    'sanitize_fieldmap_id',
    'sanitize_space',
]

_PLACEHOLDER_RE = re.compile(r'^\{([^}]+)\}$')
"""A ``{name}``-shaped entity value: a reference to a caller-supplied parameter.

Used here and in ``Query.__post_init__`` to reject a placeholder in an ordering axis.
"""


class Ordered(Struct, forbid_unknown_fields=True):
    """The ordering axis of a list result.

    Parameters
    ----------
    order
        Name of the entity that orders the result.
    """

    order: str


class Query(Struct, forbid_unknown_fields=True):
    """A single named lookup in a spec.

    Parameters
    ----------
    entities
        Ordered entity dicts describing the same logical item under different
        naming schemes. The interpreter selects the **first** alternative that
        matches anything; the shape is applied to that alternative's matches
        alone, never to a union across alternatives. Put current naming first
        and legacy naming after it.
    multi
        The result's shape, in one field: ``False`` (default) for a scalar,
        ``True`` for an unordered list, or an :class:`Ordered` for a list
        ordered along an entity.
    scope
        The scope of entities permitted in this query.
        This acts as a pass-list of entities provided by the caller.
        This is useful for indexing files that are generated from multiple
        source files, for example, a session-level BOLD reference.
    """

    entities: Annotated[list[dict], Meta(min_length=1)]
    multi: bool | Ordered = False
    scope: list[str] | None = None

    def __post_init__(self):
        if isinstance(self.multi, bool):
            return

        for i, alt in enumerate(self.entities):
            msg = (
                f'Query ordered by {self.multi.order!r} requires every alternative '
                f'to declare a non-empty {self.multi.order!r} value list'
            )
            if self.multi.order not in alt:
                raise ValueError(f'{msg}; entry {i} lacks a value.')
            value = alt[self.multi.order]
            if not isinstance(value, list):
                raise TypeError(f'{msg}; entry {i} declares {value!r}.')
            if not value:
                raise ValueError(f'{msg}; entry {i} declares an empty list.')
            for member in value:
                if isinstance(member, str) and _PLACEHOLDER_RE.match(member):
                    raise ValueError(
                        f'Query ordered by {self.multi.order!r} cannot use the '
                        f'placeholder {member!r} in its value list.'
                    )

    @property
    def is_multi(self) -> bool:
        """Whether the result is a list."""
        return bool(self.multi)

    @property
    def order(self) -> str | None:
        """The ordering entity, or ``None`` for an unordered result."""
        return self.multi.order if isinstance(self.multi, Ordered) else None


class Group(Struct, forbid_unknown_fields=True):
    """A group of named queries, optionally indexed by a parameter.

    Parameters
    ----------
    queries
        The group's named queries. Non-empty: a group with no queries has no
        purpose, and an empty one is more likely a typo than an intention.
    per
        Name of a parameter this group iterates over. Each query runs once per
        value in ``params[per]``, with that value bound as a placeholder, and
        results nest under the value. ``None`` runs each query once.
    """

    queries: Annotated[dict[str, Query], Meta(min_length=1)]
    per: str | None = None


type Spec = dict[str, Group]
"""A spec: author-named groups. The YAML top level is this mapping."""


def sanitize_space(space: str) -> str:
    """Convert TemplateFlow cohort syntax to a BIDS filename entity value.

    ``collect_derivatives`` uses parameter values verbatim, so a caller holding
    TemplateFlow strings applies this first.

    Example
    -------

    >>> sanitize_space('MNI152NLin2009cAsym:cohort-1')
    'MNI152NLin2009cAsym+1'
    """
    return space.replace(':cohort-', '+')


def sanitize_fieldmap_id(fieldmap_id: str) -> str:
    """Convert a fieldmap ID to a valid BIDS entity value.

    Fieldmap IDs are derived from ``B0FieldIdentifier`` values, which are free-form strings.
    BIDS entity values must be alphanumeric or a plus sign (`+`).
    This function follows the SDCFlows/fMRIPrep convention of stripping all non-alphanumeric
    characters from the fieldmap ID.

    Example
    -------

    >>> sanitize_fieldmap_id('fieldmap-1')
    'fieldmap1'
    >>> sanitize_fieldmap_id('topup+PA')
    'topupPA'
    """
    return re.sub(r'[^a-zA-Z0-9]', '', fieldmap_id)


def load_spec(name_or_path: str | Path) -> Spec:
    """Load a bundled spec by name (``anat``/``func``/``fmap``) or from a path.

    Bundled specs are YAML. A path is decoded as YAML too, so a JSON spec file
    loads as well.
    """
    text: str
    if isinstance(name_or_path, str) and name_or_path in ('anat', 'func', 'fmap'):
        text = (files('nipost.bids.data') / f'{name_or_path}.yml').read_text()
    else:
        text = Path(name_or_path).read_text()
    return yaml.decode(text, type=Spec)
