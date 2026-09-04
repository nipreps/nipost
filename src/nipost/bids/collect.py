# src/nipost/bids/collect.py
"""Generic, spec-driven derivative discovery."""

from __future__ import annotations

import json
from importlib.resources import files as _pkg_files
from pathlib import Path

from bids.layout import Query

from nipost.bids._layout import get_layout
from nipost.bids.spec import Query as SpecQuery
from nipost.bids.spec import Spec, _query_from_dict, sanitize_fieldmap_id, substitute_space


def _resolve(
    alt: dict,
    base: dict,
    scope: list[str] | None,
    fieldmap_id: str | None,
    space: str | None = None,
) -> dict:
    """Build a PyBIDS filter dict from one alternative.

    Spec-declared entities override caller-supplied ones.

    - ``None`` (scalar, or a member of a value list) becomes PyBIDS
      ``Query.NONE``: the entity must be ABSENT. To leave an entity
      unconstrained, omit it.
    - ``'{fieldmap_id}'`` is sanitized and substituted; when no ``fieldmap_id``
      was given the constraint is dropped so the query matches any value.
    - ``'{space}'`` is substituted with the standard space being iterated
      (after cohort conversion). It is only meaningful for
      ``space_transforms`` queries.
    """
    merged = {**_scoped(base, scope), **alt}
    out: dict = {}
    for key, value in merged.items():
        if value is None:
            out[key] = Query.NONE
        elif value == '{fieldmap_id}':
            if fieldmap_id is not None:
                out[key] = sanitize_fieldmap_id(fieldmap_id)
        elif value == '{space}':
            if space is None:
                raise ValueError(
                    "The '{space}' placeholder is only valid in space_transforms queries"
                )
            out[key] = substitute_space(space)
        elif isinstance(value, list):
            out[key] = [Query.NONE if item is None else item for item in value]
        else:
            out[key] = value
    return out


def _scoped(base: dict, scope: list[str] | None) -> dict:
    """Restrict caller-supplied entities to the ones this query accepts."""
    if scope is None:
        return base
    return {key: value for key, value in base.items() if key in scope}


def _lookup(
    layout,
    query: SpecQuery,
    base: dict,
    fieldmap_id: str | None,
    space: str | None = None,
) -> str | list | None:
    """Return the reduced result for the first alternative that matches anything.

    If no alternative matches, the cardinality's zero-match outcome is returned
    (``[]`` for ``'list'``, ``None`` otherwise).
    """
    for alt in query.alternatives:
        found = layout.get(**_resolve(alt, base, query.scope, fieldmap_id, space))
        if found:
            return _cardinality(query, found)
    return _cardinality(query, [])


def _cardinality(query: SpecQuery, files: list) -> str | list | None:
    """Reduce a list of BIDSFile objects to the shape declared by the query."""
    paths = [f.path for f in files]
    card = query.cardinality
    if card == 'single':
        return paths[0] if len(paths) == 1 else (paths or None)
    if card == 'list':
        return paths
    if card == 'optional':
        return paths or None
    if card == 'pair':
        return sorted(paths) if len(paths) == 2 else None
    if card == 'ordered':
        by_label = {f.entities.get('label'): f.path for f in files}
        ordered = [by_label[label] for label in (query.labels or []) if label in by_label]
        return ordered or None
    raise ValueError(f'Unknown cardinality: {card}')


def collect_derivatives(
    derivatives_dir: Path,
    *,
    spec: Spec,
    subject_id: str | None = None,
    entities: dict | None = None,
    std_spaces: list[str] | None = None,
    fieldmap_id: str | None = None,
) -> dict:
    """Collect precomputed derivatives described by ``spec``.

    Returns flat ``item`` results plus a ``transforms`` entry that is either a
    flat ``{name: path}`` dict (from ``spec.transforms``) or a space-nested
    ``{space: {name: path}}`` dict (from ``spec.space_transforms``).
    """
    layout = get_layout(Path(derivatives_dir))
    base: dict = dict(entities or {})
    if subject_id is not None:
        base['subject'] = subject_id

    out: dict = {}
    for key, query in spec.items.items():
        result = _lookup(layout, query, base, fieldmap_id)
        if result is not None:
            out[key] = result

    transforms: dict = {}
    # Flat transforms (func: hmc / boldref2anat / boldref2fmap)
    for key, query in spec.transforms.items():
        result = _lookup(layout, query, base, fieldmap_id)
        if result is not None:
            transforms[key] = result
    # Space-varying transforms (anat: forward / reverse per std space)
    for space in std_spaces or []:
        for key, query in spec.space_transforms.items():
            result = _lookup(layout, query, base, fieldmap_id, space)
            if result is not None:
                transforms.setdefault(space, {})[key] = result

    out['transforms'] = transforms
    return out


def collect_fieldmaps(
    derivatives_dir: Path,
    entities: dict,
    spec: dict[str, SpecQuery] | None = None,
) -> dict:
    """Collect fieldmap derivatives grouped by fieldmap id.

    Returns a dict keyed by fmapid (e.g. ``auto00000``), each value being a
    dict with keys ``fieldmap``, ``coeffs``, and ``magnitude`` (scalars when
    cardinality is ``single``).
    """
    if spec is None:
        raw = json.loads((_pkg_files('nipost.bids.data') / 'fmap.json').read_text())
        spec = {k: _query_from_dict(v) for k, v in raw.items()}

    layout = get_layout(Path(derivatives_dir))

    # Enumerate fieldmap ids: prefer the NiPreps pybids extension; fall back to
    # querying unique entity values if the method is unavailable.
    if hasattr(layout, 'get_fmapids'):
        fmapids: list[str] = layout.get_fmapids(**entities)
    else:
        fmapids = layout.get(target='fmapid', return_type='id', **entities) or []

    out: dict = {}
    for fmapid in fmapids:
        entry: dict = {}
        for key, query in spec.items():
            result = _lookup(layout, query, {**entities, 'fmapid': fmapid}, None)
            if result is not None:
                entry[key] = result
        if entry:
            out[fmapid] = entry

    return out
