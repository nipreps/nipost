# src/nipost/bids/collect.py
"""Generic, spec-driven derivative discovery."""

from __future__ import annotations

from pathlib import Path

from bids.layout import Query

from nipost.bids._layout import get_layout
from nipost.bids.spec import Query as SpecQuery
from nipost.bids.spec import Spec, sanitize_fieldmap_id, substitute_space


def _clean(entities: dict, fieldmap_id: str | None) -> dict:
    """Resolve dynamic values and drop unconstrained (None) entities.

    - ``None`` value: dropped ("no constraint"; also the std-space placeholder
      after substitution).
    - ``'*none*'`` value: mapped to PyBIDS ``Query.NONE`` (entity must be
      ABSENT) — used to exclude e.g. the ``desc-coreg`` transform from the
      ``boldref2fmap`` query so it is not confused with ``boldref2anat``.
    - ``'{fieldmap_id}'`` value: sanitized/substituted; if ``fieldmap_id`` is
      None the constraint is dropped so the query matches any value.
    """
    out: dict = {}
    for key, value in entities.items():
        if value is None:
            continue
        if value == '*none*':
            out[key] = Query.NONE
            continue
        if value == '{fieldmap_id}':
            if fieldmap_id is not None:
                out[key] = sanitize_fieldmap_id(fieldmap_id)
            continue
        out[key] = value
    return out


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
        qry = _clean({**query.entities, **base}, fieldmap_id)
        result = _cardinality(query, layout.get(**qry))
        if result is not None:
            out[key] = result

    transforms: dict = {}
    # Flat transforms (func: hmc / boldref2anat / boldref2fmap)
    for key, query in spec.transforms.items():
        qry = _clean({**query.entities, **base}, fieldmap_id)
        result = _cardinality(query, layout.get(**qry))
        if result is not None:
            transforms[key] = result
    # Space-varying transforms (anat: forward / reverse per std space)
    for space in std_spaces or []:
        space_entity = substitute_space(space)
        for key, query in spec.space_transforms.items():
            raw = {k: (space_entity if v is None else v) for k, v in query.entities.items()}
            qry = _clean({**raw, **base}, fieldmap_id)
            result = _cardinality(query, layout.get(**qry))
            if result is not None:
                transforms.setdefault(space, {})[key] = result

    out['transforms'] = transforms
    return out


def collect_fieldmaps(derivatives_dir: Path, entities: dict, spec: Spec | None = None) -> dict:
    raise NotImplementedError  # implemented in Task 12
