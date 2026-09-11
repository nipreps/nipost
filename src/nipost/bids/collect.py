# src/nipost/bids/collect.py
"""Generic, spec-driven derivative discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from nipost.bids._layout import get_layout
from nipost.bids.spec import _PLACEHOLDER_RE, Group, Query, Spec, load_spec

_DROP = object()  # sentinel: this value contributes no constraint


def _resolve(alt: dict, base: dict, scope: list[str] | None, params: dict) -> dict:
    """Build a PyBIDS filter dict from an entities spec and caller-supplied entities.

    Spec-declared entities override caller-supplied ones.
    ``scope`` restricts the caller-supplied entities to the ones this query
    accepts; ``None`` accepts all of them and ``[]`` accepts none.
    Values of the form ``{name}`` are replaced with ``params[name]``.
    A placeholder that does not appear in ``params`` is dropped.
    """
    merged = {
        # Filter supplied entities by scope, if provided
        **{key: val for key, val in base.items() if scope is None or key in scope},
        # Override with spec-declared entities, which may contain placeholders
        **alt,
    }
    out: dict = {}
    for key, value in merged.items():
        values = value if isinstance(value, list) else [value]
        resolved = []
        for declared in values:
            match = _PLACEHOLDER_RE.match(declared) if isinstance(declared, str) else None
            name = match.group(1) if match else declared
            item = params.get(name, _DROP) if match else declared
            if item is _DROP:
                continue
            if isinstance(item, (list, tuple)):
                raise TypeError(
                    f'params[{name!r}] must be a scalar for placeholder {declared!r} '
                    f'in entity {key!r}'
                )
            resolved.append(item)
        if resolved:
            out[key] = resolved
    return out


def _lookup(layout, key: str, query: Query, base: dict, params: dict) -> str | list | None:
    """Return the reduced result for the first alternative that matches anything.

    If no alternative matches, the zero-match outcome for the declared shape is
    returned: ``[]`` for an unordered list, ``None`` otherwise. That rule has
    one owner, :func:`_reduce_ordered`'s "declared value with no match" branch
    -- reached here by forwarding ``order``/``order_values`` on the no-match
    path too, taken from the first alternative since ``Query.__post_init__``
    guarantees every alternative declares the ordering entity as a non-empty
    list.
    """
    found = []
    entities = {}
    for entities in query.entities:
        found = layout.get(**_resolve(entities, base, query.scope, params))
        if found:
            break

    return _reduce(
        key,
        found,
        multi=query.is_multi,
        order=query.order,
        order_values=entities.get(query.order) if query.order else None,
    )


def _run_queries(layout, group: Group, base: dict, params: dict) -> dict:
    """Run every query in a group once, dropping the keys that found nothing."""
    return {
        key: result
        for key, query in group.queries.items()
        if (result := _lookup(layout, key, query, base, params)) is not None
    }


def _natural_key(path: str) -> list[int | str]:
    """Sort key ordering embedded integers numerically, so coeff2 precedes coeff10.

    ``re.split`` with a capture group always yields non-digit segments at
    even indices and digit runs at odd ones, so within a single key an
    ``int`` is never compared against a ``str`` at the same position.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]


def _reduce(
    key: str,
    files: list,
    *,
    multi: bool,
    order: str | None = None,
    order_values: list[str | None] | None = None,
) -> str | list[str] | None:
    """Reduce matched files to the shape a query declared.

    Ambiguous results (multiple matches for single results) produce ValueErrors.
    Missing results produce None (single result) or empty lists (multiple results).

    A ``multi`` result without ``order`` is natural-sorted by path, because
    callers such as :func:`nipost.reconstruct_fieldmap` depend on position --
    it reads ``coefficients[-1]`` as the finest B-spline level.

    Within an ordered result, each matching value must match a unique file.
    """
    if order is None:
        paths = [f.path for f in files]
        if multi:
            return sorted(paths, key=_natural_key)
        if len(paths) > 1:
            raise ValueError(f'{key!r}: expected at most one match, got {len(paths)}: {paths}')
        return paths[0] if paths else None

    # Ordered result
    by_value: dict[str | None, list[str]] = {}
    for f in files:
        by_value.setdefault(f.entities.get(order), []).append(f.path)

    for value, group in by_value.items():
        if len(group) > 1:
            raise ValueError(
                f'{key!r}: expected at most one match for {order}={value!r}, '
                f'got {len(group)}: {group}'
            )

    order_values = _coerce_axis(order_values or [], by_value)

    if not order_values or any(value not in by_value for value in order_values):
        return None

    return [by_value[value][0] for value in order_values]


def _coerce_axis(order_values: list[str | None], parsed: Iterable[str | None]) -> list[str | None]:
    """Coerce declared axis values to the type PyBIDS parsed the entity as.

    PyBIDS will coerce '01' to an int if the entity is int-typed,
    so the axis must do the same in order to match values
    (``'01' != PaddedInt('01')``).

    Coerce rather than error, because the values have already been accepted
    by PyBIDS.
    """

    if (sample := next((val for val in parsed if val is not None), None)) is None:
        return order_values
    cls = type(sample)
    coerced: list[str | None] = []
    for value in order_values:
        if value is None or isinstance(value, cls):
            coerced.append(value)
            continue
        try:
            coerced.append(cls(value))
        except (TypeError, ValueError):
            coerced.append(value)
    return coerced


def collect_derivatives(
    derivatives_dir: Path,
    *,
    spec: Spec,
    entities: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Collect precomputed derivatives described by ``spec``.

    Parameters
    ----------
    derivatives_dir
        Root of the derivatives dataset to search.
    spec
        Author-named groups of queries, from :func:`nipost.bids.spec.load_spec`.
    entities
        BIDS entities to match, e.g. ``{'subject': '01', 'task': 'rest'}``.
        A query's ``scope`` may narrow which of these it accepts.
    params
        Values for the placeholders the spec references. A group declaring
        ``per: <name>`` iterates ``params[<name>]``; any other placeholder
        takes ``params[<name>]`` as a scalar and drops its constraint when
        unbound. Values are used verbatim, so apply
        :func:`nipost.bids.spec.sanitize_space` or
        :func:`nipost.bids.spec.sanitize_fieldmap_id` first if needed.

    Returns
    -------
    dict
        One key per group declared by ``spec``, always present even when the
        group collected nothing. A plain group maps to ``{query: result}``; a
        group with ``per`` maps to ``{param value: {query: result}}``.
    """
    layout = get_layout(Path(derivatives_dir))
    base: dict = dict(entities or {})
    supplied: dict = dict(params or {})

    out: dict = {}
    for group_name, group in spec.items():
        if group.per is None:
            out[group_name] = _run_queries(layout, group, base, supplied)
            continue

        values = supplied.get(group.per) or []
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise TypeError(
                f'Group {group_name!r} iterates over {group.per!r}, so '
                f'params[{group.per!r}] must be a sequence of values, not '
                f'{values!r}'
            )
        out[group_name] = {
            value: _run_queries(layout, group, base, {**supplied, group.per: value})
            for value in values
        }
    return out


def collect_fieldmaps(
    derivatives_dir: Path,
    *,
    entities: dict,
    spec: Spec | None = None,
) -> dict[str, dict[str, dict[str, str | list[str]]]]:
    """Collect fieldmap derivatives, grouped by fieldmap id.

    Parameters
    ----------
    derivatives_dir
        Root of the derivatives dataset to search.
    entities
        BIDS entities to match, e.g. ``{'subject': '01'}``. Also scopes the
        fieldmap-id enumeration, so only ids reachable under these entities
        are collected.
    spec
        Groups of queries to run against each id. Defaults to the bundled
        ``fmap`` spec.

    Returns
    -------
    dict
        The grouped output of :func:`collect_derivatives`. With the bundled
        spec that is ``{'fieldmaps': {fmapid: {'fieldmap': ..., 'coeffs': [...],
        'magnitude': ...}}}``, where ``coeffs`` has one entry per B-spline
        level, which is what :func:`nipost.reconstruct_fieldmap` expects.
    """
    if spec is None:
        spec = load_spec('fmap')

    layout = get_layout(Path(derivatives_dir))

    fmapids: list[str] = layout.get_fmapids(**entities)

    return collect_derivatives(
        derivatives_dir, spec=spec, entities=entities, params={'fmapid': fmapids}
    )
