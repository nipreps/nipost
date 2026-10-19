# src/nipost/bids/collect.py
"""Generic, spec-driven derivative discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from bids.layout import Query as BIDSQuery

from nipost.bids._layout import get_layout
from nipost.bids.spec import _PLACEHOLDER_RE, Group, Query, Spec, load_spec


def _resolve(alt: dict, base: dict, scope: list[str] | None, params: dict) -> dict:
    """Build a PyBIDS filter dict from an entities spec and caller-supplied entities.

    Spec-declared entities override caller-supplied ones.
    ``scope`` restricts the caller-supplied entities to the ones this query
    accepts; ``None`` accepts all of them and ``[]`` accepts none.
    Values of the form ``{name}`` are replaced with ``params[name]`` verbatim,
    so ``None`` requires the entity to be absent.
    """
    merged = {
        # Filter supplied entities by scope, if provided
        **{key: val for key, val in base.items() if scope is None or key in scope},
        # Override with spec-declared entities, which may contain placeholders
        **alt,
    }
    out: dict = {}
    for key, value in merged.items():
        resolved = []
        for declared in value if isinstance(value, list) else [value]:
            match = _PLACEHOLDER_RE.match(declared) if isinstance(declared, str) else None
            if match is None:
                resolved.append(declared)
                continue
            resolved.append(params[match.group(1)])
        out[key] = resolved
    return out


def _lookup(
    layout, location: str, query: Query, base: dict, params: dict
) -> str | list[str] | None:
    """Return the reduced result for the first alternative that matches anything.

    If a single file is expected, the result is a string or ``None``.
    If an unordered list of files is expected, the result is a non-empty list or ``None``.
    If an ordered list of files is expected, the result is a complete list or ``None``.

    Ambiguous results (multiple matches for either a single file
    or a single value in an ordered list) produce ValueErrors.

    Ordered lists are sorted by matching entity values.
    Unordered lists are natural-sorted by path to ensure deterministic results,
    and to support use-cases such as :func:`nipost.reconstruct_fieldmap` that depend
    on position (``coefficients[-1]`` is the finest B-spline level).
    """
    found = []
    entities: dict = {}
    filters: dict = {}
    index = 0
    for index, entities in enumerate(query.entities):
        filters = _resolve(entities, base, query.scope, params)
        found = layout.get(**filters)
        if found:
            break
    where = f'{location} in {layout.root} (alternative {index})'

    # Unordered, single or multi
    if query.order is None:
        if not (paths := [f.path for f in found]):
            return None
        if query.is_multi:
            return sorted(paths, key=_natural_key)
        if len(paths) > 1:
            raise ValueError(
                f'{where}: expected at most one match for {filters}, got {len(paths)}: {paths}'
            )
        return paths[0]

    # Ordered result
    by_value: dict[str | None, list[str]] = {}
    for f in found:
        by_value.setdefault(f.entities.get(query.order), []).append(f.path)

    for value, group in by_value.items():
        if len(group) > 1:
            raise ValueError(
                f'{where}: expected at most one match for {filters} with '
                f'{query.order}={value!r}, got {len(group)}: {group}'
            )

    order_values = _coerce_axis(entities.get(query.order, []), by_value)

    if not order_values or any(value not in by_value for value in order_values):
        return None

    return [by_value[value][0] for value in order_values]


def _natural_key(path: str) -> list[int | str]:
    """Sort key ordering embedded integers numerically, so coeff2 precedes coeff10.

    ``re.split`` with a capture group always yields non-digit segments at
    even indices and digit runs at odd ones, so within a single key an
    ``int`` is never compared against a ``str`` at the same position.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]


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
        ``per: <name>`` iterates ``params[<name>]``: ``None`` for no iterations,
        a single value, or a sequence of values. Any other placeholder takes
        ``params[<name>]`` as a scalar, used verbatim: ``None`` requires the
        entity to be absent, and a ``bids.layout.Query`` member has its
        PyBIDS meaning. Values are used verbatim, so apply
        :func:`nipost.bids.spec.sanitize_space` or
        :func:`nipost.bids.spec.sanitize_fieldmap_id` first if needed.

    Returns
    -------
    dict
        One key per group declared by ``spec``, always present even when the
        group collected nothing. A plain group maps to ``{query: result}``; a
        group with ``per`` maps to ``{param value: {query: result}}``.

    Raises
    ------
    TypeError
        If a group's ``per`` parameter is missing or contains a
        ``bids.layout.Query`` member, a placeholder has no value in
        ``params``, or a list or tuple is bound to a scalar placeholder.
    ValueError
        If a query matches more than one file in one alternative, or two
        files share an ordering value.
    """
    base: dict = dict(entities or {})
    supplied: dict = dict(params or {})

    # Check every group before querying, so a malformed call fails on any dataset
    iterations: dict[str, list | None] = {}
    for group_name, group in spec.items():
        if group.per is not None:
            if group.per not in supplied:
                raise TypeError(
                    f'Group {group_name!r} iterates over {group.per!r}; '
                    f'pass params[{group.per!r}], or None for no iterations'
                )
            values = supplied[group.per]
            if values is None:
                values = []
            elif isinstance(values, str) or not isinstance(values, Iterable):
                values = [values]
            else:
                values = list(values)
            if any(isinstance(value, (list, tuple)) for value in values):
                raise TypeError(
                    f'Group {group_name!r}: params[{group.per!r}] cannot bind a list '
                    f'or tuple to a scalar placeholder'
                )
            if any(isinstance(value, BIDSQuery) for value in values):
                raise TypeError(
                    f'Group {group_name!r}: params[{group.per!r}] cannot contain '
                    f'bids.layout.Query members'
                )
            iterations[group_name] = values
        else:
            iterations[group_name] = None
        for query_name, query in group.queries.items():
            for i, names in enumerate(query.placeholders):
                if missing := sorted(names - supplied.keys()):
                    raise TypeError(
                        f'{group_name}/{query_name} (alternative {i}) has no value for '
                        f'params {missing}'
                    )
                for name in names:
                    if name != group.per and isinstance(supplied.get(name), (list, tuple)):
                        raise TypeError(
                            f'{group_name}/{query_name} (alternative {i}): '
                            f'params[{name!r}] must be a scalar, not a list or tuple'
                        )

    layout = get_layout(Path(derivatives_dir))

    def _run_queries(prefix: str, group: Group, params: dict) -> dict:
        """Run every query in a group once, dropping the keys that found nothing."""
        return {
            key: result
            for key, query in group.queries.items()
            if (result := _lookup(layout, f'{prefix}/{key}', query, base, params)) is not None
        }

    out: dict = {}
    for group_name, group in spec.items():
        values = iterations[group_name]
        if values is None:
            out[group_name] = _run_queries(group_name, group, supplied)
            continue
        out[group_name] = {
            value: _run_queries(f'{group_name}[{value}]', group, {**supplied, group.per: value})
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
        level and is present only when coefficient files are found, which is
        what :func:`nipost.reconstruct_fieldmap` expects.
    """
    if spec is None:
        spec = load_spec('fmap')

    layout = get_layout(Path(derivatives_dir))

    fmapids: list[str] = layout.get_fmapids(**entities)

    return collect_derivatives(
        derivatives_dir, spec=spec, entities=entities, params={'fmapid': fmapids}
    )
