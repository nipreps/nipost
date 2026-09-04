# tests/bids/test_spec.py
"""Schema dataclasses and the JSON loader."""

import pytest

pytest.importorskip('bids')

import msgspec

from nipost.bids.spec import Query, load_spec


def test_query_from_dict_accepts_entities_and_scope():
    query = msgspec.convert(
        {
            'entities': [{'space': 'run'}, {'desc': 'coreg'}],
            'scope': ['subject', 'session'],
            'cardinality': 'single',
        },
        type=Query,
    )

    assert query.entities == [{'space': 'run'}, {'desc': 'coreg'}]
    assert query.scope == ['subject', 'session']
    assert query.cardinality == 'single'


def test_bundled_specs_load():
    for name in ('anat', 'func'):
        spec = load_spec(name)
        for query in {**spec.items, **spec.transforms, **spec.space_transforms}.values():
            assert isinstance(query.entities, list)
            assert query.entities
            assert all(isinstance(alt, dict) for alt in query.entities)


def test_query_from_dict_rejects_unknown_cardinality():
    """A typo in a custom spec must surface at load time, not deep in a query."""
    with pytest.raises(ValueError, match='optional'):
        msgspec.convert({'entities': [{'suffix': 'T1w'}], 'cardinality': 'optional'}, type=Query)


def test_query_from_dict_rejects_empty_entities():
    with pytest.raises(ValueError, match='entities'):
        msgspec.convert({'entities': [], 'cardinality': 'single'}, type=Query)
