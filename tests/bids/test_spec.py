# tests/bids/test_spec.py
"""Schema structs and the YAML loader."""

import pytest

pytest.importorskip('bids')

import msgspec

from nipost.bids.spec import Query, Spec, load_spec


def test_query_decodes_entities_and_scope():
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
    for name in ('anat', 'func', 'fmap'):
        spec = load_spec(name)
        for query in {**spec.items, **spec.transforms, **spec.space_transforms}.values():
            assert isinstance(query.entities, list)
            assert query.entities
            assert all(isinstance(alt, dict) for alt in query.entities)


def test_query_rejects_unknown_cardinality():
    """A typo in a custom spec must surface at load time, not deep in a query."""
    with pytest.raises(msgspec.ValidationError, match='optional'):
        msgspec.convert({'entities': [{'suffix': 'T1w'}], 'cardinality': 'optional'}, type=Query)


def test_query_rejects_empty_entities():
    with pytest.raises(msgspec.ValidationError, match='entities'):
        msgspec.convert({'entities': [], 'cardinality': 'single'}, type=Query)


def test_query_rejects_unknown_fields():
    """A misspelled field must be rejected, not decoded to its default.

    ``scop`` for ``scope`` would otherwise produce a query with no scope at
    all, which still resolves against run-level datasets -- so the loss would
    surface only as an empty result on the group-level datasets ``scope``
    exists to reach.
    """
    with pytest.raises(msgspec.ValidationError, match='scop'):
        msgspec.convert(
            {'entities': [{'suffix': 'T1w'}], 'cardinality': 'single', 'scop': ['subject']},
            type=Query,
        )


def test_spec_rejects_unknown_sections():
    """A misspelled section must be rejected, not decoded to an empty spec."""
    with pytest.raises(msgspec.ValidationError, match='item'):
        msgspec.convert(
            {'item': {'x': {'entities': [{'suffix': 'T1w'}], 'cardinality': 'single'}}},
            type=Spec,
        )
