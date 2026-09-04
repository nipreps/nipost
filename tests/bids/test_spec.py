# tests/bids/test_spec.py
"""Schema dataclasses and the JSON loader."""

import pytest

pytest.importorskip('bids')


def test_query_from_dict_accepts_entities_sugar():
    from nipost.bids.spec import _query_from_dict

    query = _query_from_dict({'entities': {'suffix': 'T1w'}, 'cardinality': 'single'})

    assert query.alternatives == [{'suffix': 'T1w'}]
    assert query.cardinality == 'single'
    assert query.scope is None


def test_query_from_dict_accepts_alternatives_and_scope():
    from nipost.bids.spec import _query_from_dict

    query = _query_from_dict(
        {
            'alternatives': [{'space': 'run'}, {'desc': 'coreg'}],
            'scope': ['subject', 'session'],
        }
    )

    assert query.alternatives == [{'space': 'run'}, {'desc': 'coreg'}]
    assert query.scope == ['subject', 'session']


@pytest.mark.parametrize(
    'raw',
    [
        {'entities': {'suffix': 'T1w'}, 'alternatives': [{'suffix': 'T1w'}]},
        {'cardinality': 'single'},
    ],
    ids=['both-forms', 'neither-form'],
)
def test_query_from_dict_requires_exactly_one_form(raw):
    from nipost.bids.spec import _query_from_dict

    with pytest.raises(ValueError, match='exactly one'):
        _query_from_dict(raw)


def test_bundled_specs_load():
    from nipost.bids.spec import load_spec

    for name in ('anat', 'func'):
        spec = load_spec(name)
        for query in {**spec.items, **spec.transforms, **spec.space_transforms}.values():
            assert isinstance(query.alternatives, list)
            assert query.alternatives
            assert all(isinstance(alt, dict) for alt in query.alternatives)


def test_query_from_dict_rejects_unknown_cardinality():
    """A typo in a custom spec must surface at load time, not deep in a query."""
    from nipost.bids.spec import _query_from_dict

    with pytest.raises(ValueError, match='optional'):
        _query_from_dict({'entities': {'suffix': 'T1w'}, 'cardinality': 'optional'})


def test_query_from_dict_rejects_empty_alternatives():
    from nipost.bids.spec import _query_from_dict

    with pytest.raises(ValueError, match='alternatives'):
        _query_from_dict({'alternatives': []})
