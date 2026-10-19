# tests/bids/test_spec.py
"""Schema structs and the YAML loader."""

import pytest

pytest.importorskip('bids')

import msgspec

from nipost.bids.spec import (
    Group,
    Ordered,
    Query,
    load_spec,
)

# Query struct


def _query(**overrides):
    return {'entities': [{'suffix': 'T1w'}], **overrides}


def test_query_decodes_entities_and_scope():
    query = msgspec.convert(
        {
            'entities': [{'space': 'run'}, {'desc': 'coreg'}],
            'scope': ['subject', 'session'],
        },
        type=Query,
    )

    assert query.entities == [{'space': 'run'}, {'desc': 'coreg'}]
    assert query.scope == ['subject', 'session']
    assert query.multi is False


def test_query_shapes():
    # Default: Scalar
    query = msgspec.convert(_query(), type=Query)
    assert query.is_multi is False
    assert query.order is None

    # Explicit scalar
    query = msgspec.convert(_query(multi=False), type=Query)
    assert query.is_multi is False
    assert query.order is None

    # Unordered list
    query = msgspec.convert(_query(multi=True), type=Query)
    assert query.is_multi is True
    assert query.order is None

    # Ordered list
    query = msgspec.convert(
        {
            'entities': [{'suffix': 'probseg', 'label': ['GM', 'WM']}],
            'multi': {'order': 'label'},
        },
        type=Query,
    )
    assert query.is_multi is True
    assert query.order == 'label'

    # The field is optional, not nullable.
    with pytest.raises(msgspec.ValidationError, match='multi'):
        msgspec.convert(_query(multi=None), type=Query)

    # An empty dict is not Ordered
    with pytest.raises(msgspec.ValidationError, match='multi'):
        msgspec.convert(_query(multi={}), type=Query)


def test_ordered_rejects_unknown_keys():
    with pytest.raises(msgspec.ValidationError, match='by'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': ['GM']}], 'multi': {'by': 'label'}},
            type=Query,
        )


def test_ordered_is_constructible_in_python():
    """Tests and downstream callers build specs in Python, not only YAML."""
    query = Query([{'suffix': 'probseg', 'label': ['GM', 'WM']}], multi=Ordered(order='label'))

    assert query.is_multi is True
    assert query.order == 'label'


def test_order_requires_list_values():
    # One entities dict, missing label
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg'}], 'multi': {'order': 'label'}}, type=Query
        )

    # Scalar label
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': 'GM'}], 'multi': {'order': 'label'}},
            type=Query,
        )

    # Multiple entities dicts, one missing label
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {
                'entities': [
                    {'suffix': 'probseg', 'label': ['GM', 'WM']},
                    {'suffix': 'probseg'},
                ],
                'multi': {'order': 'label'},
            },
            type=Query,
        )

    # Empty label list
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': []}], 'multi': {'order': 'label'}},
            type=Query,
        )


def test_order_rejects_a_placeholder_in_its_value_list():
    """A placeholder in the ordering axis can never yield a complete result.

    Theoretically, a query could bind a parameter to a placeholder and use it in
    a sorted list, but there's no clear use case.
    Error for now and relax if/when we know what this should look like.
    """
    with pytest.raises(msgspec.ValidationError, match=r'placeholder.*\{space\}'):
        msgspec.convert(
            {
                'entities': [{'suffix': 'boldref', 'space': ['{space}', 'boldref']}],
                'multi': {'order': 'space'},
            },
            type=Query,
        )

    # Check each alternative
    with pytest.raises(msgspec.ValidationError, match='placeholder'):
        msgspec.convert(
            {
                'entities': [
                    {'suffix': 'boldref', 'space': ['boldref']},
                    {'suffix': 'boldref', 'space': ['{space}']},
                ],
                'multi': {'order': 'space'},
            },
            type=Query,
        )

    # Edge case: Unmatched braces aren't placeholders, though they shouldn't happen
    assert (
        Query(
            [{'suffix': 'boldref', 'space': ['a{b', 'boldref']}], multi=Ordered(order='space')
        ).order
        == 'space'
    )

    # Check direct construction
    with pytest.raises(ValueError, match='placeholder'):
        Query([{'suffix': 'boldref', 'space': ['{space}']}], multi=Ordered(order='space'))


def test_query_placeholders_per_alternative():
    query = Query(
        [
            {'from': 'run', 'to': '{fmapid}', 'suffix': 'xfm'},
            {'from': '{space}', 'to': ['{fmapid}', 'T1w'], 'desc': None},
            {'suffix': 'xfm'},
        ]
    )
    assert query.placeholders == [{'fmapid'}, {'space', 'fmapid'}, set()]


def test_query_rejects_empty_entities():
    with pytest.raises(msgspec.ValidationError, match='entities'):
        msgspec.convert({'entities': []}, type=Query)


def test_query_rejects_unknown_fields():
    """A misspelled field must be rejected, not decoded to its default.

    'scop' for 'scope' would otherwise produce a query with no scope at all,
    which still resolves against run-level datasets -- so the loss would
    surface only as an empty result on the group-level datasets scope exists
    to reach.
    """
    with pytest.raises(msgspec.ValidationError, match='scop'):
        msgspec.convert(_query(scop=['subject']), type=Query)

    # cardinality and labels were in an earlier (unreleased) format
    with pytest.raises(msgspec.ValidationError, match='cardinality'):
        msgspec.convert(_query(cardinality='single'), type=Query)
    with pytest.raises(msgspec.ValidationError, match='labels'):
        msgspec.convert(_query(labels=['GM', 'WM', 'CSF']), type=Query)


# Group struct


def test_group_requires_queries():
    with pytest.raises(msgspec.ValidationError, match='queries'):
        msgspec.convert({'per': 'space'}, type=Group)

    with pytest.raises(msgspec.ValidationError, match='queries'):
        msgspec.convert({'queries': {}}, type=Group)


def test_group_rejects_unknown_fields():
    with pytest.raises(msgspec.ValidationError, match='querys'):
        msgspec.convert({'querys': {'x': _query()}}, type=Group)


def test_group_decodes_per():
    group = msgspec.convert({'queries': {'x': _query()}, 'per': 'space'}, type=Group)

    assert group.per == 'space'
    assert list(group.queries) == ['x']


# load_spec()


def test_bundled_specs_load():
    for name in ('anat', 'func', 'fmap'):
        spec = load_spec(name)
        assert spec, f'{name} spec is empty'
        for group_name, group in spec.items():
            assert group.queries, f'{name}.{group_name} has no queries'
            for query in group.queries.values():
                assert isinstance(query.entities, list)
                assert query.entities
                assert all(isinstance(alt, dict) for alt in query.entities)


def test_bundled_spec_group_names():
    """The group names are the public output keys, so pin them."""
    assert set(load_spec('anat')) == {'images', 'transforms'}
    assert set(load_spec('func')) == {'boldrefs', 'transforms', 'fmap_transforms'}
    assert set(load_spec('fmap')) == {'fieldmaps'}


def test_indexed_groups_declare_their_parameter():
    assert load_spec('anat')['transforms'].per == 'space'
    assert load_spec('fmap')['fieldmaps'].per == 'fmapid'
    assert load_spec('func')['transforms'].per is None
    assert load_spec('func')['fmap_transforms'].per == 'fmapid'


def test_load_spec_reads_a_path(tmp_path):
    path = tmp_path / 'custom.yml'
    path.write_text('images:\n  queries:\n    t1w:\n      entities: [{suffix: T1w}]\n')

    spec = load_spec(path)

    assert spec['images'].queries['t1w'].entities == [{'suffix': 'T1w'}]


# Public API


def test_spec_module_API():
    """The schema types are what a downstream author constructs in Python, so
    they are named exports rather than public-by-accident. The placeholder
    regex is deliberately excluded: it is package-internal.
    """
    import nipost.bids.spec

    types = {'Group', 'Ordered', 'Query', 'Spec'}
    functions = {'load_spec', 'sanitize_fieldmap_id', 'sanitize_space'}

    assert set(nipost.bids.spec.__all__) == types | functions
    for name in nipost.bids.spec.__all__:
        assert hasattr(nipost.bids.spec, name)

    # Functions are package-level exports
    for name in functions:
        assert name in nipost.bids.__all__
    # Types are not
    for name in types:
        assert name not in nipost.bids.__all__
