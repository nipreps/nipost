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
    sanitize_fieldmap_id,
    sanitize_space,
)


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


# --- the §3.2 shape table, one test per row ---


def test_shape_omitted_is_scalar():
    query = msgspec.convert(_query(), type=Query)

    assert query.is_multi is False
    assert query.order is None


def test_shape_false_is_an_explicit_scalar():
    assert msgspec.convert(_query(multi=False), type=Query).is_multi is False


def test_shape_true_is_an_unordered_list():
    query = msgspec.convert(_query(multi=True), type=Query)

    assert query.is_multi is True
    assert query.order is None


def test_shape_ordered_is_a_list_with_an_axis():
    query = msgspec.convert(
        {
            'entities': [{'suffix': 'probseg', 'label': ['GM', 'WM']}],
            'multi': {'order': 'label'},
        },
        type=Query,
    )

    assert query.is_multi is True
    assert query.order == 'label'


def test_shape_null_is_a_decode_error():
    """'null' is not one of the variants."""
    with pytest.raises(msgspec.ValidationError, match='multi'):
        msgspec.convert(_query(multi=None), type=Query)


def test_scalar_and_ordered_is_unrepresentable():
    """The contradiction an earlier draft needed a tri-state to reject.

    'multi' holds a bool or an Ordered, never both, so there is no way to
    write a query that is a scalar and an ordered list at once. This test
    pins that the two spellings are alternatives rather than companions.
    """
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(
            {
                'entities': [{'suffix': 'probseg', 'label': ['GM']}],
                'multi': {'order': 'label', 'multi': False},
            },
            type=Query,
        )


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


# --- order validation ---


def test_order_requires_every_alternative_to_declare_it_as_a_list():
    """order takes its sequence from the entity constraint, so the constraint
    has to be there -- in every alternative, since any of them may match."""
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg'}], 'multi': {'order': 'label'}}, type=Query
        )


def test_order_rejects_a_scalar_entity_value():
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': 'GM'}], 'multi': {'order': 'label'}},
            type=Query,
        )


def test_order_checks_every_alternative_not_just_the_first():
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


def test_order_rejects_an_empty_value_list():
    """An empty sequence would drop the constraint and never yield a key."""
    with pytest.raises(msgspec.ValidationError, match='label'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': []}], 'multi': {'order': 'label'}},
            type=Query,
        )


def test_order_rejects_a_placeholder_in_its_value_list():
    """A placeholder in the ordering axis can never yield a complete result.

    ``_reduce_ordered`` matches the declared axis against entity values read
    out of filenames, which never contain ``{...}``. So such a query omitted
    its key for every binding of the parameter -- bound, unbound, or supplied
    by an enclosing ``per`` -- and did it silently, absence being a legal
    outcome. Rejected at load rather than resolved: see the design's section 4.
    """
    with pytest.raises(msgspec.ValidationError, match='placeholder'):
        msgspec.convert(
            {
                'entities': [{'suffix': 'boldref', 'space': ['{space}', 'boldref']}],
                'multi': {'order': 'space'},
            },
            type=Query,
        )


def test_order_placeholder_rejection_names_the_offending_value():
    """The message has to say which member is the problem, since the axis is
    also the entity constraint and the rest of the list is legitimate."""
    with pytest.raises(msgspec.ValidationError, match=r'\{space\}'):
        msgspec.convert(
            {
                'entities': [{'suffix': 'boldref', 'space': ['boldref', '{space}']}],
                'multi': {'order': 'space'},
            },
            type=Query,
        )


def test_order_checks_every_alternative_for_placeholders():
    """Any alternative may be the one that matches, so all are checked --
    matching how the list-and-non-empty checks already behave."""
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


def test_order_allows_a_braced_value_that_is_not_a_placeholder():
    """The guard must key off the placeholder syntax, not the presence of a
    brace, so it cannot reject a legitimate value."""
    query = Query(
        [{'suffix': 'boldref', 'space': ['a{b', 'boldref']}], multi=Ordered(order='space')
    )

    assert query.order == 'space'


def test_order_placeholder_rejection_raises_on_direct_construction():
    """Downstream authors build specs in Python, not only YAML, so the guard
    has to fire outside msgspec's decoding path too."""
    with pytest.raises(ValueError, match='placeholder'):
        Query([{'suffix': 'boldref', 'space': ['{space}']}], multi=Ordered(order='space'))


# --- rejection of malformed queries and groups ---


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


def test_query_rejects_the_removed_cardinality_field():
    """Migration must be loud: an old spec fails to load rather than
    decoding to a scalar query by accident."""
    with pytest.raises(msgspec.ValidationError, match='cardinality'):
        msgspec.convert(_query(cardinality='single'), type=Query)


def test_query_rejects_the_removed_labels_field():
    with pytest.raises(msgspec.ValidationError, match='labels'):
        msgspec.convert(_query(labels=['GM', 'WM', 'CSF']), type=Query)


def test_query_rejects_a_top_level_order_field():
    """An intermediate draft had 'order' as a sibling of 'multi'; it now nests."""
    with pytest.raises(msgspec.ValidationError, match='order'):
        msgspec.convert(
            {'entities': [{'suffix': 'probseg', 'label': ['GM']}], 'order': 'label'},
            type=Query,
        )


def test_group_requires_queries():
    with pytest.raises(msgspec.ValidationError, match='queries'):
        msgspec.convert({'per': 'space'}, type=Group)


def test_group_rejects_empty_queries():
    with pytest.raises(msgspec.ValidationError, match='queries'):
        msgspec.convert({'queries': {}}, type=Group)


def test_group_rejects_unknown_fields():
    with pytest.raises(msgspec.ValidationError, match='querys'):
        msgspec.convert({'querys': {'x': _query()}}, type=Group)


def test_group_decodes_per():
    group = msgspec.convert({'queries': {'x': _query()}, 'per': 'space'}, type=Group)

    assert group.per == 'space'
    assert list(group.queries) == ['x']


# --- the loader ---


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
    assert set(load_spec('func')) == {'boldrefs', 'transforms'}
    assert set(load_spec('fmap')) == {'fieldmaps'}


def test_indexed_groups_declare_their_parameter():
    assert load_spec('anat')['transforms'].per == 'space'
    assert load_spec('fmap')['fieldmaps'].per == 'fmapid'
    assert load_spec('func')['transforms'].per is None


def test_load_spec_reads_a_path(tmp_path):
    path = tmp_path / 'custom.yml'
    path.write_text('images:\n  queries:\n    t1w:\n      entities: [{suffix: T1w}]\n')

    spec = load_spec(path)

    assert spec['images'].queries['t1w'].entities == [{'suffix': 'T1w'}]


# --- the public sanitizers ---


def test_sanitize_space_converts_cohort_syntax():
    assert sanitize_space('MNI152NLin6Asym:cohort-1') == 'MNI152NLin6Asym+1'


def test_sanitize_space_leaves_a_plain_space_alone():
    assert sanitize_space('MNI152NLin2009cAsym') == 'MNI152NLin2009cAsym'


def test_sanitize_fieldmap_id_strips_non_alphanumerics():
    assert sanitize_fieldmap_id('auto_00000') == 'auto00000'


def test_sanitize_fieldmap_id_would_eat_a_cohort_plus():
    """Why the two sanitizers cannot be merged into one universal rule."""
    assert sanitize_fieldmap_id('MNI152NLin6Asym+1') == 'MNI152NLin6Asym1'


# --- public surface ---


def test_spec_module_declares_its_public_surface():
    """The schema types are what a downstream author constructs in Python, so
    they are named exports rather than public-by-accident. The placeholder
    regex is deliberately excluded: it is package-internal.
    """
    from nipost.bids import spec

    assert spec.__all__ == [
        'Group',
        'Ordered',
        'Query',
        'Spec',
        'load_spec',
        'sanitize_fieldmap_id',
        'sanitize_space',
    ]


def test_spec_public_surface_is_importable():
    """__all__ has to name things that exist -- a stale entry breaks
    ``from nipost.bids.spec import *`` rather than failing here."""
    from nipost.bids import spec

    for name in spec.__all__:
        assert hasattr(spec, name), name


def test_schema_types_are_not_promoted_to_the_package_namespace():
    """Deliberate: nipost.bids stays the calling surface. Authors reach the
    types through nipost.bids.spec."""
    import nipost.bids

    for name in ('Group', 'Ordered', 'Query', 'Spec'):
        assert name not in nipost.bids.__all__, name
