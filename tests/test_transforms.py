import warnings

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nipost.transforms import collapse_transforms, load_transforms

POINTS = np.random.default_rng(0).uniform(-10, 10, (50, 3))
GRID = nb.Nifti1Image(np.zeros((5, 6, 7), dtype='u1'), np.diag([2.0, 2.0, 2.0, 1.0]))


def rigid(angle, shift):
    """An Affine rotating ``angle`` radians about z, then shifting by ``shift`` mm."""
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return nt.Affine(nb.affines.from_matvec(rotation, shift))


def shift_warp(dx):
    """A dense field displacing every point ``dx`` mm along x, over -40..+40 mm."""
    affine = nb.affines.from_matvec(np.eye(3) * 4.0, -40)
    deltas = np.zeros((21, 21, 21, 3), dtype='f4')
    deltas[..., 0] = dx
    return nt.DenseFieldTransform(nb.Nifti1Image(deltas, affine), is_deltas=True)


def test_empty_list_returns_identity():
    xfm = load_transforms([], [])
    assert np.allclose(xfm.matrix, np.eye(4))


def test_single_affine_roundtrip(tmp_path):
    aff = nt.Affine(np.diag([2.0, 2.0, 2.0, 1.0]))
    p = tmp_path / 'xfm.txt'
    aff.to_filename(p, fmt='itk')

    loaded = load_transforms([p], [False])
    assert np.allclose(loaded.matrix, aff.matrix)


def test_inverse_flag_inverts(tmp_path):
    aff = nt.Affine(np.diag([2.0, 2.0, 2.0, 1.0]))
    p = tmp_path / 'xfm.txt'
    aff.to_filename(p, fmt='itk')

    loaded = load_transforms([p], [True])
    assert np.allclose(loaded.matrix, np.linalg.inv(aff.matrix))


def test_mismatched_inverse_length_raises(tmp_path):
    import pytest

    p = tmp_path / 'xfm.txt'
    nt.Affine().to_filename(p, fmt='itk')
    with pytest.raises(ValueError, match='Mismatched'):
        load_transforms([p, p], [True, False, True])


def test_collapse_transforms_merges_affines():
    a, b, c = rigid(0.1, [1, 2, 3]), rigid(-0.3, [0, -4, 2]), rigid(0.5, [2, 0, -1])
    chain = nt.TransformChain([a, nt.TransformChain([b, nt.base.TransformBase()]), c])

    collapsed = collapse_transforms(chain)

    assert type(collapsed) is nt.Affine
    assert_allclose(collapsed.map(POINTS), chain.map(POINTS), atol=1e-6)


def test_collapse_transforms_keeps_warp_between_affines():
    # The shape of a target-to-fieldmap chain through a template warp
    a, b, c, d = (rigid(0.1 * k, [k, -k, 1]) for k in range(4))
    chain = nt.TransformChain([a, nt.TransformChain([shift_warp(3.0), b, c]), d])

    collapsed = collapse_transforms(chain)

    assert [type(x) for x in collapsed.transforms] == [
        nt.Affine,
        nt.DenseFieldTransform,
        nt.Affine,
    ]
    assert_allclose(collapsed.map(POINTS), chain.map(POINTS), atol=1e-6)


def test_collapse_transforms_keeps_per_volume_affines():
    hmc = nt.linear.LinearTransformsMapping([rigid(0.1, [1, 0, 0]), rigid(0.2, [0, 1, 0])])
    chain = nt.TransformChain([hmc, rigid(0.3, [0, 0, 1]), rigid(-0.1, [2, 0, 0])])

    collapsed = collapse_transforms(chain)

    assert [type(x) for x in collapsed.transforms] == [
        nt.linear.LinearTransformsMapping,
        nt.Affine,
    ]


def test_collapse_transforms_identity():
    collapsed = collapse_transforms(nt.base.TransformBase())

    assert type(collapsed) is nt.Affine
    assert_allclose(collapsed.matrix, np.eye(4))


def test_collapse_transforms_keeps_reference():
    grid = nb.Nifti1Image(np.zeros((5, 6, 7), dtype='u1'), np.diag([2.0, 2.0, 2.0, 1.0]))
    chain = nt.TransformChain([nt.Affine(grid.affine, reference=grid), rigid(0.1, [1, 0, 0])])

    # nitransforms warns when an unset reference is read
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        collapsed = collapse_transforms(chain)
        assert collapsed.reference.shape == grid.shape


def chain_on_grid(transforms):
    """A chain whose own reference is GRID, whatever its transforms carry."""
    chain = nt.TransformChain(transforms)
    chain.reference = GRID
    return chain


@pytest.mark.parametrize(
    'make',
    [
        pytest.param(lambda: nt.base.TransformBase(reference=GRID), id='identity'),
        pytest.param(
            lambda: nt.TransformChain(
                [nt.base.TransformBase(reference=GRID), rigid(0.1, [1, 0, 0])]
            ),
            id='leading identity',
        ),
        pytest.param(
            lambda: chain_on_grid([rigid(0.1, [1, 0, 0]), rigid(0.2, [0, 1, 0])]),
            id='set on the chain',
        ),
    ],
)
def test_collapse_transforms_keeps_reference_off_the_first_affine(make):
    collapsed = collapse_transforms(make())

    assert collapsed.reference.shape == GRID.shape


def test_collapse_transforms_keeps_reference_of_a_lone_warp():
    warp = shift_warp(3.0)

    collapsed = collapse_transforms(chain_on_grid([warp]))

    assert collapsed.reference.shape == GRID.shape
    assert warp.reference.shape == (21, 21, 21)
