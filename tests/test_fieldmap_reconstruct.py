# tests/test_fieldmap_reconstruct.py
import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nipost.fieldmap import as_affine, reconstruct_fieldmap


def test_as_affine_collapses_nested_chain():
    chain = nt.TransformChain([nt.Affine(), nt.TransformChain([nt.Affine(), nt.Affine()])])
    assert type(as_affine(chain)) is nt.Affine


def test_as_affine_rejects_per_volume_affines():
    hmc = nt.linear.LinearTransformsMapping([nt.Affine(), nt.Affine()])
    assert as_affine(hmc) is None


def ras(img):
    """The RAS coordinates of every voxel of an image, along a trailing axis."""
    ijk = np.indices(img.shape[:3], dtype=float).reshape(3, -1).T
    return nb.affines.apply_affine(img.affine, ijk).reshape(*img.shape[:3], 3)


def grid(span, zoom=2.0, angle=0.0):
    """An isotropic grid spanning -span..+span mm, rotated ``angle`` radians about k."""
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    affine = nb.affines.from_matvec(rotation * zoom, rotation @ np.full(3, -span))
    return nb.Nifti1Image(np.zeros((int(2 * span / zoom) + 1,) * 3, dtype='f4'), affine)


def las_grid():
    """A 21x17x13 grid centered on 0 mm, with the first voxel axis running right to left."""
    affine = nb.affines.from_matvec(np.diag([-2.0, 2.0, 2.0]), [20, -16, -12])
    return nb.Nifti1Image(np.zeros((21, 17, 13), dtype='f4'), affine)


def asr_grid():
    """An 11-voxel grid whose voxel axes run anterior, superior and right."""
    affine = nb.affines.from_matvec(np.eye(3)[[2, 0, 1]] * 2.0, -10)
    assert nb.aff2axcodes(affine) == ('A', 'S', 'R')
    return nb.Nifti1Image(np.zeros((11, 11, 11), dtype='f4'), affine)


def series_grid():
    """A grid spanning -20..+20 mm followed by a time axis, like a BOLD series."""
    target = grid(span=20.0)
    return nb.Nifti1Image(np.zeros((*target.shape, 3), dtype='f4'), target.affine)


def ramp_coefficients():
    """Coefficients set to each knot's RAS x coordinate, on 4 mm knots over -40..+40 mm.

    Cubic B-splines reproduce cubics, so the field is exactly ``f(x, y, z) = x`` within
    -36..+36 mm. Unlike a constant, a ramp exposes flipped axes and boundary errors.
    """
    knots = grid(span=40.0, zoom=4.0)
    return [nb.Nifti1Image(ras(knots)[..., 0].astype('f4'), knots.affine)]


def shift_warp(dx):
    """A dense field displacing every point ``dx`` mm along x, which is not an Affine."""
    affine = nb.affines.from_matvec(np.eye(3) * 4.0, -40)
    deltas = np.zeros((21, 21, 21, 3), dtype='f4')
    deltas[..., 0] = dx
    return nt.DenseFieldTransform(nb.Nifti1Image(deltas, affine), is_deltas=True)


def quadratic_coefficients():
    """Coefficients ``u**2 - 1/3`` at knot index ``u`` along i, on 4 mm knots over -40..+40 mm.

    Cubic B-splines sample ``u**2`` at the knots as ``u**2 + 1/3``, so the field is
    exactly ``f = ((x + 40) / 4) ** 2``, with ``df/dx = (x + 40) / 8``.
    """
    knots = grid(span=40.0, zoom=4.0)
    u = np.indices(knots.shape, dtype='f4')[0]
    return [nb.Nifti1Image(u**2 - 1 / 3, knots.affine)]


V = np.array([0.06, 0.08, 0.0])


def jacobian(coefficients, target, transforms, v=V) -> nb.Nifti1Image:
    _, jac = reconstruct_fieldmap(coefficients, target, transforms, pe_vector=v)
    return jac


TARGETS = [
    pytest.param(grid(span=20.0), id='RAS'),
    pytest.param(grid(span=20.0, zoom=1.0), id='finer than the knots'),
    pytest.param(las_grid(), id='LAS, non-cubic'),
    pytest.param(asr_grid(), id='axis-permuted (ASR)'),
    pytest.param(grid(span=20.0, angle=np.pi / 6), id='rotated'),
    pytest.param(series_grid(), id='4-D'),
]


def test_reconstruct_fieldmap_zero_outside_support():
    # The field falls off to 0 Hz beyond 48 mm, two knot spacings past the outermost knot.
    target = grid(span=60.0)

    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, nt.Affine())

    data = np.asanyarray(field.dataobj)
    coords = ras(target)
    extent = np.abs(coords).max(axis=-1)
    # Inside support, recover x coordinates exactly; outside support, fall to 0 Hz.
    assert_allclose(data[extent <= 36.0], coords[extent <= 36.0, 0], atol=1e-4)
    assert_allclose(data[extent > 48.0], 0.0, atol=1e-6)


def test_reconstruct_fieldmap_tapers_past_the_lattice():
    # Past the outermost knots, at +/-40 mm, only those within two spacings contribute,
    # with weights 1/6, 4/6 and 1/6: (36 + 4 * 40) / 6 at 40 mm, 40 / 6 at 44 mm.
    affine = nb.affines.from_matvec(np.eye(3) * 4, [-48, 0, 0])
    target = nb.Nifti1Image(np.zeros((25, 1, 1)), affine)  # x = -48..+48 mm, y = z = 0
    taper = [196 / 6, 40 / 6, 0.0]

    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, nt.Affine())

    expected = np.r_[np.negative(taper[::-1]), np.arange(-36, 37, 4), taper]
    assert_allclose(field.dataobj[:, 0, 0], expected, atol=1e-4)


@pytest.mark.parametrize('target', TARGETS)
def test_reconstruct_fieldmap_target_geometry(target):
    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, nt.Affine())

    assert field.shape == target.shape[:3]
    assert_allclose(field.affine, target.affine)
    # Target field-of-view is inside support, recover x coordinates exactly.
    assert_allclose(field.dataobj, ras(target)[..., 0], atol=1e-4)


def test_reconstruct_fieldmap_follows_affine_transform():
    # transforms maps target points into the fieldmap, so target x reads the field at x + 6
    target = grid(span=20.0)
    translation = nb.affines.from_matvec(np.eye(3), [6.0, 0.0, 0.0])

    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, nt.Affine(translation))

    assert_allclose(field.dataobj, ras(target)[..., 0] + 6.0, atol=1e-4)


def test_reconstruct_fieldmap_follows_nonlinear_transform():
    target = grid(span=20.0)

    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, shift_warp(6.0))

    # nitransforms' spline interpolation of the dense field rings in from its edges,
    # displacing a constant 6 mm field by up to 1e-3 mm 20 mm in.
    assert_allclose(field.dataobj, ras(target)[..., 0] + 6.0, atol=5e-3)


def test_reconstruct_fieldmap_sums_levels():
    affine = nb.affines.from_matvec(np.eye(3) * 8, -80)
    coarse = nb.Nifti1Image(np.ones((21, 21, 21), dtype='f4'), affine)
    levels = [*ramp_coefficients(), coarse]
    target = grid(span=20.0)

    # The constant level reads the same on the ramp's lattice, but not the other way around.
    # Testing both orders prevents a lattice swap (e.g., evaluating all levels on the first
    # lattice) from hiding in our choice of order.
    for ordered in (levels, levels[::-1]):
        field, _ = reconstruct_fieldmap(ordered, target, nt.Affine())
        assert_allclose(field.dataobj, ras(target)[..., 0] + 1.0, atol=1e-4)


def test_reconstruct_fieldmap_oblique_lattice():
    # A lattice rotated relative to world; a transposed or un-inverted lattice mapping
    # would pass every axis-aligned case above.
    knots = grid(span=40.0, zoom=4.0, angle=np.pi / 6)
    oblique = nb.Nifti1Image(ras(knots)[..., 0].astype('f4'), knots.affine)
    target = grid(span=12.0)

    field, _ = reconstruct_fieldmap([oblique], target, nt.Affine())

    assert_allclose(field.dataobj, ras(target)[..., 0], atol=1e-4)


def test_reconstruct_fieldmap_header():
    target = grid(span=20.0)

    field, _ = reconstruct_fieldmap(ramp_coefficients(), target, nt.Affine())

    assert field.header.get_intent() == ('estimate', (), 'fieldmap Hz')
    assert field.get_data_dtype() == np.float32
    peak = np.abs(field.dataobj).max()
    assert_allclose(field.header['cal_max'], peak, rtol=1e-6)
    assert_allclose(field.header['cal_min'], -peak, rtol=1e-6)


def test_reconstruct_fieldmap_non_cubic_lattice():
    affine = nb.affines.from_matvec(np.eye(3) * 4.0, [-40.0, -32.0, -24.0])
    knots = nb.Nifti1Image(np.zeros((21, 17, 13), dtype='f4'), affine)
    lattice = nb.Nifti1Image(ras(knots)[..., 2].astype('f4'), affine)
    target = grid(span=12.0)

    field, _ = reconstruct_fieldmap([lattice], target, nt.Affine())

    assert_allclose(field.dataobj, ras(target)[..., 2], atol=1e-4)


def test_reconstruct_fieldmap_singleton_axis():
    (level,) = ramp_coefficients()
    single = level.slicer[..., np.newaxis]
    target = grid(span=20.0)

    field, _ = reconstruct_fieldmap([single], target, nt.Affine())

    assert field.shape == target.shape
    assert_allclose(field.dataobj, ras(target)[..., 0], atol=1e-4)


@pytest.mark.parametrize('target', TARGETS)
def test_reconstruct_fieldmap_jacobian_of_ramp(target):
    jac = jacobian(ramp_coefficients(), target, nt.Affine())

    assert jac.shape == target.shape[:3]
    assert_allclose(jac.affine, target.affine)
    assert_allclose(jac.dataobj, 1.06, atol=1e-6)


@pytest.mark.parametrize(
    'target',
    [
        pytest.param(grid(span=20.0), id='RAS'),
        pytest.param(las_grid(), id='LAS'),
        pytest.param(grid(span=20.0, angle=np.pi / 6), id='rotated'),
    ],
)
def test_reconstruct_fieldmap_jacobian_follows_field_curvature(target):
    jac = jacobian(quadratic_coefficients(), target, nt.Affine())

    assert_allclose(jac.dataobj, 1 + 0.06 * (ras(target)[..., 0] + 40) / 8, atol=1e-6)


def test_reconstruct_fieldmap_jacobian_sums_levels():
    affine = nb.affines.from_matvec(np.eye(3) * 8, -80)
    knots = grid(span=80.0, zoom=8.0)
    coarse = nb.Nifti1Image(ras(knots)[..., 0].astype('f4'), affine)
    levels = [*ramp_coefficients(), coarse]
    target = grid(span=20.0)

    for ordered in (levels, levels[::-1]):
        jac = jacobian(ordered, target, nt.Affine())
        assert_allclose(jac.dataobj, 1.12, atol=1e-6)


def test_reconstruct_fieldmap_jacobian_on_oblique_lattice():
    knots = grid(span=40.0, zoom=4.0, angle=0.5)
    g = np.array([0.7, -0.2, 0.5])
    oblique = nb.Nifti1Image((ras(knots) @ g).astype('f4'), knots.affine)
    v = np.array([0.06, 0.08, -0.03])

    jac = jacobian([oblique], grid(span=12.0), nt.Affine(), v=v)

    assert_allclose(jac.dataobj, 1 + g @ v, atol=1e-5)


def test_reconstruct_fieldmap_jacobian_follows_nonlinear_transform():
    target = grid(span=20.0)

    jac = jacobian(quadratic_coefficients(), target, shift_warp(6.0))

    # The warp's ringing, as in the nonlinear field test
    assert_allclose(jac.dataobj, 1 + 0.06 * (ras(target)[..., 0] + 46) / 8, atol=1e-4)


def test_reconstruct_fieldmap_jacobian_excludes_the_transform_jacobian():
    target = grid(span=10.0)
    scale = nt.Affine(np.diag([2.0, 2.0, 2.0, 1.0]))

    jac = jacobian(quadratic_coefficients(), target, scale)

    # Read at the scaled point 2x, and not multiplied by the transform's own factor of 2
    assert_allclose(jac.dataobj, 1 + 0.06 * (2 * ras(target)[..., 0] + 40) / 8, atol=1e-6)


def test_reconstruct_fieldmap_jacobian_is_one_outside_support():
    target = grid(span=60.0)

    jac = jacobian(ramp_coefficients(), target, nt.Affine())

    extent = np.abs(ras(target)).max(axis=-1)
    data = np.asanyarray(jac.dataobj)
    assert_allclose(data[extent > 48.0], 1.0, atol=1e-6)
    assert_allclose(data[extent <= 36.0], 1.06, atol=1e-6)


def test_reconstruct_fieldmap_field_unchanged_by_pe_vector():
    target = grid(span=20.0, angle=np.pi / 6)

    plain, _ = reconstruct_fieldmap(quadratic_coefficients(), target, nt.Affine())
    field, _ = reconstruct_fieldmap(quadratic_coefficients(), target, nt.Affine(), pe_vector=V)

    assert_array_equal(field.dataobj, plain.dataobj)


def test_reconstruct_fieldmap_jacobian_header():
    jac = jacobian(ramp_coefficients(), grid(span=20.0), nt.Affine())

    assert jac.header.get_intent() == ('estimate', (), 'jacobian')
    assert jac.get_data_dtype() == np.float32
