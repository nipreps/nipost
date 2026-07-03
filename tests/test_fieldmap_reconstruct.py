# tests/test_fieldmap_reconstruct.py
import nibabel as nb
import nitransforms as nt
import numpy as np

from nipost.fieldmap import aligned, as_affine, reconstruct_fieldmap


def test_aligned_true_for_same_orientation():
    assert aligned(np.eye(4), np.diag([2.0, 2.0, 2.0, 1.0]))


def test_aligned_false_for_rotated():
    rot = np.eye(4)
    rot[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert not aligned(np.eye(4), rot)


def test_as_affine_collapses_identity_chain():
    # nt.TransformChain is in nt.manip; nt.base.TransformChain does not exist in 25.x
    chain = nt.TransformChain([nt.Affine(), nt.Affine()])
    assert isinstance(as_affine(chain), nt.Affine)


def test_reconstruct_fieldmap_direct_grid(tmp_path):
    # target aligned with coefficient grid -> direct reconstruction path
    target = nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4))
    fmapref = nb.Nifti1Image(np.zeros((6, 6, 6), dtype='f4'), np.eye(4))
    coeff = nb.Nifti1Image(np.ones((5, 5, 5), dtype='f4'), np.diag([2.0, 2.0, 2.0, 1.0]))

    field = reconstruct_fieldmap(
        coefficients=[coeff],
        fmap_reference=fmapref,
        target=target,
        transforms=nt.TransformChain([nt.Affine()]),
    )
    assert field.shape == target.shape
    assert np.isfinite(np.asanyarray(field.dataobj)).all()
