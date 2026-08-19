"""End-to-end test.

This test transcribes ``repos/fmriprep-resampling-demo/resample.ipynb`` using
only nipost (plus templateflow, nibabel, nilearn).  It asserts the byte-exact
checksum ``e4a5dede``.

Note that this checksum differs from the one in the notebook because the notebook used
an old version of nilearn that defaulted to copy_header=False.
Rather than preserve that behavior, we update to copy_header=True, and update the checksum.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('bids')
pytest.importorskip('templateflow')

data = Path(__file__).parent / 'data'

pytestmark = pytest.mark.skipif(
    not (data / 'ds005365-fmriprep').exists(),
    reason='NIPOST_DEMO_ROOT not set or demo DataLad inputs not fetched',
)


def test_demo_reproduces_checksum() -> None:
    """Reproduce the fmriprep-resampling-demo checksum using only nipost."""
    import nibabel as nb
    from bids import BIDSLayout
    from nilearn import image as nli
    from templateflow import TemplateFlowClient

    # Needed for nipreps.json to support nonstandard entities
    import niworkflows.data

    from nipost import (
        load_transforms,
        prepare_epi,
        reconstruct_fieldmap,
        resample_image,
    )
    from nipost.bids import collect_derivatives, collect_fieldmaps
    from nipost.bids.spec import load_spec

    tf = TemplateFlowClient()

    raw = BIDSLayout(data / 'ds005365')
    deriv_root = data / 'ds005365-fmriprep'

    template = 'MNI152NLin2009cAsym'
    bold_file = raw.get(suffix='bold', extension='.nii.gz')[0]
    MNI_file = tf.get(template=template, suffix='mask', desc='brain', resolution='02')

    bold_entities = bold_file.get_entities()
    for ent in ('datatype', 'suffix', 'extension'):
        bold_entities.pop(ent, None)
    subject = bold_entities['subject']

    anat = collect_derivatives(
        deriv_root,
        spec=load_spec('anat'),
        subject_id=subject,
        std_spaces=[template],
    )
    func = collect_derivatives(
        deriv_root,
        spec=load_spec('func'),
        entities=bold_entities,
    )
    fmaps = collect_fieldmaps(deriv_root, entities={'subject': subject})

    hmc_xfm = func['transforms']['hmc']
    boldref2anat_xfm = func['transforms']['boldref2anat']
    anat2std_xfm = anat['transforms'][template]['forward']
    bold2std_xfms = [hmc_xfm, boldref2anat_xfm, anat2std_xfm]

    # boldref2fmap is a list; [0] is the actual fmap xfm (no desc entity)
    boldref2fmap_xfm = func['transforms']['boldref2fmap'][0]
    fmap2std_xfms = [boldref2fmap_xfm, boldref2anat_xfm, anat2std_xfm]
    fmap2std_inv = [True, False, False]

    # The notebook used: deriv.files[boldref2fmap_xfm].entities['to']
    # We replicate this via BIDSLayout with the nipreps config.

    deriv_layout = BIDSLayout(
        str(deriv_root),
        config=[niworkflows.data.load('nipreps.json')],
        validate=False,
    )
    fmapid = deriv_layout.files[boldref2fmap_xfm].entities['to']

    coeff_file = fmaps[fmapid]['coeffs']
    fmapref_file = fmaps[fmapid]['magnitude']

    bold, pe_info = prepare_epi(nb.load(bold_file), bold_file.get_metadata())
    MNI = nli.crop_img(MNI_file, copy_header=True)
    fmapref = nb.load(fmapref_file)
    coeff = nb.load(coeff_file)

    bold2std = load_transforms(
        bold2std_xfms, inverse=[False]
    )  # single-element inverse list broadcasts to all transforms in the chain
    fmap2std = load_transforms(fmap2std_xfms, inverse=fmap2std_inv)

    fmap_std = reconstruct_fieldmap([coeff], fmapref, MNI, fmap2std)

    bold_mni = resample_image(
        source=bold,
        target=MNI,
        transforms=bold2std,
        fieldmap=fmap_std,
        pe_info=pe_info,
        jacobian=False,
        mode='grid-constant',
    )

    # Data and affine are the important bits
    assert sha256(np.asanyarray(bold_mni.dataobj).tobytes()).hexdigest()[:8] == '2655c92a'
    assert sha256(bold_mni.affine.tobytes()).hexdigest()[:8] == 'c532bd33'
    # The full serialization (with header) is worth checking too
    assert sha256(bold_mni.to_bytes()).hexdigest()[:8] == 'e4a5dede'
