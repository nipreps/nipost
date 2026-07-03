"""End-to-end acceptance gate: reproduce the fmriprep-resampling-demo checksum.

This test transcribes ``repos/fmriprep-resampling-demo/resample.ipynb`` using
only nipost (plus templateflow, nibabel, nilearn).  It asserts the byte-exact
checksum ``a91beb24``, which was established in Task 0 against pinned versions:

  scipy==1.17.1  nitransforms==25.1.0  numpy==2.4.5  nibabel==5.4.2
  templateflow==25.1.2  pybids==0.22.0  niworkflows==1.14.4  nilearn==0.12.1

Run against the acceptance environment::

    cd repos/nipost
    uv venv --python 3.12 .venv-accept
    uv pip install --python .venv-accept -e '.[bids]' \\
        'scipy==1.17.1' 'nitransforms==25.1.0' 'numpy==2.4.5' 'nibabel==5.4.2' \\
        'templateflow==25.1.2' 'pybids==0.22.0' 'niworkflows==1.14.4' \\
        'nilearn==0.12.1' pytest nest-asyncio
    NIPOST_DEMO_ROOT=../fmriprep-resampling-demo .venv-accept/bin/pytest tests/integration -v
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

pytest.importorskip('bids')
pytest.importorskip('templateflow')

_DEMO_ROOT_ENV = os.environ.get('NIPOST_DEMO_ROOT', '')
DEMO = Path(_DEMO_ROOT_ENV or Path(__file__).parents[3] / 'fmriprep-resampling-demo')

pytestmark = pytest.mark.skipif(
    not _DEMO_ROOT_ENV or not (DEMO / 'inputs' / 'ds005365-fmriprep').exists(),
    reason='NIPOST_DEMO_ROOT not set or demo DataLad inputs not fetched',
)


def test_demo_reproduces_checksum() -> None:
    """Reproduce the fmriprep-resampling-demo checksum using only nipost."""
    import nibabel as nb
    from bids import BIDSLayout
    from nilearn import image as nli
    from templateflow import TemplateFlowClient

    from nipost import (
        ensure_positive_cosines,
        get_trt,
        load_transforms,
        reconstruct_fieldmap,
        resample_image,
    )
    from nipost.bids import collect_derivatives, collect_fieldmaps
    from nipost.bids.spec import load_spec

    tf = TemplateFlowClient()

    # --- Cell: Load BIDS layouts --------------------------------------------------
    raw = BIDSLayout(str(DEMO / 'inputs' / 'ds005365'))
    deriv_root = DEMO / 'inputs' / 'ds005365-fmriprep'

    # --- Cell: Identify the BOLD file and template --------------------------------
    template = 'MNI152NLin2009cAsym'
    bold_file = raw.get(suffix='bold', extension='.nii.gz')[0]
    MNI_file = tf.get(template=template, suffix='mask', desc='brain', resolution='02')

    # --- Cell: Extract BOLD entities ----------------------------------------------
    bold_entities = bold_file.get_entities()
    for ent in ('datatype', 'suffix', 'extension'):
        bold_entities.pop(ent, None)
    subject = bold_entities['subject']

    # --- Cell: Collect derivatives ------------------------------------------------
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

    # --- Cell: Build bold2std transform chain ------------------------------------
    hmc_xfm = func['transforms']['hmc']
    boldref2anat_xfm = func['transforms']['boldref2anat']
    anat2std_xfm = anat['transforms'][template]['forward']
    bold2std_xfms = [hmc_xfm, boldref2anat_xfm, anat2std_xfm]

    # --- Cell: Build fmap2std transform chain ------------------------------------
    # boldref2fmap is a list; [0] is the actual fmap xfm (no desc entity)
    boldref2fmap_xfm = func['transforms']['boldref2fmap'][0]
    fmap2std_xfms = [boldref2fmap_xfm, boldref2anat_xfm, anat2std_xfm]
    fmap2std_inv = [True, False, False]

    # --- Cell: Extract fieldmap id from the xfm filename -------------------------
    # The notebook used: deriv.files[boldref2fmap_xfm].entities['to']
    # We replicate this via BIDSLayout with the nipreps config.
    import niworkflows.data

    deriv_layout = BIDSLayout(
        str(deriv_root),
        config=[niworkflows.data.load('nipreps.json')],
        validate=False,
    )
    fmapid = deriv_layout.files[boldref2fmap_xfm].entities['to']

    # --- Cell: Load fieldmap coefficient and reference ---------------------------
    coeff_file = fmaps[fmapid]['coeffs']
    fmapref_file = fmaps[fmapid]['magnitude']

    # --- prepare_bold helper (notebook cell) -------------------------------------
    def prepare_bold(
        bids_file: object,
    ) -> tuple[nb.Nifti1Image, list[tuple[int, float]]]:
        bold = nb.load(bids_file)  # type: ignore[arg-type]
        source, axcodes = ensure_positive_cosines(bold)

        metadata = bids_file.get_metadata()  # type: ignore[attr-defined]
        trt = get_trt(metadata, bids_file)
        pe_dir = metadata['PhaseEncodingDirection']
        pe_axis = 'ijk'.index(pe_dir[0])

        pe_flip = pe_dir.endswith('-')
        axis_flip = axcodes[pe_axis] in 'LPI'

        pe_info: list[tuple[int, float]] = [
            (pe_axis, -trt if (axis_flip ^ pe_flip) else trt)
        ] * source.shape[3]

        return source, pe_info

    # --- Cell: Load images and transforms ----------------------------------------
    bold, pe_info = prepare_bold(bold_file)
    MNI = nli.crop_img(MNI_file)
    fmapref = nb.load(fmapref_file)
    coeff = nb.load(coeff_file)

    bold2std = load_transforms(bold2std_xfms, inverse=[False])
    fmap2std = load_transforms(fmap2std_xfms, inverse=fmap2std_inv)

    # --- Cell: Reconstruct the fieldmap in standard space ------------------------
    fmap_std = reconstruct_fieldmap([coeff], fmapref, MNI, fmap2std)

    # --- Cell: Resample BOLD to MNI ----------------------------------------------
    bold_mni = resample_image(
        source=bold,
        target=MNI,
        transforms=bold2std,
        fieldmap=fmap_std,
        pe_info=pe_info,
        jacobian=False,
        mode='grid-constant',
    )

    # --- Assert checksum ----------------------------------------------------------
    assert sha256(bold_mni.to_bytes()).hexdigest()[:8] == 'a91beb24'
