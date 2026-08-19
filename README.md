# nipost

**nipost** is a standalone library for one-shot resampling of fMRIPrep _minimal
derivatives_ into a target space. It combines head-motion correction (HMC),
susceptibility distortion correction (SDC), and spatial normalization in a
**single interpolation step**, avoiding accumulation of interpolation errors
that would occur if each transform were applied sequentially.

## Installation

```bash
pip install nipost
```

For BIDS derivative discovery (`collect_derivatives`, `collect_fieldmaps`):

```bash
pip install 'nipost[bids]'
```

**Requires Python ≥ 3.12.**

## Quick start

```python
import nibabel as nb
from nipost import load_transforms, reconstruct_fieldmap, resample_image
from nipost.bids import collect_derivatives, collect_fieldmaps
from nipost.bids.spec import load_spec

# 1. Discover derivatives from an fMRIPrep output directory
func = collect_derivatives(deriv_root, spec=load_spec('func'), entities=bold_entities)
anat = collect_derivatives(
    deriv_root, spec=load_spec('anat'), subject_id=subject, std_spaces=['MNI152NLin2009cAsym']
)
fmaps = collect_fieldmaps(deriv_root, entities={'subject': subject})

# 2. Build transform chains (HMC → boldref→anat → anat→std)
bold2std = load_transforms(
    [func['transforms']['hmc'], func['transforms']['boldref2anat'], anat2std_xfm],
    inverse=[False],
)
fmap2std = load_transforms(
    [func['transforms']['boldref2fmap'][0], func['transforms']['boldref2anat'], anat2std_xfm],
    inverse=[True, False, False],
)

# 3. Reconstruct the fieldmap (B-Spline coefficients → Hz image in target space)
coeff = nb.load(fmaps[fmapid]['coeffs'])
fmapref = nb.load(fmaps[fmapid]['magnitude'])
fmap_std = reconstruct_fieldmap([coeff], fmapref, target, fmap2std)

# 4. Resample BOLD in one shot — HMC + SDC + normalization simultaneously
bold_mni = resample_image(
    source=bold,
    target=target,
    transforms=bold2std,
    fieldmap=fmap_std,
    pe_info=pe_info,
)
```

## API reference

### Core (no optional dependencies)

| Symbol                           | Description                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| `nipost.resample_image`          | Resample a 3-/4-D BOLD image into a target space, applying HMC + SDC in one interpolation pass. |
| `nipost.reconstruct_fieldmap`    | Evaluate B-Spline fieldmap coefficients and resample the result into a target space.            |
| `nipost.load_transforms`         | Load a series of transform files and compose them into a `nitransforms` chain.                  |
| `nipost.get_trt`                 | Derive the total readout time from BIDS sidecar metadata.                                       |
| `nipost.ensure_positive_cosines` | Reorient an image so all direction cosines are positive (normalises PE axis bookkeeping).       |

### `nipost[bids]` extra

Requires `pybids` and `niworkflows`.

| Symbol                            | Description                                                           |
| --------------------------------- | --------------------------------------------------------------------- |
| `nipost.bids.collect_derivatives` | Spec-driven discovery of fMRIPrep derivatives (images, transforms).   |
| `nipost.bids.collect_fieldmaps`   | Collect B-Spline fieldmap derivatives grouped by fieldmap ID.         |
| `nipost.bids.spec.load_spec`      | Load a bundled spec (`"anat"` / `"func"`) or a custom JSON spec file. |

## Python version support

nipost supports Python ≥ 3.12.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
