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
transforms = func['transforms']
anat2std_xfm = anat['transforms']['MNI152NLin2009cAsym']['forward']

# fMRIPrep's --bold-coreg-level decides whether each run is coregistered to
# the anatomical reference directly, or via a shared session/subject boldref
# template. Select whichever leg the dataset actually provides.
run2anat_xfms = next(
    [transforms[k] for k in xfmset]
    for xfmset in [
        ['run2anat'],
        ['run2session', 'session2anat'],
        ['run2subject', 'subject2anat'],
    ]
    if all(k in transforms for k in xfmset)
)

bold2std = load_transforms(
    [transforms['hmc'], *run2anat_xfms, anat2std_xfm],
    inverse=[False],
)

# 3. Reconstruct the fieldmap (B-Spline coefficients → Hz image in target space)
# run2fmap is a list; it is [] on a dataset with no fieldmap (SDC skipped),
# so index it only after checking it is non-empty.
fmap_std = None
if transforms['run2fmap']:
    fmap2std = load_transforms(
        [transforms['run2fmap'][0], *run2anat_xfms, anat2std_xfm],
        inverse=[True, *[False] * (len(run2anat_xfms) + 1)],
    )
    # 'coeffs' is always a list: one entry per B-Spline level.
    coeffs = [nb.load(path) for path in fmaps[fmapid]['coeffs']]
    fmapref = nb.load(fmaps[fmapid]['magnitude'])
    fmap_std = reconstruct_fieldmap(coeffs, fmapref, target, fmap2std)

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

Requires `pybids`, `niworkflows`, and `msgspec[yaml]`.

| Symbol                            | Description                                                          |
| --------------------------------- | -------------------------------------------------------------------- |
| `nipost.bids.collect_derivatives` | Spec-driven discovery of fMRIPrep derivatives (images, transforms).  |
| `nipost.bids.collect_fieldmaps`   | Collect B-Spline fieldmap derivatives grouped by fieldmap ID.        |
| `nipost.bids.spec.load_spec`      | Load a bundled spec (`"anat"` / `"func"` / `"fmap"`) or a spec file. |

### Spec schema

A spec has up to three sections — `items` (flat results), `transforms` (flat,
under `transforms`), and `space_transforms` (nested under
`transforms[space]`). The bundled specs are YAML; `load_spec` reads any path,
and a JSON file works too, JSON being valid YAML. Each entry is a query:

```yaml
# illustrative: shows both alternatives and scope on one query
example_boldref:
  entities:
    - space: run
      suffix: boldref
    - desc: coreg
      suffix: boldref
  scope: [subject, session]
  cardinality: single
```

- **`entities`** — ordered entity dicts describing one logical item under
  different naming schemes. The first alternative that matches anything is
  used; cardinality applies to that alternative alone, never to a union. Put
  current naming first.
- **`scope`** — allowlist of caller-supplied entity names the query accepts;
  everything else the caller passed is dropped. Omit it to accept all. Needed
  for derivatives written once per session or subject, which carry no
  run-level entities.
- **`cardinality`** — required on every query; there is no default. Absence is
  never an error, since precomputed derivatives are whatever exists, but
  ambiguity always is:
  - `single` — scalar path; key omitted when absent, raises on 2+ matches.
  - `list` — always a list, possibly empty; never raises.
  - `pair` — sorted 2-list, or `None` below 2 matches, raises on 3+.
  - `ordered` — ordered by the `labels` field, raises if a label matches more
    than once.

  `list` and `pair` results are natural-sorted by path — callers such as
  `nipost.reconstruct_fieldmap` depend on that ordering, since it reads
  `coefficients[-1]` as the finest B-spline level.

- **Entity values** — `null` means the entity must be **absent**; omit the key
  to leave it unconstrained. `"{fieldmap_id}"` and `"{space}"` are substituted
  from the corresponding argument, in a scalar value or inside a value list.
- **Misspelled keys are errors.** A key the schema does not define — `scop`
  for `scope`, `item` for `items` — fails to load rather than leaving the real
  field at its default.

The bundled `func` spec collects `hmc_boldref`, `run_boldref`,
`session_boldref`, `subject_boldref`, and the transforms `hmc`, `run2anat`,
`run2fmap`, `run2session`, `run2subject`, `session2anat`, `subject2anat`.
Absent items simply omit their key; `run2fmap` (cardinality `list`) is the
exception — it is always present, but may be `[]` on a dataset with no
fieldmap.

## Python version support

nipost supports Python ≥ 3.12.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
