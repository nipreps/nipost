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
    deriv_root,
    spec=load_spec('anat'),
    entities={'subject': subject},
    params={'space': ['MNI152NLin2009cAsym']},
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
    coeffs = [nb.load(path) for path in fmaps['fieldmaps'][fmapid]['coeffs']]
    fmapref = nb.load(fmaps['fieldmaps'][fmapid]['magnitude'])
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

| Symbol                             | Description                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `nipost.bids.collect_derivatives`  | Spec-driven discovery of fMRIPrep derivatives (images, transforms).                        |
| `nipost.bids.collect_fieldmaps`    | Collect B-Spline fieldmap derivatives, grouped by fieldmap ID under the spec's group name. |
| `nipost.bids.spec.load_spec`       | Load a bundled spec (`"anat"` / `"func"` / `"fmap"`) or a spec file.                       |
| `nipost.bids.sanitize_space`       | Convert TemplateFlow cohort syntax to a BIDS entity value.                                 |
| `nipost.bids.sanitize_fieldmap_id` | Convert an fMRIPrep fieldmap ID to a BIDS `fmapid` value.                                  |
| `nipost.bids.spec.Spec`            | A spec: a mapping of group names to `Group`s. Alias for `dict[str, Group]`.                |
| `nipost.bids.spec.Group`           | A named group of queries, optionally indexed by a parameter via `over`.                    |
| `nipost.bids.spec.Query`           | A single named lookup: entity alternatives, result shape, and caller-entity scope.         |
| `nipost.bids.spec.Ordered`         | The `multi` variant declaring an ordering axis: `Ordered(order='label')`.                  |

The four schema types are what you construct when building a spec in Python
rather than loading YAML. The `multi`/`order` rules are enforced either way; the
non-emptiness of `entities` and `queries` is a decode-time constraint, so an
empty one built in Python is accepted and simply collects nothing:

```python
from nipost.bids.spec import Group, Ordered, Query

spec = {
    'anat': Group(
        queries={
            'tpms': Query(
                entities=[{'suffix': 'probseg', 'label': ['GM', 'WM', 'CSF']}],
                multi=Ordered(order='label'),
            )
        }
    )
}
```

They live in `nipost.bids.spec`, not `nipost.bids`, which stays the calling
surface.

### Spec schema

A spec is a mapping of group names you choose to groups of named queries. A
group may declare `over: <param>`, which runs its queries once per value of
that parameter and nests the results under each value. The bundled specs are
YAML; `load_spec` reads any path, and a JSON spec file works too, JSON being
valid YAML.

```yaml
# illustrative: one plain group and one indexed group
example_images:
  queries:
    session:
      entities:
        - space: session
          suffix: boldref
      scope: [subject, session]
    tpms:
      entities:
        - suffix: probseg
          label: [GM, WM, CSF]
      multi: { order: label }
transforms:
  over: space
  queries:
    forward:
      entities:
        - from: T1w
          to: "{space}"
          suffix: xfm
```

- **`entities`** — ordered entity dicts describing one logical item under
  different naming schemes. The first alternative that matches anything is
  used; the shape applies to that alternative alone, never to a union. Put
  current naming first.
- **`multi`** — the result's shape, in one field. Omit it for a scalar path:
  a match raises on two or more, and no match omits the key. `multi: true`
  gives an unordered list, possibly empty, natural-sorted by path — which
  `nipost.reconstruct_fieldmap` depends on, since it reads `coefficients[-1]`
  as the finest B-spline level. `multi: {order: <entity>}` gives one path per
  value that entity is declared with, in that order; two files sharing a value
  is an error, and a declared value with no match omits the key, since an
  incomplete tuple is the absence of the item rather than a partial one.
  The ordering axis may not contain a parameter reference: it is matched against
  entity values read out of filenames, which never contain one, so the query
  could never return a complete result. `load_spec` rejects it.
  Axis values are coerced to the type PyBIDS parsed the entity as, so a `run`
  axis may be declared `[1, 2]` or `['01', '02']` interchangeably.
- **`scope`** — allowlist of caller-supplied entity names the query accepts;
  everything else the caller passed is dropped. Omit it to accept all. Needed
  for derivatives written once per session or subject, which carry no run-level
  entities.
- **Entity values** — `null` means the entity must be **absent**; omit the key
  to leave it unconstrained. A `{name}`-shaped string is a parameter reference:
  bound parameters substitute, unbound ones drop the constraint.
  A placeholder is rejected outright in the value list of an ordering entity
  (see `multi`, above). A parameter bound this way — inside a query's
  `entities`, rather than by an enclosing `over` — must be a scalar; only a
  group's `over` parameter may be a
  sequence, since that is the one place a sequence has a defined meaning
  (one iteration per value). Binding a sequence anywhere else raises
  `TypeError` naming the parameter.
- **Misspelled keys are errors.** A key `Group` or `Query` does not define —
  `scop` for `scope`, `querys` for `queries` — fails to load. Group _names_ are
  open, since you choose them, so a misspelled group name surfaces as a
  `KeyError` when you index the result.

Parameter values are used verbatim, for matching and for output keys alike.
Apply `nipost.bids.sanitize_space` to a TemplateFlow cohort string
(`MNI152NLin6Asym:cohort-1` → `MNI152NLin6Asym+1`) or
`nipost.bids.sanitize_fieldmap_id` to an fMRIPrep fieldmap ID before passing
them.

The bundled `func` spec has a `boldrefs` group (`hmc`, `run`, `session`,
`subject`) and a `transforms` group (`hmc`, `run2anat`, `run2fmap`,
`run2session`, `run2subject`, `session2anat`, `subject2anat`). Absent items
omit their key; `run2fmap` (`multi: true`) is the exception — it is always
present, but may be `[]` on a dataset with no fieldmap.

## Python version support

nipost supports Python ≥ 3.12.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
