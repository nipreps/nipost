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
fmaps = collect_fieldmaps(deriv_root, entities={'subject': subject})
func = collect_derivatives(
    deriv_root,
    spec=load_spec('func'),
    entities=bold_entities,
    params={'fmapid': list(fmaps['fieldmaps'])},
)
anat = collect_derivatives(
    deriv_root,
    spec=load_spec('anat'),
    entities={'subject': subject},
    params={'space': ['MNI152NLin2009cAsym']},
)

# 2. Build transform chains (HMC → boldref→anat → anat→std)
transforms = func['transforms']
anat2std_xfm = anat['transforms']['MNI152NLin2009cAsym']['forward']

# fMRIPrep's --bold-coreg-level decides whether each run is coregistered to
# the anatomical reference directly, or via a shared session/subject boldref
# template. Select whichever chain the dataset provides.
run2anat_xfms = next(
    [transforms[k] for k in xfmset]
    for xfmset in [
        ['run2anat'],
        ['run2session', 'session2anat'],
        ['run2subject', 'subject2anat'],
    ]
    if all(k in transforms for k in xfmset)
)

run2std = load_transforms(
    [transforms['hmc'], *run2anat_xfms, anat2std_xfm],
    inverse=[False],
)

# 3. Reconstruct the fieldmap (B-Spline coefficients → Hz image in target space)
# The run's fieldmap is the ID whose entry found a transform; none means SDC was skipped
fmap_std = None
fmapid, fmap_xfms = next(
    ((k, v) for k, v in func['fmap_transforms'].items() if v), (None, None)
)
if fmapid is not None:
    fmap2std = load_transforms(
        [fmap_xfms['run2fmap'], *run2anat_xfms, anat2std_xfm],
        inverse=[True, *[False] * (len(run2anat_xfms) + 1)],
    )
    # 'coeffs' is a list: one entry per B-Spline level.
    coeffs = [nb.load(path) for path in fmaps['fieldmaps'][fmapid]['coeffs']]
    fmapref = nb.load(fmaps['fieldmaps'][fmapid]['magnitude'])
    fmap_std = reconstruct_fieldmap(coeffs, fmapref, target, fmap2std)

# 4. Resample BOLD in one shot — HMC + SDC + normalization simultaneously
bold_mni = resample_image(
    source=bold,
    target=target,
    transforms=run2std,
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
| `nipost.bids.spec.Group`           | A named group of queries, optionally indexed by a parameter via `per`.                    |
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

### Spec schema

A spec is a mapping of group names you choose to groups of named queries. A
group may declare `per: <param>`, which runs its queries once per value of
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
  per: space
  queries:
    forward:
      entities:
        - from: T1w
          to: "{space}"
          suffix: xfm
```

- **`per`** — the name of a parameter the group iterates over; a group that
  declares it requires it in `params`, or a `TypeError` is raised.
  `params[per]` accepts `None` (no iterations, the group's result is `{}`),
  a scalar (one iteration, keyed by that value, including strings),
  or any other iterable (one iteration per member, keyed by the member, `None` included).
  Every requested value is a key in the result, including for queries that
  find nothing (`{}`).
- **`entities`** — a sequence of entity dicts describing one logical item under
  different naming schemes. The first alternative that matches anything is used.
  Put the current naming first.
  - **Entity values** — `null` means the entity must be **absent**; omit the key
    to leave it unconstrained.
    A plain value (string, number) requires the entity value to match exactly.
    A sequence of values requires the entity to match one of the values exactly.
    Sequences may contain `null` to match files that omit the entity.
    A `"{name}"`-shaped string is a parameter reference, substituted verbatim,
    and this may be a single value or a member of a sequence,
    except for an ordering entity (see `multi`, below).
  - **Placeholder values** — If `"{name}"` is used, then `"name"` must appear in
    the `params` dict passed to `collect_derivatives`.
    The values passed to `params` may be `None`, a single value, or,
    in the case of a group's `per` parameter, an iterable of values.
    Members of the `bids.layout.Query` class are also accepted at this time,
    except in a group's `per` parameter (where they raise `TypeError`),
    and are passed through to PyBIDS, but this behavior is not guaranteed
    and may stop operating if nipost adopts a different query engine.
    If you depend on this behavior, please open an issue to discuss your use case.
- **`multi`** — the cardinality of the query result.
  `multi: false` or absent indicates a scalar path, and more than one match
  raises an error; no match omits the key.
  `multi: true` indicates an unordered list, natural-sorted by path;
  no match omits the key (rather than giving `[]`).
  `multi: {order: <entity>}` gives one path per value that entity is declared with,
  in that order; two files sharing a value is an error,
  and a declared value with no match omits the key.
  The ordering axis may not contain a placeholder value.
  Axis values are coerced to the type PyBIDS parsed the entity as, so a `run`
  axis may be declared `[1, 2]` or `['01', '02']` interchangeably.
- **`scope`** — allowlist of caller-supplied entity names the query accepts;
  everything else the caller passed is dropped. Omit it to accept all. Needed
  for derivatives written once per session or subject, which carry no run-level
  entities.

Parameter values are used verbatim, for matching and for output keys alike.
Apply `nipost.bids.sanitize_space` to a TemplateFlow cohort string
(`MNI152NLin6Asym:cohort-1` → `MNI152NLin6Asym+1`) or
`nipost.bids.sanitize_fieldmap_id` to an fMRIPrep fieldmap ID before passing
them.

The bundled `func` spec has a `boldrefs` group (`hmc`, `run`, `session`,
`subject`), a `transforms` group (`hmc`, `run2anat`, `run2session`,
`run2subject`, `session2anat`, `subject2anat`), and a `fmap_transforms` group
(`per: fmapid`, query `run2fmap`) keyed by fieldmap ID. Absent items omit
their key. Pass the fieldmap IDs from `collect_fieldmaps`' result as
`params={'fmapid': ...}` to collect each run's fieldmap transform, or
`params={'fmapid': None}` to skip `fmap_transforms` entirely.

## Python version support

nipost supports Python ≥ 3.12.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
