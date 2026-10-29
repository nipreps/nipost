### Added

- `nipost.transforms.collapse_transforms` reduces a transform to its fewest
  transforms, merging each run of affines.

### Changed

- `reconstruct_fieldmap` drops `fmap_reference` and evaluates the splines exactly at
  each target voxel, for any transform, instead of reconstructing on an intermediate
  grid and resampling.

### Fixed

- Targets with flipped voxel axes (e.g. LAS) no longer receive the field in the
  wrong voxel order.
- Target voxels outside the fieldmap's field of view no longer receive 0 Hz; the
  field tapers smoothly to zero past the coefficient support.

### Removed

- `grid_bspline_weights` and `aligned` are removed from `nipost.fieldmap`.
