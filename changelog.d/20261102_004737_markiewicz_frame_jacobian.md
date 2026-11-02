### Added

- `reconstruct_gradient` evaluates the fieldmap's gradient at the voxels of a target space.

### Changed

- `resample_image(fieldmap_gradient=...)` replaces `jacobian=`. Each frame is multiplied by
  its own Jacobian, following head motion and its `pe_info`, with fold-over clipped to 0.
  Without a gradient, no intensity modulation is applied.
