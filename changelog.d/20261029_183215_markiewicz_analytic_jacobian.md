### Changed

- `pe_vector()` is new: the phase-encoding direction in fieldmap space.
- `reconstruct_fieldmap(..., pe_vector=...)` returns `(fieldmap, jacobian)`, with the
  Jacobian computed from the analytic spline derivative.
- `resample_image(jacobian=...)` takes that image and defaults to no modulation.
  Negative values (fold-over) are clipped to 0.
  `jacobian=True` is no longer supported.
