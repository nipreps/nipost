### Fixed

- Cubic spline reconstruction outside the fieldmap field of view.
  Edge weights were extrapolated from the basis polynomial, rather
  than tapering across the spline support.
  Instead of falling to zero, the reconstructed field would flip
  sign and grow in magnitude.
