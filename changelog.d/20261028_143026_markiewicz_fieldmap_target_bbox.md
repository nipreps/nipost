### Fixed

- The bounding box of the resampling target is used to determine the fieldmap
  reconstruction region. When the target is larger than the fieldmap field of view,
  the attenuation of the field to zero will now be smooth in all cases.
  If the target is smaller, some unnecessary computation may be avoided.
