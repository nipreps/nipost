"""BIDS derivatives discovery (requires the ``nipost[bids]`` extra)."""

try:
    from nipost.bids.collect import collect_derivatives, collect_fieldmaps
    from nipost.bids.spec import load_spec, sanitize_fieldmap_id, sanitize_space
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nipost.bids requires the 'bids' extra. Install with: pip install 'nipost[bids]'"
    ) from exc

__all__ = [
    'collect_derivatives',
    'collect_fieldmaps',
    'load_spec',
    'sanitize_fieldmap_id',
    'sanitize_space',
]
