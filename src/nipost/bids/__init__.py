"""BIDS derivative discovery (requires the ``nipost[bids]`` extra)."""

try:
    import bids  # noqa: F401
    import niworkflows  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nipost.bids requires the 'bids' extra. Install with: pip install 'nipost[bids]'"
    ) from exc

from nipost.bids.collect import collect_derivatives, collect_fieldmaps

__all__ = ['collect_derivatives', 'collect_fieldmaps']
