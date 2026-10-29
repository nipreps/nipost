"""Utilities for loading transforms for resampling"""

from pathlib import Path

import nitransforms as nt


def load_transforms(xfm_paths: list[Path], inverse: list[bool]) -> nt.base.TransformBase:
    """Load a series of transforms as a nitransforms TransformChain

    An empty list will return an identity transform
    """
    if len(inverse) == 1:
        inverse *= len(xfm_paths)
    elif len(inverse) != len(xfm_paths):
        raise ValueError('Mismatched number of transforms and inverses')

    chain = None
    for path, inv in zip(xfm_paths[::-1], inverse[::-1], strict=False):
        path = Path(path)
        if path.suffix == '.h5':
            # Load as a TransformChain
            xfm = nt.manip.load(path, fmt='h5')
        else:
            xfm = nt.linear.load(path)
        if inv:
            xfm = ~xfm
        if chain is None:
            chain = xfm
        else:
            chain += xfm
    if chain is None:
        chain = nt.Affine()  # Identity
    return chain


def collapse_transforms(xfm: nt.base.TransformBase) -> nt.base.TransformBase:
    """Reduce a transform to the fewest transforms that map points the same way

    Nested chains are flattened, identities dropped, and each run of consecutive
    affines merged into one, so a chain of affines becomes a single Affine, and
    a warp between affines becomes an ``[Affine, warp, Affine]`` chain.
    Per-volume affines (``LinearTransformsMapping``) are left as they are.
    The first transform keeps the reference space of the original.
    """
    raise NotImplementedError
