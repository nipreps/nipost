"""Utilities for loading transforms for resampling"""

from collections.abc import Iterator
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
    The result keeps the reference space of the original.
    """
    # Read _reference, as reading an unset reference warns
    reference = xfm._reference
    collapsed: list[nt.base.TransformBase] = []
    for leaf in _flatten(xfm):
        if type(leaf) is nt.base.TransformBase:  # Identity
            continue
        if type(leaf) is nt.Affine and collapsed and type(collapsed[-1]) is nt.Affine:
            first = collapsed[-1]
            # Read _reference, as reading an unset reference warns
            collapsed[-1] = nt.Affine(leaf.matrix @ first.matrix, reference=first._reference)
        else:
            collapsed.append(leaf)

    if not collapsed:
        return nt.Affine(reference=reference)
    if len(collapsed) == 1:
        [only] = collapsed
        if reference is None or (only._reference is not None and only._reference == reference):
            return only
        if type(only) is nt.Affine:
            return nt.Affine(only.matrix, reference=reference)

    # Set the reference on the new chain, rather than on the caller's transforms
    chain = nt.TransformChain(collapsed)
    if reference is not None:
        chain.reference = reference
    return chain


def _flatten(xfm: nt.base.TransformBase) -> Iterator[nt.base.TransformBase]:
    """Yield the transforms of arbitrarily nested chains, in the order applied"""
    if isinstance(xfm, nt.TransformChain):
        for x in xfm.transforms:
            yield from _flatten(x)
    else:
        yield xfm
