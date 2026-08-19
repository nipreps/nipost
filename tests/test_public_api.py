import nipost


def test_public_symbols_exposed():
    for name in (
        'resample_image',
        'reconstruct_fieldmap',
        'prepare_epi',
        'load_transforms',
        'get_trt',
        'ensure_positive_cosines',
    ):
        assert hasattr(nipost, name), name

    assert isinstance(nipost.__version__, str)
