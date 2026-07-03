import nipost


def test_public_symbols_exposed():
    for name in (
        'resample_image',
        'reconstruct_fieldmap',
        'load_transforms',
        'get_trt',
        'ensure_positive_cosines',
    ):
        assert hasattr(nipost, name), name
