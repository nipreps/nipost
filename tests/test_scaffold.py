import nipost


def test_package_importable():
    assert isinstance(nipost.__version__, str)
