def test_package_exposes_version() -> None:
    import tafr_ids

    assert isinstance(tafr_ids.__version__, str)
    assert tafr_ids.__version__
