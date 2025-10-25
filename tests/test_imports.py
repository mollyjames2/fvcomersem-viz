def test_import():
    import fvcomersemviz as m

    assert hasattr(m, "__version__")
