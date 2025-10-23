try:
    from ._version import version as __version__  # written by setuptools-scm at build time
except Exception:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # for very old Pythons (not expected)
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    try:
        __version__ = version("fvcomersem-viz")
    except PackageNotFoundError:
        __version__ = "0+unknown"

