from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qstatic-diag")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
