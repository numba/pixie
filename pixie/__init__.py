from importlib.metadata import version, PackageNotFoundError
from pixie.compiler import PIXIECompiler, TranslationUnit, ExportConfiguration

__all__ = (PIXIECompiler, TranslationUnit, ExportConfiguration)

try:
    __version__ = version("numba-pixie")
except PackageNotFoundError:
    try:
        from setuptools_scm import get_version
        __version__ = get_version()
    except ImportError:
        raise NotImplementedError("unable to determine version at runtime")
