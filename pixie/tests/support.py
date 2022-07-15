from unittest import TestCase
import lief
from llvmlite import binding as llvm
import tempfile
import contextlib
import sys

# NOTE: This is copied from:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/tests/support.py#L754

def import_dynamic(modname):
    """
    Import and return a module of the given name.  Care is taken to
    avoid issues due to Python's internal directory caching.
    """
    import importlib
    importlib.invalidate_caches()
    __import__(modname)
    return sys.modules[modname]


class PixieTestCase(TestCase):

    @classmethod
    def _init_llvm(cls):
        llvm.initialize()
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()

    @contextlib.contextmanager
    def load_pixie_module(self, name):
        """
        Loads a pixie module from the TestCase temporary directory
        """
        sys.path.append(self.tmpdir.name)
        try:
            lib = import_dynamic(name)
            yield lib
        finally:
            sys.path.remove(self.tmpdir.name)
            sys.modules.pop(name, None)

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls._init_llvm()
        TestCase.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
        TestCase.tearDownClass()
