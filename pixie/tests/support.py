from unittest import TestCase, skipUnless
from llvmlite import binding as llvm
import tempfile
import contextlib
import sys
import subprocess
from functools import lru_cache

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

    def assertEqualPIXIE(self, pixielib1, pixielib2, strict=True):
        pi1 = pixielib1.__PIXIE__
        pi2 = pixielib2.__PIXIE__

        required_identical = ('bitcode', 'c_header', 'linkage', 'uuid')
        relaxed_identical = ('is_specialized',)

        if strict:
            fields = required_identical + relaxed_identical
        else:
            fields = required_identical

        for field in fields:
            assert pi1[field] == pi2[field], f"Field '{field}' doesn't match."

        # check symbol table
        pi1sym = pi1['symbols']
        pi2sym = pi2['symbols']

        # check python symbols match
        assert pi1sym.keys() == pi2sym.keys()

        def item_checker(d1, d2):
            assert d1.keys() == d2.keys()
            fields = ('ctypes_cfunctype',
                      'symbol',
                      'module',
                      'source_file',
                      # 'address', # runtime defined
                      # 'cfunc',  # runtime defined
                      'feature_variants',
                      'baseline_feature',
                      # 'metadata', # could be anything
                      )
            for field in fields:
                assert d1[field] == d2[field], f"Field '{field}' doesn't match"

        for pysym in pi1sym.keys():
            # check that the symbol sigs match:
            assert pi1sym[pysym].keys() == pi2sym[pysym].keys()
            for sig in pi1sym[pysym].keys():
                d1 = pi1sym[pysym][sig]
                d2 = pi2sym[pysym][sig]
                item_checker(d1, d2)


@lru_cache
def _has_clang():
    cmd = ('clang', '--help')
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10., check=True)
        return bool(not result.returncode)
    except FileNotFoundError:
        return False


needs_clang = skipUnless(_has_clang(), "Test needs clang")
