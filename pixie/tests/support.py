from unittest import TestCase, skipUnless
from llvmlite import binding as llvm
import tempfile
import contextlib
import os
import sys
import subprocess
import unittest
from functools import lru_cache
import types as pytypes

# NOTE: This is copied from:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/tests/support.py#L754  # noqa: E501


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

    # From:
    # https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/tests/support.py#L559-L606
    def subprocess_test_runner(self, test_module, test_class=None,
                               test_name=None, envvars=None, timeout=60,
                               _subproc_test_env="1"):
        """
        Runs named unit test(s) as specified in the arguments as:
        test_module.test_class.test_name. test_module must always be supplied
        and if no further refinement is made with test_class and test_name then
        all tests in the module will be run. The tests will be run in a
        subprocess with environment variables specified in `envvars`.
        If given, envvars must be a map of form:
            environment variable name (str) -> value (str)
        It is most convenient to use this method in conjunction with
        @needs_subprocess as the decorator will cause the decorated test to be
        skipped unless the `SUBPROC_TEST` environment variable is set to
        the same value of ``_subproc_test_env``
        (this special environment variable is set by this method such that the
        specified test(s) will not be skipped in the subprocess).


        Following execution in the subprocess this method will check the test(s)
        executed without error. The timeout kwarg can be used to allow more time
        for longer running tests, it defaults to 60 seconds.
        """
        parts = (test_module, test_class, test_name)
        fully_qualified_test = '.'.join(x for x in parts if x is not None)
        cmd = [sys.executable, '-m', 'unittest', fully_qualified_test]
        env_copy = os.environ.copy()
        env_copy['SUBPROC_TEST'] = _subproc_test_env
        try:
            env_copy['COVERAGE_PROCESS_START'] = os.environ['COVERAGE_RCFILE']
        except KeyError:
            pass   # ignored
        envvars = pytypes.MappingProxyType({} if envvars is None else envvars)
        env_copy.update(envvars)
        status = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout,
                                env=env_copy, universal_newlines=True)
        streams = (f'\ncaptured stdout: {status.stdout}\n'
                   f'captured stderr: {status.stderr}')
        self.assertEqual(status.returncode, 0, streams)
        # Python 3.12.1 report
        no_tests_ran = "NO TESTS RAN"
        if no_tests_ran in status.stderr:
            self.skipTest(no_tests_ran)
        else:
            self.assertIn('OK', status.stderr)

    # From:
    # https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/tests/support.py#L608-L635
    def run_test_in_subprocess(maybefunc=None, timeout=60, envvars=None):
        """Runs the decorated test in a subprocess via invoking numba's test
        runner. kwargs timeout and envvars are passed through to
        subprocess_test_runner."""
        def wrapper(func):
            def inner(self, *args, **kwargs):
                if os.environ.get("SUBPROC_TEST", None) != func.__name__:
                    # Not in a subprocess test env, so stage the call to run the
                    # test in a subprocess which will set the env var.
                    class_name = self.__class__.__name__
                    self.subprocess_test_runner(
                        test_module=self.__module__,
                        test_class=class_name,
                        test_name=func.__name__,
                        timeout=timeout,
                        envvars=envvars,
                        _subproc_test_env=func.__name__,
                    )
                else:
                    # env var is set, so we're in the subprocess, run the
                    # actual test.
                    func(self)
            return inner

        if isinstance(maybefunc, pytypes.FunctionType):
            return wrapper(maybefunc)
        else:
            return wrapper

        # From:
        # https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/tests/support.py#1090-L1106
    def run_in_subprocess(self, code, flags=None, env=None, timeout=30):
        """Run a snippet of Python code in a subprocess with flags, if any are
        given. 'env' is passed to subprocess.Popen(). 'timeout' is passed to
        popen.communicate().

        Returns the stdout and stderr of the subprocess after its termination.
        """
        if flags is None:
            flags = []
        cmd = [sys.executable,] + flags + ["-c", code]
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env)
        out, err = popen.communicate(timeout=timeout)
        if popen.returncode != 0:
            msg = "process failed with code %s: stderr follows\n%s\n"
            raise AssertionError(msg % (popen.returncode, err.decode()),
                                 out, err, popen.returncode)
        return out, err, popen.returncode


@lru_cache
def _has_clang():
    cmd = ('clang', '--help')
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10.,
                                check=True)
        return bool(not result.returncode)
    except FileNotFoundError:
        return False


needs_clang = skipUnless(_has_clang(), "Test needs clang")


# From:
# https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/tests/support.py#195-L200
# Decorate a test with @needs_subprocess to ensure it doesn't run unless the
# `SUBPROC_TEST` environment variable is set. Use this in conjunction with:
# TestCase::subprocess_test_runner which will execute a given test in subprocess
# with this environment variable set.
_exec_cond = os.environ.get('SUBPROC_TEST', None) == '1'
needs_subprocess = unittest.skipUnless(_exec_cond, "needs subprocess harness")
