from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, needs_subprocess
from pixie.cpus import x86
from pixie.mcext import c
import ctypes
import os
import unittest


llvm_foo_double_double = """
    define void @"_Z3fooPdS_"(double* %".1", double* %".2", double* %".out")
    {
    entry:
        %.3 = load double, double * %.1
        %.4 = load double, double * %.2
        %"res" = fadd double %".3", %".4"
        store double %"res", double* %".out"
        ret void
    }
    """


class TestIsaEnvVarDispatch(PixieTestCase):
    """Tests that a PIXIE library will do ISA based dispatch using the env
    var override"""

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        tus = (TranslationUnit("llvm_foo_double_double",
                               llvm_foo_double_double),)

        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPdS_',
                                 signature='void(double*, double*, double*)',)

        cls._targets_features = (x86.sse3, x86.avx)

        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu='nocona',
                               baseline_features=x86.sse2,
                               targets_features=cls._targets_features,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        cls._export_config = export_config

        libfoo.compile()

    @PixieTestCase.run_test_in_subprocess(envvars={"PIXIE_USE_ISA": "SSE2"})
    def test_envar_dispatch_valid(self):
        # Checks that PIXIE will dispatch so a given ISA env var, it's highly
        # unlikely to find a machine that just supports SSE2 and so this is
        # used as the test value.
        with self.load_pixie_module('foo_library') as foo_library:
            out = ctypes.c_double(0)
            foo_data = foo_library.__PIXIE__['symbols']['foo']
            foo_sym = foo_data['void(double*, double*, double*)']
            cfunc = foo_sym['cfunc']

            # first run sets up the fnptr cache
            cfunc(ctypes.byref(ctypes.c_double(20.)),
                  ctypes.byref(ctypes.c_double(10.)), ctypes.byref(out))
            assert out.value == 30.

            # second run uses the fnptr cache
            cfunc(ctypes.byref(ctypes.c_double(50.)),
                  ctypes.byref(ctypes.c_double(60.)), ctypes.byref(out))
            assert out.value == 110.

            selected_isa = foo_library.__PIXIE__['selected_isa']

            assert selected_isa == "SSE2"

    @needs_subprocess
    def test_impl_envar_dispatch_invalid(self):
        # Checks that PIXIE will dispatch so a given ISA env var, it's highly
        # unlikely to find a machine that just supports SSE2 and so this is
        # used as the test value.
        with self.load_pixie_module('foo_library') as foo_library:
            out = ctypes.c_double(0)
            foo_data = foo_library.__PIXIE__['symbols']['foo']
            foo_sym = foo_data['void(double*, double*, double*)']
            cfunc = foo_sym['cfunc']

            # first run sets up the fnptr cache
            cfunc(ctypes.byref(ctypes.c_double(20.)),
                  ctypes.byref(ctypes.c_double(10.)), ctypes.byref(out))

    def test_envvar_dispatch_invalid(self):
        themod = self.__module__
        thecls = type(self).__name__
        parts = (themod, thecls, "test_impl_envar_dispatch_invalid")
        fully_qualified_test = '.'.join(x for x in parts if x is not None)
        env = os.environ.copy()
        bad_isa = "NONSENSE"
        env["PIXIE_USE_ISA"] = bad_isa
        env["SUBPROC_TEST"] = "1"

        with self.assertRaises(AssertionError) as raises:
            self.run_in_subprocess(fully_qualified_test,
                                   flags=['-m', 'unittest'],
                                   env=env)
        out, err, retcode = raises.exception.args[1:]
        assert retcode == c.sysexits.EX_SOFTWARE.constant
        self.assertIn(f"No matching library is available for ISA \"{bad_isa}\"",
                      out.decode())
        assert err == b""


if __name__ == '__main__':
    unittest.main()
