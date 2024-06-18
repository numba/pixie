from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, needs_clang
import ctypes
import os
import numpy as np
import subprocess
import tempfile
import unittest


@needs_clang
class TestCCompiler(PixieTestCase):
    """Tests a basic clang wrapper to compile some C code to a PIXIE library.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        def tu_from_c_source(fname):
            prefix = 'pixie-c-build-'
            with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:
                outfile = os.path.join(build_dir, 'tmp.bc')
                cmd = ('clang', '-x', 'c', '-fPIC', '-mcmodel=small',
                       '-emit-llvm', fname, '-o', outfile, '-c')
                subprocess.run(cmd)
                with open(outfile, 'rb') as f:
                    data = f.read()
            return TranslationUnit(fname, data)

        # C language source for a function f = cos(x) + 1 and its derivative
        # dfdx = -sin(x)

        src1 = """
        #include<math.h>

        void f(double* x, double* result){
            *result = cos(*x) + 1;
        }
        """

        src2 = """
        #include<math.h>

        void dfdx(double* x, double* result){
            *result = -sin(*x);
        }
        """

        src3 = """
        #include<math.h>

        void bar(double* x, double* result){
            *result = 123.45;
        }
        """

        tus = []
        for source in (src1, src2, src3):
            with tempfile.NamedTemporaryFile('wt') as ntf:
                ntf.write(source)
                ntf.flush()
                tus.append(tu_from_c_source(ntf.name))

        export_config = ExportConfiguration()
        # NOTE: bar is not exported, this is to allow checking the exports in
        # the PIXIE dictionary vs. the library exports.
        export_config.add_symbol(python_name='f',
                                 symbol_name='f',
                                 signature='void(double*, double*)',)
        export_config.add_symbol(python_name='dfdx',
                                 symbol_name='dfdx',
                                 signature='void(double*, double*)',)

        target_descr = cls.default_test_config()

        bfeat = target_descr.baseline_target.features
        compiler = PIXIECompiler(library_name='objective_functions',
                                 translation_units=tus,
                                 export_configuration=export_config,
                                 baseline_cpu=target_descr.baseline_target.cpu,
                                 baseline_features=bfeat,
                                 python_cext=True,
                                 output_dir=cls.tmpdir.name)
        compiler.compile()

    def test_compile_and_call_c_code(self):
        with self.load_pixie_module('objective_functions') as foo_library:
            symbols = foo_library.__PIXIE__['symbols']
            f = symbols['f']['void(double*, double*)']['cfunc']
            arg = ctypes.c_double(2. * np.pi)
            out = ctypes.c_double(0)
            f(ctypes.byref(arg), ctypes.byref(out))
            assert out.value == 2.0

            dfdx = symbols['dfdx']['void(double*, double*)']['cfunc']
            arg = ctypes.c_double(-np.pi/2.)
            out = ctypes.c_double(0)
            dfdx(ctypes.byref(arg), ctypes.byref(out))
            assert out.value == 1.0

            # bar should not be exported
            assert symbols.get("bar") is None

    def test_check_function_forwarding(self):
        # checks that PIXIE added exports for the symbols that were public in
        # the embedded library to the interposing PIXIE library.
        with self.load_pixie_module('objective_functions') as foo_library:
            DSO = ctypes.CDLL(foo_library.__file__)
            # functions "f", "dfdx" and "bar" should be available.
            f_func = DSO.f
            dfdx_func = DSO.dfdx
            bar_func = DSO.bar
            for func in (f_func, dfdx_func, bar_func):
                func.argtypes = (ctypes.POINTER(ctypes.c_double),) * 2
                func.restype = None

            def call(func):
                arg = ctypes.c_double(0)
                out = ctypes.c_double(0)
                func(ctypes.byref(arg), ctypes.byref(out))
                return out.value

            assert call(f_func) == 2.0
            assert call(dfdx_func) == -0.0
            assert call(bar_func) == 123.45


if __name__ == '__main__':
    unittest.main()
