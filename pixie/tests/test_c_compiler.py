from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.cpus import x86
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
                cmd = ('clang', '-x', 'c++', '-fPIC', '-mcmodel=small',
                       '-emit-llvm', fname, '-o', outfile, '-c')
                subprocess.run(cmd)
                with open(outfile, 'rb') as f:
                    data = f.read()
            return TranslationUnit(fname, data)

        # C language source for a function f = cos(x) + 1 and its derivative
        # dfdx = -sin(x)

        src1 = """
        #include<math.h>

        void __attribute__((always_inline)) f(double* x, double* result){
            *result = cos(*x) + 1;
        }
        """

        src2 = """
        #include<math.h>

        void __attribute__((always_inline)) dfdx(double* x, double* result){
            *result = -sin(*x);
        }
        """

        tus = []
        for source in (src1, src2):
            with tempfile.NamedTemporaryFile('wt') as ntf:
                ntf.write(source)
                ntf.flush()
                tus.append(tu_from_c_source(ntf.name))

        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='f',
                                 symbol_name='_Z1fPdS_',
                                 signature='void(double*, double*)',)
        export_config.add_symbol(python_name='dfdx',
                                 symbol_name='_Z4dfdxPdS_',
                                 signature='void(double*, double*)',)
        compiler = PIXIECompiler(library_name='objective_functions',
                                 translation_units=tus,
                                 export_configuration=export_config,
                                 baseline_cpu='nocona',
                                 baseline_features=x86.sse3,
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


if __name__ == '__main__':
    unittest.main()
