import ctypes
import numpy as np
import re
import tempfile
import unittest

from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, needs_clang
from llvmlite import binding as llvm


@needs_clang
class TestCCompiler(PixieTestCase):
    """Tests a basic clang wrapper to compile some C code to a PIXIE library.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

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
        for source, debug in ((src1, True), (src2, False), (src3, True)):
            with tempfile.NamedTemporaryFile('wt') as ntf:
                ntf.write(source)
                ntf.flush()
                if debug:
                    clang_flags = ("-g",)
                else:
                    clang_flags = ()
                tus.append(TranslationUnit.from_c_source(
                    ntf.name, extra_flags=clang_flags))

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

    def test_check_debuginfo(self):
        with self.load_pixie_module('objective_functions') as foo_library:
            # check bitcode for debug info
            mod = llvm.parse_bitcode(foo_library.__PIXIE__['bitcode'])
            llvm_ir = str(mod)
            # expect 2 DISubprograms, one for "f" and one for "bar", but not
            # one for dfdx.
            functions = set()
            matcher = re.compile(r".*name:\s+\"(.*)\",.*")
            for line in llvm_ir.splitlines():
                if "!DISubprogram" in line:
                    functions.add(matcher.match(line).groups()[0])

            assert functions == {"f", "bar"}


if __name__ == '__main__':
    unittest.main()
