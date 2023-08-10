from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase
from pixie.cpus import x86
import ctypes
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


class TestIsaDispatch(PixieTestCase):
    """Tests that a PIXIE library will do ISA based dispatch"""

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        tus = (TranslationUnit("llvm_foo_double_double",
                               llvm_foo_double_double),)

        export_config = ExportConfiguration('embed_dso')
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPdS_',
                                 signature='void(double*, double*, double*)',)

        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu='nocona',
                               baseline_features=x86.sse2,
                               targets_features=(x86.sse3, x86.avx),
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        cls._export_config = export_config

        libfoo.compile()

    def test_dispatch(self):

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


if __name__ == '__main__':
    unittest.main()
