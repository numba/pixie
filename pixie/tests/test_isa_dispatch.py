from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase
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

        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPdS_',
                                 signature='void(double*, double*, double*)',)

        target_descr = cls.default_test_config()
        cls._target_descr = target_descr

        bfeat = target_descr.baseline_target.features
        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu=target_descr.baseline_target.cpu,
                               baseline_features=bfeat,
                               targets_features=target_descr.additional_targets,
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

            selected_isa = foo_library.__PIXIE__['selected_isa']
            target_features = set()
            cfg = getattr(self._target_descr.arch, 'default_configuration')
            target_features.add(max(cfg['baseline_features']))
            [target_features.add(max(d.features)) for d in
             cfg['targets_features']]
            cpu_features = self.get_process_cpu_features()
            got = max(set(cpu_features) & set(target_features))
            assert selected_isa == str(got)


if __name__ == '__main__':
    unittest.main()
