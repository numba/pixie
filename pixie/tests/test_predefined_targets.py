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


class TestIsaDispatchWithPredefinedTargets(PixieTestCase):
    """Tests that a PIXIE library will do ISA based dispatch when compiled
       against predefined target strings."""

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        tus = (TranslationUnit("llvm_foo_double_double",
                               llvm_foo_double_double),)

        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPdS_',
                                 signature='void(double*, double*, double*)',)

        cls._dpts = dpts = cls.default_predefined_target_strings()
        target_descr = cls.default_test_config()
        cls._target_descr = target_descr

        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu=dpts.baseline_target,
                               baseline_features=dpts.baseline_target,
                               targets_features=dpts.additional_targets,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        libfoo.compile()

        # compile another copy under a different name, this time to check the
        # "default_configuration", use the previous target_descr to access the
        # default_configuration for the current arch
        default_config = target_descr.arch.default_configuration
        libbar = PIXIECompiler(library_name='bar_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               python_cext=True,
                               output_dir=cls.tmpdir.name,
                               **default_config)
        libbar.compile()

    def test_compilation_on_predefined_target_strings(self):

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

            predef = getattr(self._target_descr.arch, "predefined", None)

            targets_features = set()
            for target in ((self._dpts.baseline_target,)
                           + self._dpts.additional_targets):
                feat = max(getattr(predef, target).features)
                targets_features.add(feat)

            cpu_features = self.get_process_cpu_features()
            got = max(set(cpu_features) & set(targets_features))
            assert selected_isa == str(got)

    def test_compilation_on_default_configuration(self):

        with self.load_pixie_module('bar_library') as bar_library:
            out = ctypes.c_double(0)
            foo_data = bar_library.__PIXIE__['symbols']['foo']
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

            selected_isa = bar_library.__PIXIE__['selected_isa']

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
