from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, x86_64_only, arm64_only
from pixie.targets import x86_64, arm64
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

llvm_foo_i64_i64 = """
    define void @"_Z3fooPlS_"(i64* %".1", i64* %".2", i64* %".out")
    {
    entry:
        %.3 = load i64, i64 * %.1
        %.4 = load i64, i64 * %.2
        %"res" = add nsw i64 %".4", %".3"
        store i64 %"res", i64* %".out"
        ret void
    }
    """


class TestSpecialize(PixieTestCase):
    """Test that specialization works.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        tus = []
        tus.append(TranslationUnit("llvm_foo_double_double",
                                   llvm_foo_double_double))
        tus.append(TranslationUnit("llvm_foo_i64_i64",
                                   llvm_foo_i64_i64))

        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPlS_',
                                 signature='void(i64*, i64*, i64*)',)
        export_config.add_symbol(python_name='foo',
                                 symbol_name='_Z3fooPdS_',
                                 signature='void(double*, double*, double*)',)

        target_descr = cls.default_test_config()
        bcpu = target_descr.baseline_target.cpu
        bfeat = target_descr.baseline_target.features
        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu=bcpu,
                               baseline_features=bfeat,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        cls._export_config = export_config

        libfoo.compile()

    def _check_specialize(self, baseline_cpu, baseline_features,
                          target_feature):
        foo_ref = None
        with self.load_pixie_module('foo_library') as foo_library:
            # use to hold a reference to the original loaded module
            foo_ref = foo_library
            # cursory check that foo_library is ok
            assert foo_library.__PIXIE__['is_specialized'] is False
            # now specialize
            foo_library.__PIXIE__['specialize'](
                baseline_cpu=baseline_cpu,
                baseline_features=baseline_features,
                targets_features=(target_feature,),
            )

        # Reload
        with self.load_pixie_module('foo_library') as foo_library_specialized:
            # check that it's the specialized version
            assert foo_library_specialized.__PIXIE__['is_specialized'] is True

            # check the original and the reloaded are "equal"
            self.assertEqualPIXIE(foo_ref, foo_library_specialized,
                                  strict=False)

            # TODO: check the machine code is about right for the given chips

    @x86_64_only
    def test_specialize_x86_64(self):
        self._check_specialize('nocona',
                               x86_64.features.sse3,
                               x86_64.features.avx512f)

    @arm64_only
    def test_specialize_arm64(self):
        self._check_specialize('apple_m1',
                               arm64.features.v8_4a,
                               arm64.features.v8_6a)


if __name__ == '__main__':
    unittest.main()
