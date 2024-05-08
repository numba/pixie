from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.cpus import x86
from pixie.tests.support import PixieTestCase
from llvmlite import binding as llvm
import unittest

try:
    import lief
    _HAS_LIEF = True
except ImportError:
    _HAS_LIEF = False


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


class TestBasicDSO(PixieTestCase):
    """Test basic DSO things for PIXIE libraries, like symbols existing and the
    module namespace layout.
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

        libfoo = PIXIECompiler(library_name='foo_library',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu='nocona',
                               baseline_features=x86.sse3,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        cls._export_config = export_config

        libfoo.compile()

    @unittest.skipUnless(_HAS_LIEF, "Test requires py-lief package")
    def test_import_and_symbols(self):
        with self.load_pixie_module('foo_library') as foo_library:
            binary = lief.parse(foo_library.__file__)
            symbols = set([x.name for x in binary.exported_symbols])
            pixie_symbols = foo_library.__PIXIE__['symbols']
            exports = set()
            for name_info in pixie_symbols.values():
                for details in name_info.values():
                    sym = details['symbol']
                    assert sym not in exports  # all symbols should be "new"
                    exports.add(sym)
            assert exports
            assert exports.issubset(symbols)

    def test_module_layout(self):
        with self.load_pixie_module('foo_library') as foo_library:
            fn_namespace = foo_library.__PIXIE__['symbols']
            assert len(fn_namespace) == 1  # 1 foo entry
            assert len(fn_namespace['foo']) == 2  # 2 foo symbols

            for input_entry in self._export_config._data:
                pixie_entry = fn_namespace[input_entry.python_name]
                details = pixie_entry[input_entry.signature]
                assert details['symbol'] == input_entry.symbol_name

            # check the bitcode is legal
            mod = llvm.parse_bitcode(foo_library.__PIXIE__['bitcode'])
            mod.verify()


if __name__ == '__main__':
    unittest.main()
