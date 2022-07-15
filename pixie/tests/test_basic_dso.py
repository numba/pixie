from pixie import PIXIECompiler
from pixie.compiler import pixie_info
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

_double_double_entry = dict(python_name='foo',
                            symbol_name='_Z3fooPdS_',
                            signature='void(double*, double*, double*)',
                            llvm_ir=llvm_foo_double_double)

_i64_i64_entry = dict(python_name='foo',
                    symbol_name='_Z3fooPlS_',
                    signature='void(i64*, i64*, i64*)',
                    llvm_ir=llvm_foo_i64_i64)


class TestBasicDSO(PixieTestCase):
    """Test basic DSO things for PIXIE libraries, like symbols existing and the
    module namespace layout.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        libfoo = PIXIECompiler('foo_library', output_dir=cls.tmpdir.name)

        libfoo.add_function(**_double_double_entry)
        libfoo.add_function(**_i64_i64_entry)

        libfoo.compile_ext()
        cls.pixie_lib_decl = libfoo

    @unittest.skipUnless(_HAS_LIEF, "Test requires py-lief package")
    def test_import_and_symbols(self):
        with self.load_pixie_module('foo_library') as foo_library:
            binary = lief.parse(foo_library.__file__)
            symbols = set([x.name for x in binary.exported_symbols])
            exports = set(self.pixie_lib_decl._exported_functions.keys())
            assert exports.issubset(symbols)

    def test_module_layout(self):
        with self.load_pixie_module('foo_library') as foo_library:
            fn_namespace = foo_library.foo
            assert len(fn_namespace) == 2 # 2 entries
            for i in range(len(fn_namespace)):
                info = fn_namespace[i]
                assert isinstance(info, pixie_info)

            for idx, input_entry in enumerate((_double_double_entry,
                                            _i64_i64_entry)):
                pixie_entry = fn_namespace[idx]
                assert str(pixie_entry.symbol_name) == input_entry['symbol_name']
                assert str(pixie_entry.signature) == input_entry['signature']
                mod =  llvm.parse_bitcode(pixie_entry.bitcode)
                mod.verify()
                func_names = [x.name for x in mod.functions]
                assert len(func_names) == 1
                assert func_names[0] == input_entry['symbol_name']


if __name__ == '__main__':
    unittest.main()
