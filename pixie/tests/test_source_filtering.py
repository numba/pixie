# tests that input IR/bitcode with target-cpu and target-features _does not_
# impact code generation
from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase
from llvmlite import binding as llvm
import uuid
import unittest


llvm_foo_double_double = """
    define void @"_Z3fooPdS_"(double* %".1", double* %".2", double* %".out") #0
    {
    entry:
        %.3 = load double, double * %.1
        %.4 = load double, double * %.2
        %"res" = fadd double %".3", %".4"
        store double %"res", double* %".out"
        ret void
    }

    attributes #0 = { noinline nounwind "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "tune-cpu"="generic" "frame-pointer"="all"}
    """  # noqa: E501

llvm_foo_i64_i64 = """
    define void @"_Z3fooPlS_"(i64* %".1", i64* %".2", i64* %".out") #0
    {
    entry:
        %.3 = load i64, i64 * %.1
        %.4 = load i64, i64 * %.2
        %"res" = add nsw i64 %".4", %".3"
        store i64 %"res", i64* %".out"
        ret void
    }

    attributes #0 = { noinline nounwind "target-cpu"="x86-64" "target-features"="+ssse3" "tune-cpu"="generic" "frame-pointer"="all"}
    """  # noqa: E501


class TestSourceFiltering(PixieTestCase):

    def _check(self, tus):
        library_name = str(uuid.uuid4().hex)
        export_config = ExportConfiguration()
        target_descr = self.default_test_config()
        bfeat = target_descr.baseline_target.features
        lib = PIXIECompiler(library_name=library_name,
                            translation_units=tus,
                            export_configuration=export_config,
                            baseline_cpu=target_descr.baseline_target.cpu,
                            baseline_features=bfeat,
                            python_cext=True,
                            output_dir=self.tmpdir.name)

        self._export_config = export_config
        lib.compile()

        with self.load_pixie_module(library_name) as lib:
            llvm_ir = str(llvm.parse_bitcode(lib.__PIXIE__['bitcode']))
            self.assertNotIn("target-cpu", llvm_ir)
            self.assertNotIn("target-features", llvm_ir)

    def test_multiple_ir(self):

        tus = []
        tus.append(TranslationUnit("llvm_foo_double_double",
                                   llvm_foo_double_double))
        tus.append(TranslationUnit("llvm_foo_i64_i64",
                                   llvm_foo_i64_i64))

        self._check(tus)

    def test_multiple_bitcode(self):

        tus = []
        foo_dd_bc = llvm.parse_assembly(llvm_foo_double_double).as_bitcode()
        self.assertIn("target-cpu", str(llvm.parse_bitcode(foo_dd_bc)))
        tus.append(TranslationUnit("llvm_foo_double_double", foo_dd_bc))
        foo_ii_bc = llvm.parse_assembly(llvm_foo_i64_i64).as_bitcode()
        self.assertIn("target-cpu", str(llvm.parse_bitcode(foo_ii_bc)))
        tus.append(TranslationUnit("llvm_foo_i64_i64", foo_ii_bc))

        self._check(tus)

    def test_single_ir(self):

        tus = []
        tus.append(TranslationUnit("llvm_foo_double_double",
                                   llvm_foo_double_double))

        self._check(tus)

    def test_single_bitcode(self):

        tus = []
        foo_dd_bc = llvm.parse_assembly(llvm_foo_double_double).as_bitcode()
        self.assertIn("target-cpu", str(llvm.parse_bitcode(foo_dd_bc)))
        tus.append(TranslationUnit("llvm_foo_double_double",
                                   foo_dd_bc))

        self._check(tus)


if __name__ == '__main__':
    unittest.main()
