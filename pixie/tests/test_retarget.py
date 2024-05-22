from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, x86_64_only
import llvmlite.binding as llvm
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


class TestInstructionSetRetarget(PixieTestCase):
    """Tests that a PIXIE library can be used to retarget to another ISA"""

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

    @x86_64_only
    def test_retarget_to_skylake(self):

        with self.load_pixie_module('foo_library') as foo_library:

            out = ctypes.c_double(0)
            foo_data = foo_library.__PIXIE__['symbols']['foo']
            foo_sym = foo_data['void(double*, double*, double*)']
            cfunc = foo_sym['cfunc']
            cfunc(ctypes.byref(ctypes.c_double(20.)),
                  ctypes.byref(ctypes.c_double(10.)), ctypes.byref(out))
            assert out.value == 30.

            # NOTE: This function is derived from:
            # https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/codegen.py

            def recompile(bc, chip=''):

                def create_execution_engine():
                    target = llvm.Target.from_default_triple()
                    if chip:
                        target_machine = target.create_target_machine(chip)
                    else:
                        target_machine = target.create_target_machine()
                    backing_mod = llvm.parse_assembly("")
                    engine = llvm.create_mcjit_compiler(backing_mod,
                                                        target_machine)
                    return engine, target_machine

                def compile_ir(engine, bitc):
                    mod = llvm.parse_bitcode(bitc)
                    mod.verify()
                    engine.add_module(mod)
                    engine.finalize_object()
                    engine.run_static_constructors()
                    return mod

                ee, tm = create_execution_engine()
                mod = compile_ir(ee, bc)
                asm = tm.emit_assembly(mod)
                return ee, asm

            # Run the recompiled function via ctypes for this arch
            ee, _ = recompile(foo_library.__PIXIE__['bitcode'])
            recompiled_fptr = ee.get_function_address(foo_sym['symbol'])
            recompiled_cfunc = foo_sym['ctypes_cfunctype'](recompiled_fptr)
            outptr = ctypes.c_double(0.0)
            recompiled_cfunc(ctypes.byref(ctypes.c_double(1.0)),
                             ctypes.byref(ctypes.c_double(3.5)),
                             ctypes.byref(outptr))
            assert outptr.value == 4.5

            # Recompile for skylake and check instruction in asm
            _, asm = recompile(foo_library.__PIXIE__['bitcode'],
                               chip='skylake-avx512')
            assert 'vaddsd' in asm


if __name__ == '__main__':
    unittest.main()
