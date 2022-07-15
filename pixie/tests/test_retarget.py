from pixie import PIXIECompiler
from pixie.tests.support import PixieTestCase
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
    """Tests that a PIXIE library can be used to retarget to another instruction
    set"""

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        libfoo = PIXIECompiler('foo_library', output_dir=cls.tmpdir.name)

        libfoo.add_function(python_name='foo',
                            symbol_name='_Z3fooPdS_',
                            signature='void(double*, double*, double*)',
                            llvm_ir=llvm_foo_double_double)

        libfoo.compile_ext()

    def test_retarget_to_skylake(self):

        with self.load_pixie_module('foo_library') as foo_library:

            out = ctypes.c_double(0)
            foo_library.foo[0].ctypes_wrapper(ctypes.byref(ctypes.c_double(20.)),
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
                    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
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

            # Run the function via ctypes for this arch
            f0 = foo_library.foo[0]
            ee, _ = recompile(f0.bitcode)
            recompiled_fptr = ee.get_function_address(f0.symbol_name)
            sigct = f0.signature.as_ctypes()
            cfunc = ctypes.CFUNCTYPE(sigct.return_type, *sigct.argument_types)(recompiled_fptr)
            outptr = ctypes.c_double(0.0)
            res = cfunc(ctypes.byref(ctypes.c_double(1.0)), ctypes.byref(ctypes.c_double(3.5)), ctypes.byref(outptr))
            assert outptr.value == 4.5

            # Recompile for skylake and check instruction in asm
            _, asm = recompile(f0.bitcode, chip='skylake-avx512')
            assert 'vaddsd' in asm


if __name__ == '__main__':
    unittest.main()
