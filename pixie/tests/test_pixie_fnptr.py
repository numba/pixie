from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.cpus import x86
from pixie.tests.support import PixieTestCase
import ctypes
import timeit
import unittest
import llvmlite.binding as llvm

llvm_function = """
    define dso_local void @_Z8functionPi(i64 * %".out") #0 {
        store i64 1, i64* %".out"
        ret void
    }
    attributes #0 = { alwaysinline norecurse nounwind}
    """

llvm_optimise = """
    define dso_local void @_Z9optimiserPFlvEPlS1_(void (i64*)* nocapture %0, i64* nocapture readonly %1, i64* nocapture %".out") #0 {
    %3 = load i64, i64* %1, align 8
    %4 = icmp eq i64 %3, 0
    br i1 %4, label %9, label %5

    5:
    %6 = load i64, i64* %1, align 8
    %7 = icmp ugt i64 %6, 1
    %8 = select i1 %7, i64 %6, i64 1
    br label %9

    9:
    %10 = phi i64 [ 0, %2 ], [ %8, %5 ]
    store i64 %10, i64* %".out"
    ret void
    }
    attributes #0 = { alwaysinline norecurse nounwind}
    """  # noqa: E501

llvm_specialize = r"""
    declare dso_local void @_Z8functionPi(i64*)
    declare dso_local void @_Z9optimiserPFlvEPlS1_(void (i64*)* nocapture %0, i64* nocapture readonly %1, i64* nocapture %".out")
    define dso_local void @_Z10specializePl(i64* %".out") local_unnamed_addr #3 {
    %1 = alloca i64, align 8
    %2 = bitcast i64* %1 to i8*
    store i64 10000000, i64* %1, align 8
    call void @_Z9optimiserPFlvEPlS1_(void (i64*)* @_Z8functionPi, i64* %1, i64* %".out")
    ret void
    }
    """  # noqa: E501


class TestMultipleFunctionsSingleModule(PixieTestCase):
    """Tests a multi-function module and how it optimises.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        optlib_tus = (TranslationUnit("llvm_optimise", llvm_optimise),
                      TranslationUnit("llvm_function", llvm_function),
                      TranslationUnit("llvm_specialize", llvm_specialize))
        optlib_export_config = ExportConfiguration()
        optlib_export_config.add_symbol(python_name='optimise',
                                        symbol_name='_Z9optimiserPFlvEPlS1_',
                                        signature='void(void*, i64*, i64*)',)

        optlib_export_config.add_symbol(python_name='function',
                                        symbol_name='_Z8functionPi',
                                        signature='void(i64*)',)

        optlib_export_config.add_symbol(python_name='specialize',
                                        symbol_name='_Z10specializePl',
                                        signature='void(i64*)',)

        optlib = PIXIECompiler(library_name='opt_library',
                               translation_units=optlib_tus,
                               export_configuration=optlib_export_config,
                               baseline_cpu='nocona',
                               baseline_features=x86.sse3,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        optlib.compile()

    def test_optimise_to_current_hardware(self):

        with self.load_pixie_module('opt_library') as opt_library:
            fn_data = opt_library.__PIXIE__['symbols']['function']
            fn = fn_data['void(i64*)']['cfunc']
            n = ctypes.c_long(10000000)
            out = ctypes.c_long(0)
            opt_func_data = opt_library.__PIXIE__['symbols']['optimise']
            opt_func = opt_func_data['void(void*, i64*, i64*)']['cfunc']
            opt_func(fn, ctypes.byref(n), ctypes.byref(out))
            assert out.value == n.value

            # NOTE: This function is in part derived from:
            # https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/codegen.py
            # and also:
            # https://github.com/numba/llvmlite/blob/main/docs/source/user-guide/examples/ll_fpadd.py

            def recompile(bc, *extra_bc, chip=''):

                def _pass_manager_builder(**kwargs):
                    pmb = llvm.create_pass_manager_builder()
                    pmb.opt_level = 3
                    pmb.loop_vectorize = True
                    pmb.slp_vectorize = True
                    pmb.inlining_threshold = -1
                    return pmb

                def _function_pass_manager(tm, llvm_module, **kwargs):
                    pm = llvm.create_function_pass_manager(llvm_module)
                    tm.add_analysis_passes(pm)
                    with _pass_manager_builder(**kwargs) as pmb:
                        pmb.populate(pm)
                    return pm

                def create_execution_engine():

                    target = llvm.Target.from_default_triple()
                    if chip:
                        target_machine = target.create_target_machine(chip,
                                                                      opt=3)
                    else:
                        target_machine = target.create_target_machine(opt=3)
                    backing_mod = llvm.parse_assembly("")
                    engine = llvm.create_mcjit_compiler(backing_mod,
                                                        target_machine)
                    return engine, target_machine

                def compile_ir(tm, bitc, *extra_bc):

                    mod = llvm.parse_bitcode(bitc)
                    mod.verify()

                    for xtra in extra_bc:
                        extra_mod = llvm.parse_bitcode(xtra)
                        extra_mod.verify()
                        mod.link_in(extra_mod)

                    def _optimize_functions(ll_module):
                        with _function_pass_manager(tm, ll_module) as fpm:
                            for func in ll_module.functions:
                                fpm.initialize()
                                fpm.run(func)
                                fpm.finalize()

                    # opt funcs
                    _optimize_functions(mod)
                    # opt mod
                    pm = llvm.create_module_pass_manager()
                    tm.add_analysis_passes(pm)
                    with _pass_manager_builder() as pmb:
                        pmb.populate(pm)
                    stat, remarks = pm.run_with_remarks(mod)

                    return mod

                def setup_ee(engine, mod):
                    # Now add the module and make sure it is ready
                    # execution
                    engine.add_module(mod)
                    engine.finalize_object()
                    engine.run_static_constructors()

                ee, tm = create_execution_engine()
                mod = compile_ir(tm, bc, *extra_bc)
                setup_ee(ee, mod)
                asm = tm.emit_assembly(mod)
                elf = tm.emit_object(mod)
                return ee, asm, elf, str(mod)

            # Run the function via ctypes for this arch
            bitcode = opt_library.__PIXIE__['bitcode']

            specialize_pixie = opt_library.__PIXIE__['symbols']['specialize']

            ee, asm, elf, final_ll = recompile(bitcode)
            sym_name = specialize_pixie['void(i64*)']['symbol']
            recompiled_fptr = ee.get_function_address(sym_name)
            cfunc_ty = specialize_pixie['void(i64*)']['ctypes_cfunctype']
            cfunc = cfunc_ty(recompiled_fptr)
            out = ctypes.c_long(0)
            cfunc(ctypes.byref(out))
            assert out.value == n.value

            iters = 1000000

            nref = ctypes.byref(n)
            outref = ctypes.byref(out)
            start = timeit.default_timer()
            for x in range(iters):
                opt_func(fn, nref, outref)
            stop = timeit.default_timer()
            elapsed_sym = stop - start

            start = timeit.default_timer()
            for x in range(iters):
                cfunc(outref)
            stop = timeit.default_timer()
            elapsed_jit = stop - start

            assert elapsed_jit < elapsed_sym


if __name__ == '__main__':
    unittest.main()
