from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.cpus import x86
from pixie.tests.support import PixieTestCase
import unittest


try:
    from numba import njit, types
    from numba.extending import overload, intrinsic
    from numba.core import cgutils
    from llvmlite import binding as llvm
    from llvmlite import ir as llvmir
    import numpy as np
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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


@unittest.skipUnless(_HAS_NUMBA, "Numba required for test")
class TestNumbaBinding(PixieTestCase):
    """Tests a few ways of binding from Numba to a PIXIE library.
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

        libfoo.compile()

    def test_bind(self):
        with self.load_pixie_module('foo_library') as numba_library:
            foo_sym = numba_library.__PIXIE__['symbols']['foo']
            assert len(foo_sym) == 2  # 2 variants
            for sig, details in foo_sym.items():
                cfunc = details['cfunc']

                @njit
                def sink(*args):
                    pass

                if "double" in sig:
                    dtype = np.float64
                else:
                    dtype = np.int64

                @njit
                def foo():
                    x = np.array(10, dtype=dtype)
                    y = np.array(20, dtype=dtype)
                    res = np.array(0, dtype=dtype)
                    cfunc(x.ctypes, y.ctypes, res.ctypes)
                    sink(x, y, res)
                    return res

                assert foo() == 30

    def test_overload_fixed_types(self):
        with self.load_pixie_module('foo_library') as numba_library:
            # Function to overload
            def foo(x, y):
                pass

            def _foo_dispatch(x, y):
                pass

            @overload(_foo_dispatch)
            def ol__foo_dispatch(x, y):

                def isdouble(z):
                    return isinstance(z, types.Float) and z.bitwidth == 64

                def isi64(z):
                    return isinstance(z, types.Integer) and z.bitwidth == 64

                @njit
                def sink(*args):
                    pass

                foo_sym = numba_library.__PIXIE__['symbols']['foo']
                if isdouble(x) and isdouble(y):
                    fn = foo_sym['void(double*, double*, double*)']['cfunc']
                    dt = np.float64
                elif isi64(x) and isi64(y):
                    fn = foo_sym['void(i64*, i64*, i64*)']['cfunc']
                    dt = np.int64
                else:
                    return None

                def impl(x, y):
                    x = np.array(x, dtype=dt)
                    y = np.array(y, dtype=dt)
                    res = np.array(0, dtype=dt)
                    fn(x.ctypes, y.ctypes, res.ctypes)
                    sink(x, y, res)
                    return np.take(res, 0)
                return impl

            @njit(['double(double, double)', 'int64(int64, int64)'])
            def foo_dispatch(x, y):
                return _foo_dispatch(x, y)

            @overload(foo)
            def ol_foo(x, y):
                def impl(x, y):
                    return foo_dispatch(x, y)
                return impl

            @njit
            def call_foo():
                return foo(10, 20), foo(1.0, 2.0)

            assert call_foo() == (30, 3.0)

    def test_overload_recompile(self):

        with self.load_pixie_module('foo_library') as numba_library:
            # Function to overload
            def foo(x, y):
                pass

            @intrinsic
            def recompiled_foo(tyctx, ty_x, ty_y):
                if ty_x != ty_y:
                    return None, None

                sig = ty_x(ty_x, ty_y)

                def codegen(cgctx, builder, sig, llargs):
                    bitcode = numba_library.__PIXIE__['bitcode']
                    tmpmod = llvm.parse_bitcode(bitcode)
                    # deliberately mess with the symbol name to make sure it is
                    # definitely not the symbol from the c-extension being used
                    different_symbol_name = "DIFFERENT_SYMBOL"
                    foo_sym = numba_library.__PIXIE__['symbols']['foo']
                    sym_data = foo_sym['void(double*, double*, double*)']
                    sym_name = sym_data['symbol']
                    new_ir = str(tmpmod).replace(sym_name,
                                                 different_symbol_name)
                    mod = llvm.parse_assembly(new_ir)
                    cgctx.active_code_library.add_llvm_module(mod)
                    double_ptr = llvmir.DoubleType().as_pointer()
                    fnty = llvmir.FunctionType(llvmir.VoidType(),
                                               (double_ptr, double_ptr,
                                                double_ptr))
                    fn = cgutils.get_or_insert_function(builder.module, fnty,
                                                        different_symbol_name)
                    x_ptr = cgutils.alloca_once_value(builder, llargs[0])
                    y_ptr = cgutils.alloca_once_value(builder, llargs[1])
                    r_ptr = cgutils.alloca_once(builder, llargs[0].type)
                    ret = builder.call(fn, (x_ptr, y_ptr, r_ptr))  # noqa: F841
                    # TODO: something with ret?
                    return builder.load(r_ptr)
                return sig, codegen

            @overload(foo)
            def ol_foo(x, y):
                def impl(x, y):
                    return recompiled_foo(x, y)
                return impl

            @njit
            def call_foo():
                return foo(1.0, 2.0)

            assert call_foo() == 3.0


if __name__ == '__main__':
    unittest.main()
