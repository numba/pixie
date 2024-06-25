from llvmlite import binding as llvm
from numba import njit
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir as llvmir
from numba_helpers import pixie_converter


# MVP "blended compilation"

# ------------------------------------------------------------------------------
# Objective functions from C code:
#
# def f(x):
#     return np.cos(x) + 1
#
# def dfdx(x):
#     return -np.sin(x)
#
# The following binds to these functions for use in Numba

def gen_pixie_callsite(pixie_lib, pysym, pixie_sig):
    @intrinsic
    def bind_call(tyctx, ty_x):

        sig = ty_x(ty_x)

        def codegen(cgctx, builder, sig, llargs):
            bitcode = pixie_lib.__PIXIE__['bitcode']
            mod = llvm.parse_bitcode(bitcode)
            cgctx.active_code_library.add_llvm_module(mod)
            foo_sym = pixie_lib.__PIXIE__['symbols'][pysym]
            sym_name = foo_sym[pixie_sig]['symbol']
            double_ptr = llvmir.DoubleType().as_pointer()
            fnty = llvmir.FunctionType(llvmir.VoidType(),
                                       (double_ptr, double_ptr))
            fn = cgutils.get_or_insert_function(builder.module, fnty,
                                                sym_name)
            fn.attributes.add('alwaysinline')
            x_ptr = cgutils.alloca_once_value(builder, llargs[0])
            r_ptr = cgutils.alloca_once(builder, llargs[0].type)
            builder.call(fn, (x_ptr, r_ptr))
            return builder.load(r_ptr)
        return sig, codegen

    @njit(forceinline=True, no_cpython_wrapper=True)
    def pixie_trampoline(x):
        return bind_call(x)
    return pixie_trampoline


import objective_functions  # noqa: E402
f = gen_pixie_callsite(objective_functions, 'f', 'void(double*, double*)')
dfdx = gen_pixie_callsite(objective_functions, 'dfdx',
                          'void(double*, double*)')

# ------------------------------------------------------------------------------

# Optimiser from python code AOT compiled via Numba AOT prototype.
# The `pixie_converter` makes the Numba compatible functions in the `optimiser`
# module available to Numba.

import optimiser  # noqa: E402
numba_optimiser = pixie_converter(optimiser)


# ------------------------------------------------------------------------------
# Call the NR_root function to do root finding for functions f and dfdx
# originally from C code. The .jit variant will use the bitcode from the
# PIXIE library and take part in optimisation at compile time. The .aot
# variant will use the symbol from the PIXIE library and not take part in
# optimisation at compile time as it is an opaque call.


@njit
def specialized_find_roots(eps=1e-7, max_it=50):
    result_jit = numba_optimiser.NR_root.jit(f, dfdx, 0.5, eps, max_it)
    result_aot = numba_optimiser.NR_root.aot(f, dfdx, 0.8, 1e-3, max_it)
    return result_jit, result_aot


# This should print two numbers near the value of pi.
print(specialized_find_roots())

# DEBUG:
# print(specialized_find_roots.inspect_llvm(specialized_find_roots.signatures[0]))
# print(specialized_find_roots.inspect_asm(specialized_find_roots.signatures[0]))
# specialized_find_roots.inspect_types()
