import numpy as np
from timeit import default_timer
from llvmlite import binding as llvm
from numba import njit
from numba_helpers import pixie_converter, gen_pixie_raw_callsite

# ------------------------------------------------------------------------------
# "Blending" input languages through AOT compiled modules.
# Example application, a "custom" solver.
# ------------------------------------------------------------------------------
#
# This example demonstrates the use of a Newton-Raphson based solver to compute
# the roots of the function `cos(x)`. The purpose of this example is to emulate
# code being written in different languages and packaged by different parties
# and yet the user is able to make choices about how to incorporate the packaged
# code.
#
# 1. One "author" writes the objective function in Cython code and compiles into
#    a PIXIE library shipped by their package.,
# 2. A second "author" writes the objective function derivative in C code and
#    compiles into a PIXIE library shipped by their package.
# 3. A third "author" writes a Newton-Raphson solver in Python and compiles into
#    another PIXIE library shipped by their package using the Numba AOT
#    compiler.
# 4. The libraries in 1., 2. and 3. are used by a third author in their custom
#    application, which is compiled with Numba's JIT compiler, and choices can
#    be made about whether to call via bitcode or via symbols."

# ------------------------------------------------------------------------------
# A reminder... here are functions described in 1. and 2.
#
# Objective function from Cython module (pseudo-code):
#
# def f(x):
#     return cos(x)
#
# Objective function derivative from C module (pseudo-code):
#
# def dfdx(x):
#     return -sin(x)
#
# The following binds to these functions for use in Numba JIT code.

import objective_function  # noqa: E402
f = gen_pixie_raw_callsite(objective_function, 'f', 'void(double*, double*)')

import objective_function_derivative  # noqa: E402
dfdx = gen_pixie_raw_callsite(objective_function_derivative, 'dfdx',
                              'void(double*, double*)')

# ------------------------------------------------------------------------------
# Import and adapt the Newton-Raphson optimiser from python code AOT compiled
# via Numba AOT prototype. The `pixie_converter` makes the Numba compatible
# functions in the `optimiser` module available to Numba.

import optimiser  # noqa: E402
numba_optimiser = pixie_converter(optimiser)


# ------------------------------------------------------------------------------
# Call the NR_root function to do root finding for functions f and dfdx
# originally from cython and C code respectively. The .jit variant will use the
# bitcode from the PIXIE library and take part in optimisation at compile time.
# The .aot variant will use the symbol from the PIXIE library and not take part
# in optimisation at compile time as it is an opaque call.


@njit
def specialized_find_roots(eps=1e-7, max_it=50):
    result_jit = numba_optimiser.NR_root.jit(f, dfdx, 0.5, eps, max_it)
    result_aot = numba_optimiser.NR_root.aot(f, dfdx, 0.8, 1e-3, max_it)
    return result_jit, result_aot


# This should print two numbers near the value of pi.
print("Numerical result of the root find (jit, aot), it's pi/2!")
print(specialized_find_roots(), "\n")

# ------------------------------------------------------------------------------
# Now do something to demonstrate the power of additional context... split the
# JIT and AOT variants out into their own functions and measure their execution
# time.


@njit
def specialized_find_roots_jit():
    return numba_optimiser.NR_root.jit(f, dfdx, 0.5, 1e-8, 50)


@njit
def specialized_find_roots_aot():
    return numba_optimiser.NR_root.aot(f, dfdx, 0.5, 1e-8, 50)


n = 10000000  # iterations

print(f"Performance Results from {n} iterations:")
# time the AOT compiled function call
specialized_find_roots_aot()
ts = default_timer()
for x in range(n):
    specialized_find_roots_aot()
te = default_timer()
aot_time = te - ts
print(f"aot time: {aot_time} (s)")

# time the JIT compiled function call
specialized_find_roots_jit()
ts = default_timer()
for x in range(n):
    specialized_find_roots_jit()
te = default_timer()
jit_time = te - ts
print(f"jit time: {jit_time} (s)")
print(f"Speed up: ~{round(aot_time / jit_time, 2)}x")


# What has the JIT done? It's worked out that everything is a constant,
# including the functions from the cython and C modules. It's then unrolled the
# Numba-AOT compiled Newton-Raphson iteration loop, constant propagated and
# constant folded so as to realise that the answer is always 1.570796... (pi/2).
# This is expressed as a hexadecimal constant in the LLVM IR and subsequently
# generated machine code. This is why the JIT version is much quicker than the
# AOT version... it has to do no work and it did this because it had enough
# context to figure that out!

# Extract the LLVM IR for the specialized_find_roots_jit function, note the
# storing of a constant value to the `retptr` (return pointer).
sfrj = specialized_find_roots_jit  # a shorter alias!
mangled_name = sfrj.overloads[sfrj.signatures[0]].fndesc.mangled_name
mod = llvm.parse_assembly(sfrj.inspect_llvm(sfrj.signatures[0]))
function_ir = str([f for f in mod.functions if f.name == mangled_name][0])
print("\n\nLLVM IR for JIT compiled function:\n")
print(function_ir)

# What's that constant number? (it's pi / 2)
const = hex((np.pi / np.float64(2.)).view(np.uint64))
print(f"The constant pi/2 in hex: {const}")
assert const in function_ir.lower()
