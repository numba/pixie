from numba import njit, typeof
import timeit
import numpy as np

from numba_helpers import pixie_converter
import daxpy

# This is a demonstration of the execution latency vs performance trade-off
# that compilation to a PIXIE library enables. The `daxpy` C-Extension is a
# PIXIE  library that is made available to Numba through the `pixie_converter`.
# The `daxpy` function in the Numba binding to the `daxpy` C-Extension is then
# called via a symbol (AOT) or via JIT, the former compiles more quicky but
# executes more slowly, the latter compiles more slowly but executes more
# quickly, the user gets to choose!

numba_daxpy = pixie_converter(daxpy)


def numpy_result(a, x, y):
    acc = 0
    for i in range(100000):
        a = a * x + y
        acc += a.sum()
    return acc


@njit
def call_aot(a, x, y):
    acc = 0
    for i in range(100000):
        numba_daxpy.daxpy.aot(a, x, y)
        acc += a.sum()
    return acc


@njit
def call_jit(a, x, y):
    acc = 0
    for i in range(100000):
        numba_daxpy.daxpy.jit(a, x, y)
        acc += a.sum()
    return acc


def gen_input(n):
    return np.arange(1, n + 1).astype(np.float64), .1, .2


def check():
    n = 10
    np_args = gen_input(n)
    expected = numpy_result(*np_args)
    aot_args = gen_input(n)
    aot_result = call_aot(*aot_args)
    jit_args = gen_input(n)
    jit_result = call_jit(*jit_args)

    np.testing.assert_allclose(expected, aot_result)
    np.testing.assert_allclose(expected, jit_result)


def runtime_perf_test():
    print("Run time test:")
    args = gen_input(5)
    ts = timeit.default_timer()
    call_aot(*args)
    te = timeit.default_timer()
    print(f"Elapsed (AOT) {te - ts}")
    args = gen_input(5)
    ts = timeit.default_timer()
    call_jit(*args)
    te = timeit.default_timer()
    print(f"Elapsed (JIT) {te - ts}")


def compiletime_perf_test():
    njit()(lambda x: x)(1)  # load numba and compile NRT

    print("Compile time test:")
    args = gen_input(5)
    argtys = tuple(typeof(x) for x in args)

    ts = timeit.default_timer()
    call_aot.compile(argtys)
    te = timeit.default_timer()
    print(f"Elapsed (AOT) {te - ts}")

    ts = timeit.default_timer()
    call_jit.compile(argtys)
    te = timeit.default_timer()
    print(f"Elapsed (JIT) {te - ts}")


if __name__ == "__main__":
    compiletime_perf_test()
    runtime_perf_test()
