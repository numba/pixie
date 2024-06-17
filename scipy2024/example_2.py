# Example 2 shows various ways to call functions from a __PIXIE__ dictionary
import objective_function
import numpy as np
import ctypes
from numba_helpers import gen_pixie_raw_callsite
from numba import njit

# Call Cython entry point from Python
result = np.zeros(1)
objective_function.f(0, result)
print(f"Python call via cython function export, result: {result}")

# Call PIXIE ctypes wrapper from Python
pixie_sig = 'void(double*, double*)'
cfunc = objective_function.__PIXIE__['symbols']['f'][pixie_sig]['cfunc']  # noqa
x = ctypes.c_double(0.)
result = ctypes.c_double(0.)
cfunc(ctypes.byref(x), ctypes.byref(result))
print(f"Python call via __PIXIE__ ctypes binding, result: {result.value}")


# Call PIXIE from numba using ctypes wrapped function
@njit
def call_f_ctypes(x):
    x_tmp = np.zeros(1)
    x_tmp[0] = x
    res_tmp = np.zeros(1)
    cfunc(x_tmp.ctypes, res_tmp.ctypes)
    return res_tmp[0]


print("Numba call via __PIXIE__ ctypes raw binding, "
      f"result: {call_f_ctypes(0.)}")


# Call PIXIE ctypes wrapper from Numba
# Adapt the call site
f = gen_pixie_raw_callsite(objective_function, 'f', pixie_sig)


@njit
def call_f_wrapper(x):
    return f(x)


print("Numba call via __PIXIE__ ctypes wrapper binding, "
      f"result: {call_f_wrapper(0.)}")
