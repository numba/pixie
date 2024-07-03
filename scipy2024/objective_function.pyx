# This file contains code for a simple objective function `result = cos(x)`.

import cython
cimport libc.math

# This is the declaration that is portable
cdef public void _Z1fPdS_(double* x, double* result):
    result[0] = libc.math.cos(x[0])


# This is cython export
def f(x, cython.double[::1] ret):
    x_ptr = cython.cast(cython.double, x)
    _Z1fPdS_(&x_ptr, &ret[0])
