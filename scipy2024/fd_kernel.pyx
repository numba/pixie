# cython: infer_types=True
import numpy as np
import cython
cimport libc.math

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def central_diff_order2(float[::1] x, float h):
    n = x.shape[0]
    cdef Py_ssize_t i
    for i in range(1, n - 1):
        x[i] = (x[i + 1] - float(2.) * x[i] + x[i - 1]) / (h * h)
