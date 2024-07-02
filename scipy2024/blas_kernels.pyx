# cython: infer_types=True
import numpy as np
import cython
cimport libc.math

@cython.boundscheck(False)
@cython.wraparound(False)
def sgemv(float[::1] y, float[:, ::1] A, float[::1] x):
    cdef size_t rows = A.shape[0]
    cdef size_t cols = A.shape[1]
    cdef size_t i, j
    cdef float tmp

    for i in range(rows):
        tmp = 0.
        for j in range(cols):
            tmp += A[i, j] * x[j]
        y[i] = tmp
