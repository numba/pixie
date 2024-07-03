import numpy as np
import timeit
import blas_kernels

rows = 1024
cols = 4096

dt = np.float32
A = np.ones((rows, cols), dt)
x = np.ones(cols, dt)
y = np.zeros(rows, dt)


def work():
    blas_kernels.sgemv(y, A, x)


times = timeit.repeat(work, repeat=10, number=1)
print(f"Fastest time: {min(times):f} (s).")

np.testing.assert_allclose(np.dot(A, x), y)
