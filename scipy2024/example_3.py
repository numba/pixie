import numpy as np
import timeit
import fd_kernel

n = 1_000_000
x = np.arange(n, dtype=np.float32)
h = np.float32(0.1)


def work():
    fd_kernel.central_diff_order2(x, h)


times = timeit.repeat(work, repeat=10, number=1)
print(f"Fastest time: {min(times)} (s).")
