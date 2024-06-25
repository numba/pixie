from pixie_numba_compiler import TranslationUnit, aot, Library
from numba import types


# PIXIE AOT
@aot(types.double[::1](types.double[::1], types.double, types.double))
def daxpy(a, x, y):
    for i in range(len(a)):
        a[i] = a[i] * x + y
    return a


if __name__ == "__main__":
    translation_unit = TranslationUnit()
    translation_unit.add(daxpy)
    export_lib = Library('daxpy', (translation_unit,), outdir='.')
    export_lib.compile()
    # check it imports
    import daxpy
