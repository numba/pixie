from pixie_numba_compiler import TranslationUnit, aot, Library
from numba import types

# PIXIE AOT
fn_type = types.FunctionType(types.double(types.double))


# Adapted from: https://github.com/numba/numba-examples/blob/cc0304f9fa75530809dc19fb7168de32b3d1a931/tutorials/nasa_apps_oct_2019/answers/1%20-%20Numba%20basics.ipynb  # noqa: E501
# under the terms of the license:
# https://github.com/numba/numba-examples/blob/cc0304f9fa75530809dc19fb7168de32b3d1a931/LICENSE  # noqa: E501
# Which is as follows:
#
# BSD 2-Clause License
#
# Copyright (c) 2017, Numba
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@aot((types.double(fn_type, fn_type, types.double, types.double,
                   types.int64)),)
def NR_root(f, dfdx, x0, eps, max_it):
    converged = False
    for i in range(max_it):
        y = f(x0)
        yp = dfdx(x0)
        if (abs(yp) < eps):
            break
        x1 = x0 - y / yp
        if abs(x1 - x0) <= eps:
            converged = True
            break
        x0 = x1
    if converged:
        return x1
    else:
        raise RuntimeError("Solution did not converge")


if __name__ == "__main__":
    translation_unit = TranslationUnit()
    translation_unit.add(NR_root)
    export_lib = Library('optimiser', (translation_unit,), outdir='.')
    export_lib.compile()
    # check it imports
    import optimiser  # noqa: F401
