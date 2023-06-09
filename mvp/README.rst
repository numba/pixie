Numba MVP 2023
##############

This directory contains a couple of examples of uses of PIXIE, one of which is
the "blended compilation" MVP as described in Numba's road map for 2023.

Instructions to run the examples:

0. Create an environment and install PIXIE locally. Obtain a ``clang`` version
   compatible with the LLVM version that the ``llvmlite`` installation is built
   against (probably 11 or 14). Install a compatible version of Numba into the
   environment (the ``llvmlite`` version will dictate compatibility). If using
   ``conda`` something like this is likely to provide a suitable environment::

   $ conda create -n "<env name>" numba setuptools clang=14 llvmlite=0.40

1. A copy of Numba's source is needed to bootstrap the compiler as the LLVM IR
  from a cut-down version of Numba's ``helperlib`` is needed. Obtain this source
  via any means, the rest of the document will refer to the directory containing
  the source as ``<numba_src>``.

2. Bootstrap the Numba AOT compiler by running::
   $ python bootstrap.py <numba_src>

   This will create a ``_limited_helpermod.c`` and a ``_limited_helpermod.ll``,
   the latter is used to bootstrap the Numba AOT compiler.

3. To run the MVP "blended compilation" example.

   1. Run::
   $ python pixie_c_compiler.py

   to build an ``objective_functions`` C-extension module from the two `.c`
   files present in this directory.

   2. Run::
   $ python numba_aot_optimiser.py

   to build an ``optimiser`` C-extension module from Numba AOT compiled python
   code.

   3. Run::
   $ python numba_mvp.py

   to see the MVP example running. See the Python file for an explanation of
   what is happening.

4. To run a BLAS-like ``DAXPY`` example:

   1. Run::
   $ python numba_aot_daxpy.py

   to build an ``daxpy`` C-extension module from Numba AOT compiled python
   code.

   2. Run::
   $ python mvp_daxpy_call.py

   to see a DAXPY example running. See the Python file for an explanation of
   what is happening.
