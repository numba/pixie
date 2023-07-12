Numba MVP 2023
##############

This directory contains a couple of examples of uses of PIXIE, one of which is
the "blended compilation" MVP as described in Numba's road map for 2023. It also
contains a notebook demonstrating PIXIE and its interaction with a
proof-of-concept Numba ahead-of-time (AOT) compiler and more detail on the MVP.

NOTE: PIXIE and this demonstration currently target x86_64 CPUs running Linux
operating systems. Other CPUs and operating systems will be supported in the
future.

Set up instructions:

0. First create an environment and install PIXIE locally. PIXIE depends on the
   ``setuptools`` and `llvmlite`` packages, this demonstration
   also needs a suitable ``clang`` and a Numba installation. To do this, obtain
   a ``clang`` version compatible with the LLVM version that the ``llvmlite``
   installation is built against (probably 11 or 14). Install a compatible
   version of Numba into the environment (the ``llvmlite`` version will dictate
   compatibility). If using ``conda`` something like this is likely to provide a
   suitable environment for running the demonstration and installing PIXIE::

   $ conda create -n "<env name>" numba setuptools clang=14 llvmlite=0.40 python=3.10 python-graphviz jupyterlab gcc_linux-64 gxx_linux-64

1. A copy of Numba's source is needed to bootstrap the Numba AOT compiler as the
   LLVM IR from a cut-down version of Numba's ``helperlib`` C code is needed.
   Obtain this source via a git clone method of your choice, the rest of this
   document will refer to the directory containing the source as
   ``<numba_src>``.

2. Apply the patch ``numba_bootstrap.patch`` to the Numba source (this just
   comments out some code that end up generating invalid relocations for the
   linker when running in Numba's JIT compiler). Something similar to this may
   be appropriate to do the patching::

   $ pushd <numba_src>
   $ git apply <path/to/this/directory/>numba_bootstrap.patch
   $ popd

3. Bootstrap the Numba AOT compiler by running::

   $ python bootstrap.py <numba_src>

   This will create a ``_limited_helpermod.ll`` that is used to bootstrap the
   Numba AOT compiler.

4. The notebook ``pixie_demonstration.ipynb`` can now be launched to run a
   demonstration, ``jupyter lab`` is an appropriate tool for this. Alternatively
   steps 5. and 6. show similar from the command line.

5. To run the MVP "blended compilation" example.

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

6. To run a BLAS-like ``DAXPY`` example:

   1. Run::

      $ python numba_aot_daxpy.py

      to build an ``daxpy`` C-extension module from Numba AOT compiled python
      code.

   2. Run::

      $ python mvp_daxpy_call.py

      to see a DAXPY example running. See the Python file for an explanation of
      what is happening.
