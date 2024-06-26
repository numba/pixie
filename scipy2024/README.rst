PIXIE @ SciPy 2024
##################

This directory contains some example uses of PIXIE that are part of a
demonstration of this technology presented at SciPy 2024. They are organised as
follows:

1. Demo 1 explores the contents of the ``__PIXIE__`` module level
   directory.
2. Demo 2 demonstrates various ways to call functions in a PIXIE
   compiled extension module.
3. Demo 3 is used to demonstrate the power of ISA specific dispatch.
4. Demo 4 is based on the "blended compilation" MVP as described in Numba's road
   map for 2023. It uses C, Cython and python source (the latter
   compiled with a prototype Numba-AOT compiler) in PIXIE extension modules and
   mixes the use of AOT and JIT compiled symbols at compile time. Higher
   performance is demonstrated through the presence of the additional context
   PIXIE modules provide.

NOTE: This demonstration currently targets x86_64 CPUs running Linux operating
systems and Apple M1 silicon running OSX. Other CPUs and operating systems will
be supported in the future.

Set up instructions for all examples:

0. First create an environment and install PIXIE locally. PIXIE depends on the
   ``setuptools`` and ``llvmlite`` packages, this demonstration
   also needs a suitable ``clang`` and a Numba installation. If using ``conda``
   something like this is likely to provide a suitable environment for running
   the demonstration and installing PIXIE, it currently relies on development
   versions of both ``Numba`` and ``llvmlite``, which are available from the
   ``numba`` channel under the ``dev`` label.

   For ``x86_64`` Linux users::

   $ conda create -n "<env name>" numba/label/dev::numba=0.61.0dev0 numba/label/dev::llvmlite=0.44.0dev0 setuptools clang=14 python=3.11 gcc_linux-64 gxx_linux-64 cython

   For ``osx-arm64`` users (Apple M1 silicon)::

   $ conda create -n "<env name>" numba/label/dev::numba=0.61.0dev0 numba/label/dev::llvmlite=0.44.0dev0 setuptools python=3.11 clang_osx-arm64=14 clangxx_osx-arm64=14 cython

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


Running the demos:
==================

1. Demo 1. explores the contents of the ``__PIXIE__`` module level directory.

   Run::

      $ python pixie_compile_objective_function.py

      to build an ``objective_function`` extension module from the
      ``objective_function.pyx`` Cython source.


      $ python example_1.py

      See the Python file for an explanation of what is happening.

2. Demo 2. demonstrates various ways to call functions in a PIXIE compiled
   extension module.

   Run::

      $ python pixie_compile_objective_function.py

      to build an ``objective_function`` extension module from the
      ``objective_function.pyx`` Cython source.

      $ python example_2.py

      to demonstrate using the extension module from both the Python interpreter
      and Numba ``@jit`` compiled regions. See the Python file for an
      explanation of what is happening.

3. Demo 3. demonstrates the power of ISA based dispatch (this demo is for
   ``x86_64` systems only and best run on a recently new machine).

   Run::

      $ python pixie_compile_fd_kernel.py

      to build an ``fd_kernel`` extension module from the ``fd_kernel.pyx``
      Cython source.


      $ python example_3.py

      This will print a runtime for executing with the embedded library variant
      the best matches the CPU on which it is running. Now rerun with the oldest
      ISA available in the default configuration ("sse2") via setting an
      environment variable:

      $ PIXIE_USE_ISA="sse2" python example_3.py

      This will print another runtime, note how much slower the code runs with
      the older instruction set.

4. Demo 4. demonstrates multiple input source languages and use of both AOT and
   JIT compiled code.

   1. Run::

      $ python pixie_compile_objective_function_derivative.py

      to build an ``objective_function_derivative`` extension module from the
      ``objective_function_derivative.c`` C-language source file.

   2. Run::

      $ python pixie_compile_objective_function.py

      to build an ``objective_function`` extension module from the
      ``objective_function.pyx`` Cython source file.

   3. Run::

      $ python numba_aot_compile_optimiser.py

      to build an ``optimiser`` C-extension module from Numba AOT compiled
      python code found in the same file (this is compiling Python source into
      and extension module).

   4. Run::

      $ python example_4.py

      to see the example running. See the Python file for an explanation of
      what is happening.
