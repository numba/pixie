# Use this to bootstrap the AOT compiler demo from a Numba source checkout.
# Make sure that the patch `numba_bootstrap.patch` is applied to the Numba
# source first!
import os
import subprocess
import sys
import numpy as np
import sysconfig


def _extract(fname):
    nb_root = os.path.abspath(provided_path)
    print(("Bootstrapping Numba AOT against Numba source directory:"
           f" {nb_root}."))
    print("* Bootstrapping started.")

    np_include = np.get_include()
    py_include = sysconfig.get_config_var('INCLUDEPY')
    nb_include = os.path.join(nb_root, 'numba')
    helpermod = os.path.join(nb_include, '_helpermod.c')

    cmd = ('clang', '-x', 'c', '-fPIC', '-mcmodel=small', '-emit-llvm',
           '-I', nb_include, '-I', np_include, '-I', py_include,
           helpermod, '-S', '-o', '_limited_helpermod.ll')
    msg = ("* Running this command to create LLVM IR of '_helpermod.c' "
           "that's AOT compatible:\n")
    print(msg)
    print("$ " + ' '.join(cmd) + "\n")
    subprocess.run(cmd, timeout=30)
    print(("* Bootstrapping complete. Output is '_limited_helpermod.ll'"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: bootstrap.py <path to numba source>")
    provided_path = sys.argv[1]
    if not os.path.exists(provided_path):
        raise RuntimeError(f"Path: {provided_path} does not exist.")
    _extract(provided_path)
