# Use this to bootstrap the AOT compiler demo from a Numba source checkout.
import os
import subprocess
import sys
import numpy as np
import sysconfig

# Adapted from: https://github.com/numba/numba/blob/7f056946c7a69f8739c07ef5a1bdb4b4b5be72cd/numba/_helpermod.c

_banned = """
    declmethod(test_dict);
    declmethod(dict_new_sized);
    declmethod(dict_set_method_table);
    declmethod(dict_free);
    declmethod(dict_length);
    declmethod(dict_lookup);
    declmethod(dict_insert);
    declmethod(dict_insert_ez);
    declmethod(dict_delitem);
    declmethod(dict_popitem);
    declmethod(dict_iter_sizeof);
    declmethod(dict_iter);
    declmethod(dict_iter_next);
    declmethod(dict_dump);
    declmethod(test_list);
    declmethod(list_new);
    declmethod(list_set_method_table);
    declmethod(list_free);
    declmethod(list_base_ptr);
    declmethod(list_size_address);
    declmethod(list_length);
    declmethod(list_allocated);
    declmethod(list_is_mutable);
    declmethod(list_set_is_mutable);
    declmethod(list_setitem);
    declmethod(list_getitem);
    declmethod(list_append);
    declmethod(list_delitem);
    declmethod(list_delete_slice);
    declmethod(list_iter_sizeof);
    declmethod(list_iter);
    declmethod(list_iter_next);
    declmethod(get_py_random_state);
    declmethod(get_np_random_state);
    declmethod(get_internal_random_state);
    declmethod(rnd_shuffle);
    declmethod(rnd_init);
    declmethod(poisson_ptrs);
    { "rnd_get_state", (PyCFunction) _numba_rnd_get_state, METH_O, NULL },
    { "rnd_get_py_state_ptr", (PyCFunction) _numba_rnd_get_py_state_ptr, METH_NOARGS, NULL },
    { "rnd_get_np_state_ptr", (PyCFunction) _numba_rnd_get_np_state_ptr, METH_NOARGS, NULL },
    { "rnd_seed", (PyCFunction) _numba_rnd_seed, METH_VARARGS, NULL },
    { "rnd_set_state", (PyCFunction) _numba_rnd_set_state, METH_VARARGS, NULL },
    { "rnd_shuffle", (PyCFunction) _numba_rnd_shuffle, METH_O, NULL },
    { "_import_cython_function", (PyCFunction) _numba_import_cython_function, METH_VARARGS, NULL },
    numba_rnd_ensure_global_init();
"""  # noqa: E501


def _extract(fname):
    nb_root = os.path.abspath(provided_path)
    print(("Bootstrapping Numba AOT against Numba source directory:"
           f" {nb_root}."))
    print("* Bootstrapping started.")
    helpermod = '_helpermod'
    fname = os.path.join(nb_root, 'numba', f'{helpermod}.c')
    with open(fname, 'rt') as f:
        src = f.read()

    banned = set([x.strip() for x in _banned.splitlines() if x])
    new_src = []
    for line in src.splitlines():
        for ban in banned:
            if ban in line:
                break
        else:
            new_src.append(line)

    np_include = np.get_include()
    py_include = sysconfig.get_config_var('INCLUDEPY')
    nb_include = os.path.join(nb_root, 'numba')
    limited = f'_limited{helpermod}'
    msg = (f"* Copying and patching Numba's '{helpermod}.c' to a reduced "
           f"version: '{limited}.c'.")
    print(msg)
    with open(f"{limited}.c", 'wt') as f:
        f.write('\n'.join(new_src))

    cmd = ('clang', '-x', 'c', '-fPIC', '-mcmodel=small', '-emit-llvm',
           '-I', nb_include, '-I', np_include, '-I', py_include,
           f'{limited}.c', '-S')
    msg = (f"* Running this command to create LLVM IR of the '{limited}.c' "
           "that's AOT compatible:\n")
    print(msg)
    print("$ " + ' '.join(cmd) + "\n")
    subprocess.run(cmd, timeout=30)
    print(("* Bootstrapping complete. "
           f"Outputs are '{limited}.c' and '{limited}.ll'"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: bootstrap.py <path to numba source>")
    provided_path = sys.argv[1]
    if not os.path.exists(provided_path):
        raise RuntimeError(f"Path: {provided_path} does not exist.")
    _extract(provided_path)
