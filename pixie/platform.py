# NOTE: This module is a copy (with modifications) of:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/platform.py

from distutils.ccompiler import CCompiler, new_compiler
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler
from distutils import log

import functools
import os
import sys
from tempfile import mkdtemp, TemporaryDirectory
from contextlib import contextmanager

_configs = {
    # DLL suffix, Python C extension suffix
    'win': ('.dll', '.pyd'),
    'default': ('.so', '.so'),
}


def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]


find_shared_ending = functools.partial(get_configs, 0)
find_pyext_ending = functools.partial(get_configs, 1)


@contextmanager
def _gentmpfile(suffix, directory):
    # windows locks the tempfile so use a tempdir + file, see
    # https://github.com/numba/numba/issues/3304
    try:
        tmpdir = mkdtemp(dir=directory)
        ntf = open(os.path.join(tmpdir, "temp%s" % suffix), 'wt')
        yield ntf
    finally:
        try:
            ntf.close()
            os.remove(ntf)
        except Exception:
            pass
        else:
            os.rmdir(tmpdir)


def _check_external_compiler():
    # see if the external compiler bound in numpy.distutil is present
    # and working
    compiler = new_compiler()
    customize_compiler(compiler)
    with TemporaryDirectory() as basetmpdir:
        for suffix in ['.c', '.cxx']:
            try:
                with _gentmpfile(suffix, basetmpdir) as ntf:
                    simple_c = "int main(void) { return 0; }"
                    ntf.write(simple_c)
                    ntf.flush()
                    ntf.close()
                    # *output_dir* is set to avoid the compiler putting temp
                    # files in the current directory.
                    compiler.compile([ntf.name], output_dir=basetmpdir)
            except Exception:  # likely CompileError or file system issue
                return False
    return True

# boolean on whether the externally provided compiler is present and
# functioning correctly


_external_compiler_ok = _check_external_compiler()


class _DummyExtension(object):
    libraries = []


class Toolchain(object):

    def __init__(self):
        if not _external_compiler_ok:
            self._raise_external_compiler_error()

        # Need to import it here since setuptools may monkeypatch it
        from distutils.dist import Distribution
        self._verbose = False
        self._compiler = new_compiler()
        customize_compiler(self._compiler)
        self._build_ext = build_ext(Distribution())
        self._build_ext.finalize_options()
        self._py_lib_dirs = self._build_ext.library_dirs
        self._py_include_dirs = self._build_ext.include_dirs

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        # DEBUG will let Numpy spew many messages, so stick to INFO
        # to print commands executed by distutils
        log.set_threshold(log.INFO if value else log.WARN)

    def _raise_external_compiler_error(self):
        basemsg = ("Attempted to compile AOT function without the "
                   "compiler used by `numpy.distutils` present.")
        conda_msg = "If using conda try:\n\n#> conda install %s"
        plt = sys.platform
        if plt.startswith('linux'):
            if sys.maxsize <= 2 ** 32:
                compilers = ['gcc_linux-32', 'gxx_linux-32']
            else:
                compilers = ['gcc_linux-64', 'gxx_linux-64']
            msg = "%s %s" % (basemsg, conda_msg % ' '.join(compilers))
        elif plt.startswith('darwin'):
            compilers = ['clang_osx-64', 'clangxx_osx-64']
            msg = "%s %s" % (basemsg, conda_msg % ' '.join(compilers))
        elif plt.startswith('win32'):
            winmsg = "Cannot find suitable msvc."
            msg = "%s %s" % (basemsg, winmsg)
        else:
            msg = "Unknown platform %s" % plt
        raise RuntimeError(msg)

    def compile_objects(self, sources, output_dir,
                        include_dirs=(), depends=(), macros=(),
                        extra_cflags=None):
        """
        Compile the given source files into a separate object file each,
        all beneath the *output_dir*.  A list of paths to object files
        is returned.

        *macros* has the same format as in distutils: a list of 1- or 2-tuples.
        If a 1-tuple (name,), the given name is considered undefined by
        the C preprocessor.
        If a 2-tuple (name, value), the given name is expanded into the
        given value by the C preprocessor.
        """
        objects = self._compiler.compile(sources,
                                         output_dir=output_dir,
                                         include_dirs=include_dirs,
                                         depends=depends,
                                         macros=macros or [],
                                         extra_preargs=extra_cflags)
        return objects

    def link_shared(self, output, objects, libraries=(),
                    library_dirs=(), export_symbols=(),
                    extra_ldflags=None):
        """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
        output_dir, output_filename = os.path.split(output)
        self._compiler.link(CCompiler.SHARED_OBJECT, objects,
                            output_filename, output_dir,
                            libraries, library_dirs,
                            export_symbols=export_symbols,
                            extra_postargs=extra_ldflags)

    def get_python_libraries(self):
        """
        Get the library arguments necessary to link with Python.
        """
        libs = self._build_ext.get_libraries(_DummyExtension())
        if sys.platform == 'win32':
            # Under Windows, need to link explicitly against the CRT,
            # as the MSVC compiler would implicitly do.
            # (XXX msvcrtd in pydebug mode?)
            libs = libs + ['msvcrt']
        return libs

    def get_python_library_dirs(self):
        """
        Get the library directories necessary to link with Python.
        """
        return list(self._py_lib_dirs)

    def get_python_include_dirs(self):
        """
        Get the include directories necessary to compile against the Python
        """
        return list(self._py_include_dirs)

    def get_ext_filename(self, ext_name):
        """
        Given a C extension's module name, return its intended filename.
        """
        return self._build_ext.get_ext_filename(ext_name)
