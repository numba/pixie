from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
import subprocess
import tempfile
import os
from pixie.cpus import x86


def tu_from_c_source(fname):
    prefix = f'pixie-c-build-{fname}-'
    with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:
        outfile = os.path.join(build_dir, 'tmp.bc')
        cmd = ('clang', '-x', 'c++', '-fPIC', '-mcmodel=small', '-emit-llvm',
               fname, '-o', outfile, '-c')
        subprocess.run(cmd)
        with open(outfile, 'rb') as f:
            data = f.read()
    return TranslationUnit(fname, data)


def mvp_compile():
    src = ('objective_function.c', 'objective_function_derivative.c')
    tus = []
    for s in src:
        tus.append(tu_from_c_source(s))
    export_config = ExportConfiguration(versioning_strategy='embed_dso')
    export_config.add_symbol(python_name='f',
                             symbol_name='_Z1fPdS_',
                             signature='void(double*, double*)',)
    export_config.add_symbol(python_name='dfdx',
                             symbol_name='_Z4dfdxPdS_',
                             signature='void(double*, double*)',)
    compiler = PIXIECompiler(library_name='objective_functions',
                             translation_units=tus,
                             export_configuration=export_config,
                             baseline_cpu='nocona',
                             baseline_features=x86.sse3,
                             python_cext=True,
                             output_dir='.')
    compiler.compile()


if __name__ == "__main__":

    mvp_compile()
