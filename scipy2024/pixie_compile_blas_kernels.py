from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration


def mvp_compile():
    src = 'blas_kernels.pyx'
    extra_clang_flags = ('-ffast-math',)
    tus = (TranslationUnit.from_cython_source(src,
           extra_clang_flags=extra_clang_flags),)
    export_config = ExportConfiguration()
    compiler = PIXIECompiler(library_name='blas_kernels',
                             translation_units=tus,
                             export_configuration=export_config,
                             **get_default_configuration(),
                             opt_flags={'loop_vectorize': True,
                                        'slp_vectorize': True},
                             python_cext=True,
                             output_dir='.')
    compiler.compile()


if __name__ == "__main__":
    mvp_compile()
