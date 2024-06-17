from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration


def mvp_compile():
    src = 'fd_kernel.pyx'
    tus = (TranslationUnit.from_cython_source(src),)
    export_config = ExportConfiguration()
    compiler = PIXIECompiler(library_name='fd_kernel',
                             translation_units=tus,
                             export_configuration=export_config,
                             **get_default_configuration(),
                             python_cext=True,
                             output_dir='.')
    compiler.compile()


if __name__ == "__main__":
    mvp_compile()
