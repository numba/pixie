from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration


def mvp_compile():
    src = 'objective_function_derivative.c'
    tus = []
    tus.append(TranslationUnit.from_c_source(src))
    export_config = ExportConfiguration()
    export_config.add_symbol(python_name='dfdx',
                             symbol_name='_Z4dfdxPdS_',
                             signature='void(double*, double*)',)
    compiler = PIXIECompiler(library_name='objective_function_derivative',
                             translation_units=tus,
                             export_configuration=export_config,
                             **get_default_configuration(),
                             python_cext=True,
                             output_dir='.')
    compiler.compile()


if __name__ == "__main__":
    mvp_compile()
