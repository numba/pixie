import argparse
import pathlib
import os

from pixie import (
    PIXIECompiler,
    TranslationUnit,
    ExportConfiguration,
    targets,
)


def _generate_parser(descr):
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("-g", action="store_true",
                        help="compile with debug info")
    parser.add_argument("-v", action="store_true", help="enable verbose output")
    parser.add_argument(
        "-O", metavar="<n>", type=int, choices=(0, 1, 2, 3),
        help="optimization level", default=3,)
    parser.add_argument("-o", metavar="<lib>", help="output library")
    parser.add_argument("files", help="input source files", nargs="+")
    return parser


def _translate_common_options(parser, args):
    # opt >= 2 adds in vectorization
    opt_flags = dict()
    if args['O'] >= 2:
        opt_flags['loop_vectorize'] = True
        opt_flags['slp_vectorize'] = True

    # use of `-g` needs to be passed to clang
    clang_flags = ()
    if args['g']:
        clang_flags += ('-g',)

    # work out the library name
    if args['o'] is not None:
        library_name = args['o']
    elif len(args['files']) == 1:
        base = os.path.basename(args['files'][0])
        library_name = os.path.splitext(base)[0]
    else:
        msg = ("Option -o (output library) is missing and cannot be inferred "
               "as there are multiple input files.")
        parser.error(msg)

    return opt_flags, clang_flags, library_name


def pixie_cc():

    parser = _generate_parser("pixie-cc")
    args = vars(parser.parse_args())

    opt_flags, clang_flags, library_name = \
        _translate_common_options(parser, args)

    inps = args["files"]
    tus = []
    for inp in inps:
        path = pathlib.Path(inp)
        tus.append(TranslationUnit.from_c_source(str(path),
                                                 extra_flags=clang_flags))

    export_config = ExportConfiguration()
    target_description = targets.get_default_configuration()
    compiler = PIXIECompiler(
        library_name=library_name,
        translation_units=tus,
        export_configuration=export_config,
        **target_description,
        opt_flags=opt_flags,
        python_cext=True,  # TODO allow users to specify this
        output_dir=".",  # TODO use $PWD for now
    )
    compiler.compile()


def pixie_cythonize():
    parser = _generate_parser("pixie-cythonize")
    args = vars(parser.parse_args())

    opt_flags, clang_flags, library_name = \
        _translate_common_options(parser, args)

    # cythonize the source
    inps = args["files"]
    tus = []
    for inp in inps:
        path = pathlib.Path(inp)
        tus.append(TranslationUnit.from_cython_source(str(path),
                   extra_clang_flags=clang_flags))

    export_config = ExportConfiguration()
    target_description = targets.get_default_configuration()
    compiler = PIXIECompiler(
        library_name=library_name,
        translation_units=tus,
        export_configuration=export_config,
        **target_description,
        opt_flags=opt_flags,
        python_cext=True,  # TODO allow users to specify this
        output_dir=".",  # TODO use $PWD for now
    )
    compiler.compile()
