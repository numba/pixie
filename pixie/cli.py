import argparse
import os
import subprocess
import tempfile

from llvmlite import binding as llvm

from pixie.compiler import (
    PIXIECompiler,
    TranslationUnit,
    ExportConfiguration,
    TargetDescription,
)


def tu_from_c_source(fname):
    # TODO put this in support.py or utils.py
    prefix = "pixie-c-build-"
    with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:
        outfile = os.path.join(build_dir, "tmp.bc")
        cmd = (
            "clang",
            "-x",
            "c",
            "-fPIC",
            "-mcmodel=small",
            "-emit-llvm",
            fname,
            "-o",
            outfile,
            "-c",
        )
        print(" ".join(cmd))
        subprocess.run(cmd)
        with open(outfile, "rb") as f:
            data = f.read()
    return TranslationUnit(fname, data)


def default_test_config(triple=None):
    # TODO put this in support.py or utils.py
    # this just encodes some defaults for testing purposes
    if triple is None:
        triple = llvm.get_process_triple()

    arch = triple.split("-")[0]
    if arch == "x86_64":
        from pixie.targets.x86_64 import features, predefined

        return TargetDescription(
            triple,
            predefined.x86_64.cpu,
            predefined.x86_64.features,
            (
                features.sse2,
                predefined.x86_64_v2,
                predefined.x86_64_v3,
                predefined.x86_64_v4,
                features.avx512bitalg,
            ),
        )
    elif arch == "arm64":
        from pixie.targets.arm64 import features, predefined

        return TargetDescription(
            triple,
            predefined.apple_m1.cpu,
            predefined.apple_m1.features,
            (features.v8_5a, features.v8_6a, features.bf16),
        )
    else:
        raise ValueError(f"Unsupported triple: '{triple}'.")


def pixie_cc():
    parser = argparse.ArgumentParser(description="pixie-cc")
    parser.add_argument("-g", action="store_true", help="enable debug info")
    parser.add_argument("-o", help="output library")
    parser.add_argument(
        "-O", metavar="<n>", type=int, nargs=1, help="optimization level"
    )
    parser.add_argument("c-source", help="input source file")
    args = parser.parse_args()
    print(args)
    tranlastion_units = [tu_from_c_source(vars(args)["c-source"])]
    export_config = ExportConfiguration()
    target_description = default_test_config()
    compiler = PIXIECompiler(
        library_name=args.o,
        translation_units=tranlastion_units,
        export_configuration=export_config,
        baseline_cpu=target_description.baseline_target.cpu,
        baseline_features=target_description.baseline_target.features,
        targets_features=target_description._get_targets_features(
            target_description.baseline_target.features,
            target_description.baseline_target.cpu,
        ),
        python_cext=True,  # TODO allow users to specify this
        output_dir=".",  # TODO use $PWD for now
    )
    compiler.compile()


def pixie_cythonize():
    parser = argparse.ArgumentParser(description="pixie-cythonize")
    parser.add_argument("pyx-source", help="input source file")
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    pixie_cc()
