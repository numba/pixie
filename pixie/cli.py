import argparse


def pixie_cc():
    parser = argparse.ArgumentParser(description="pixie-cc")
    parser.add_argument("-g", action="store_true", help="enable debug info")
    parser.add_argument(
        "-O", metavar="<n>", type=int, nargs=1, help="optimization level"
    )
    parser.add_argument("c-source", help="input source file")
    args = parser.parse_args()
    print(args)


def pixie_cythonize():
    parser = argparse.ArgumentParser(description="pixie-cythonize")
    parser.add_argument("pyx-source", help="input source file")
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    pixie_cc()
