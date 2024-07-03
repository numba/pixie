# Command line interface

PIXIE provides a CLI for building PIXIE libraries.


## `pixie-cc` 

```text
usage: pixie-cc [-h] [-g] [-v] [-O <n>] [-o <lib>] files [files ...]

pixie-cc

positional arguments:
  files       input source files

options:
  -h, --help  show this help message and exit
  -g          compile with debug info
  -v          enable verbose output
  -O <n>      optimization level
  -o <lib>    output library
```

## `pixie-cython` 


```text
usage: pixie-cythonize [-h] [-g] [-v] [-O <n>] [-o <lib>] files [files ...]

pixie-cythonize

positional arguments:
  files       input source files

options:
  -h, --help  show this help message and exit
  -g          compile with debug info
  -v          enable verbose output
  -O <n>      optimization level
  -o <lib>    output library
```
