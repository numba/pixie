# Command line interface

PIXIE provides a CLI for building PIXIE libraries.


## `pixie-cc` 

```text
usage: pixie-cc [-h] [-g] [-v] [-o <lib>] [-O <n>] c-source

pixie-cc

positional arguments:
  c-source    input source file

options:
  -h, --help  show this help message and exit
  -g          enable debug info
  -v          enable verbose
  -o <lib>    output library
  -O <n>      optimization level
```

## `pixie-cython` 


```text
usage: pixie-cythonize [-h] [-g] [-v] [-o <lib>] [-O <n>] pyx-source

pixie-cythonize

positional arguments:
  pyx-source  input source file

options:
  -h, --help  show this help message and exit
  -g          enable debug info
  -v          enable verbose
  -o <lib>    output library
  -O <n>      optimization level
```
