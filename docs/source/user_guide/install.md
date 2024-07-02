# Setup and Installation

To install with conda:

```bash
% conda install -c numba -c numba/label/dev pixie 
```

(Note: `numba/label/dev` required for the dev version of `llvmlite`)

To work with C source code, including Cython, you need clang=14

```bash
% conda install clang=14
```

To work with Numba code, you need the latest dev build of Numba

```bash
% conda install -c numba/label/dev numba
```
