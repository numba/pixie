---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Compiling a PIXIE library from Cython

This notebook is building upon knowledge of [Compiling a PIXIE library from C](./simple_pixie_c_lib.md)

This notebook requires a working clang 14 compiler and cython. You can install them with `conda install clang=14 cython`.

+++

Building on the previous C library, the simplest way to make Python binding for `add_f64` is to use Cython. Here, we make `cy_add()` for use in Python code:

```{code-cell} ipython3
%%writefile simple_add_cython.pyx
cimport cython

cdef public void add_f64(double *x, double *y, double *out):
    out[0] = x[0] + y[0]

def cy_add(x, y):
    cdef cython.double out = 0.0
    cdef cython.double a = x
    cdef cython.double b = y
    add_f64(&a, &b, &out)
    return out
```

Next, use PIXIE to compile the Cython source file into a PIXIE library:

```{code-cell} ipython3
from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration
```

```{code-cell} ipython3
src = "simple_add_cython.pyx"
tus = [
    TranslationUnit.from_cython_source(src),
]
```

```{code-cell} ipython3
export_config = ExportConfiguration()
export_config.add_symbol(python_name='add_f64',
                         symbol_name='add_f64',
                         signature='void(double*, double*, double*)',)
compiler = PIXIECompiler(library_name='simple_add_cython', # name must match cython file
                         translation_units=tus,
                         export_configuration=export_config,
                         **get_default_configuration(),
                         python_cext=True,
                         output_dir='.')
compiler.compile()
```

Now that we have made a DSO with name `simple_add_cython` as a Python C-extension library, we can import it.

```{code-cell} ipython3
import simple_add_cython
```

The library is Cython module as expected with an export of `cy_add`:

```{code-cell} ipython3
help(simple_add_cython)
```

Output:

```
Help on module simple_add_cython:

NAME
    simple_add_cython

FUNCTIONS
    cy_add(x, y)

DATA
    __PIXIE__ = {'available_isas': ['v8_6a', 'v8_4a', 'baseline'], 'bitcod...
    __test__ = {}

FILE
    /path/to/simple_add_cython.cpython-311-darwin.so
```

```{code-cell} ipython3
simple_add_cython.cy_add(1, 2)
```

Output:

```
3.0
```

+++

It is also a PIXIE library, so it has the `__PIXIE__` attribute on the module:

```{code-cell} ipython3
pixie_dict = simple_add_cython.__PIXIE__
```

```{code-cell} ipython3
pixie_dict
```

Output:

```
{'symbols': {'add_f64': {'void(double*, double*, double*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
    'symbol': 'add_f64',
    'module': None,
    'source_file': None,
    'address': 4569974972,
    'cfunc': <CFunctionType object at 0x1110690d0>,
    'metadata': None}}},
 'c_header': ['<write it>'],
 'linkage': None,
 'bitcode': b'...[skipped]...',
 'uuid': '82506a0f-8d17-4e2a-a2db-e6760f507910',
 'is_specialized': False,
 'available_isas': ['v8_6a', 'v8_4a', 'baseline'],
 'specialize': <function specialize.<locals>.impl(baseline_cpu='host', baseline_features=None, targets_features=None)>,
 'selected_isa': 'v8_4a'}

```

+++

All the exported PIXIE symbols are accessible as shown previously in the C example.

```{code-cell} ipython3
pixie_dict['symbols']
```

Output:

```
{'add_f64': {'void(double*, double*, double*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
   'symbol': 'add_f64',
   'module': None,
   'source_file': None,
   'address': 4397500604,
   'cfunc': <CFunctionType object at 0x105fc8950>,
   'metadata': None}}}
```

```{code-cell} ipython3

```
