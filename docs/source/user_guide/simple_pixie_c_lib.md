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

# Compiling a PIXIE library from C 

This notebook requires a working clang 14 compiler. You can install it with `conda install clang=14`.

+++

First, write a simple C program; for example:

```{code-cell} ipython3
%%writefile simple_add.c

void add_f64(double *x, double *y, double *out) {
    *out = *x + *y;
}

void add_f32(float *x, float *y, float *out) {
    *out = *x + *y;
}
```

Next, use PIXIE to compile the C source file into a PIXIE library:

```{code-cell} ipython3
from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration
```

```{code-cell} ipython3
src = "simple_add.c"
tus = [
    TranslationUnit.from_c_source(src),
]
```

```{code-cell} ipython3
export_config = ExportConfiguration()
# Note that both C symbols are stored into the same Python name 
# such that it is building an overloaded function (like C++).
export_config.add_symbol(python_name='add',
                         symbol_name='add_f64',
                         signature='void(double*, double*, double*)',)
export_config.add_symbol(python_name='add',
                         symbol_name='add_f32',
                         signature='void(float*, float*, float*)',)
```

```{code-cell} ipython3
compiler = PIXIECompiler(library_name='my_c_example',
                         translation_units=tus,
                         export_configuration=export_config,
                         **get_default_configuration(),
                         python_cext=True,     # True to make a Python C-extension 
                         output_dir='.')
compiler.compile()
```

Now that we have made a DSO with name `my_c_example` as a Python C-extension library, we can import it.

```{code-cell} ipython3
import my_c_example
```

A PIXIE library has a special `__PIXIE__` attribute

```{code-cell} ipython3
pixie_dict = my_c_example.__PIXIE__
```

```{code-cell} ipython3
pixie_dict
```

Output:

```
{'symbols': {'add': {'void(double*, double*, double*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
    'symbol': 'add_f64',
    'module': None,
    'source_file': None,
    'address': 4435237788,
    'cfunc': <CFunctionType object at 0x10dca4dd0>,
    'metadata': None},
   'void(float*, float*, float*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
    'symbol': 'add_f32',
    'module': None,
    'source_file': None,
    'address': 4435237896,
    'cfunc': <CFunctionType object at 0x10dca4950>,
    'metadata': None}}},
 'c_header': ['<write it>'],
 'linkage': None,
 'bitcode': b'...[skipped]...',
 'uuid': '148a0d91-5fc6-4cb5-8c26-3533a584f82e',
 'is_specialized': False,
 'available_isas': ['v8_6a', 'v8_4a', 'baseline'],
 'specialize': <function specialize.<locals>.impl(baseline_cpu='host', baseline_features=None, targets_features=None)>,
 'selected_isa': 'v8_4a'}
```

+++

One of the entries is the list of symbols. We can get a Python callable as a `ctypes.CFUNCTYPE` for the `add()` function.

```{code-cell} ipython3
pixie_dict['symbols']
```

Output:
```
{'add': {'void(double*, double*, double*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
   'symbol': 'add_f64',
   'module': None,
   'source_file': None,
   'address': 4336196508,
   'cfunc': <CFunctionType object at 0x106513ad0>,
   'metadata': None},
  'void(float*, float*, float*)': {'ctypes_cfunctype': ctypes.CFUNCTYPE.<locals>.CFunctionType,
   'symbol': 'add_f32',
   'module': None,
   'source_file': None,
   'address': 4336196616,
   'cfunc': <CFunctionType object at 0x106513a10>,
   'metadata': None}}}
```

+++

list the signatures for the symbol `add`:

```{code-cell} ipython3
symbol_add = pixie_dict['symbols']['add']
signatures = list(symbol_add.keys())
signatures
```

Output:
```
['void(double*, double*, double*)', 'void(float*, float*, float*)']
```

```{code-cell} ipython3
from ctypes import c_double, byref

# Get the double precision definition.
add_cfunc = symbol_add[signatures[0]]['cfunc']
# Call the ctypes.CFUNCTYPE
out = c_double()
add_cfunc(byref(c_double(1.2)), byref(c_double(3.4)), byref(out))
print(out)
```

Output:

```
c_double(4.6)
```

```{code-cell} ipython3

```
