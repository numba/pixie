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

# Compiling a PIXIE library from LLVM

This notebook is building upon knowledge of [Compiling a PIXIE library from C](./simple_pixie_c_lib.md)
and show how to build a PIXIE library from LLVM-IR.

PIXIE can directly ingest LLVM-IR such that any LLVM-emitting compiler frontend 
can build PIXIE libraries.

Let's start with some LLVM-IR as Python strings:

```{code-cell} ipython3
llvm_add_f64 = """
define void @add_f64(double* %".1", double* %".2", double* %".out")
{
entry:
    %.3 = load double, double * %.1
    %.4 = load double, double * %.2
    %"res" = fadd double %".3", %".4"
    store double %"res", double* %".out"
    ret void
}
"""

llvm_add_f32 = """
define void @add_f32(float* %".1", float* %".2", float* %".out")
{
entry:
    %.3 = load float, float * %.1
    %.4 = load float, float * %.2
    %"res" = fadd float %".3", %".4"
    store float %"res", float* %".out"
    ret void
}
"""
```

Next, use PIXIE to compile the LLVM-IR file into a PIXIE library:

```{code-cell} ipython3
from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.targets import get_default_configuration
```

```{code-cell} ipython3
src = "simple_add_llvm.ll"
tus = [
    # The only difference is here. 
    # The TranslationUnit takes the LLVM-IR string directly.
    TranslationUnit("llvm_add_f64", llvm_add_f64),
    TranslationUnit("llvm_add_f32", llvm_add_f32),
]
```

```{code-cell} ipython3
export_config = ExportConfiguration()
export_config.add_symbol(python_name='add',
                         symbol_name='add_f64',
                         signature='void(double*, double*, double*)',)
export_config.add_symbol(python_name='add',
                         symbol_name='add_f32',
                         signature='void(float*, float*, float*)',)
compiler = PIXIECompiler(library_name='simple_add_llvm', # name must match cython file
                         translation_units=tus,
                         export_configuration=export_config,
                         **get_default_configuration(),
                         python_cext=True,
                         output_dir='.')
compiler.compile()
```

Now that we have made a DSO with name `simple_add_llvm` as a Python C-extension library, we can import it and exercise the `add()` function via the `ctypes.CFUNCTYPE` in the `__PIXIE__`.

```{code-cell} ipython3
import simple_add_llvm
```

```{code-cell} ipython3
help(simple_add_llvm)
```

Output:

```
Help on module simple_add_llvm:

NAME
    simple_add_llvm

DATA
    __PIXIE__ = {'available_isas': ['v8_4a', 'v8_6a', 'baseline'], 'bitcod...

FILE
    /path/to/simple_add_llvm.cpython-311-darwin.so
```

```{code-cell} ipython3
pixie_dict = simple_add_llvm.__PIXIE__
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
