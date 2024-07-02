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

## ``__PIXIE__``

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

## Get Symbols

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

+++

## Call exported symbols

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

+++

## Inspect available ISAs

A key feature in PIXIE is compiling for architecture variants. On MacOS arm64, PIXIE specializes for the ARM v8.4a and v8.6a profiles, which matches Apple M1 and Apple M2, respectively. 

```{code-cell} ipython3
print("Available ISAs:", pixie_dict['available_isas'])
print("Selected ISA:", pixie_dict['selected_isa'])
```

Output (Ran on a M1):

```
Available ISAs: ['v8_4a', 'v8_6a', 'baseline']
Selected ISA: v8_4a
```

+++

## Advanced: Using the LLVM bitcode

Another key feature in PIXIE is the embedded LLVM bitcode. We can retrieve the bitcode with llvmlite as follows:

```{code-cell} ipython3
from llvmlite import binding as llvm

mod_add = llvm.parse_bitcode(pixie_dict['bitcode'])
print(mod_add)
```

Output:

```llvm
source_filename = "simple_add.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn
define void @add_f64(double* nocapture noundef readonly %0, double* nocapture noundef readonly %1, double* nocapture noundef writeonly %2) local_unnamed_addr #0 {
  %4 = load double, double* %0, align 8, !tbaa !10
  %5 = load double, double* %1, align 8, !tbaa !10
  %6 = fadd double %4, %5
  store double %6, double* %2, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn
define void @add_f32(float* nocapture noundef readonly %0, float* nocapture noundef readonly %1, float* nocapture noundef writeonly %2) local_unnamed_addr #0 {
  %4 = load float, float* %0, align 4, !tbaa !14
  %5 = load float, float* %1, align 4, !tbaa !14
  %6 = fadd float %4, %5
  store float %6, float* %2, align 4, !tbaa !14
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"branch-target-enforcement", i32 0}
!2 = !{i32 1, !"sign-return-address", i32 0}
!3 = !{i32 1, !"sign-return-address-all", i32 0}
!4 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 1, !"Code Model", i32 1}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{!"clang version 14.0.6"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !12, i64 0}
```

+++

We can make a separate PIXIE library that references the `add_f64` function and have it optimized with the embedded bitcode.

+++

Here's a C program that references `add_f64` in a loop:

```{code-cell} ipython3
%%writefile simple_add_loop.c

/* external reference */
void add_f64(double *x, double *y, double *out);

void loop_add_f64(double *x, double *y, double *out, int size) {
    if (out != x && out != y) return;
    for (int i=0; i<size; ++i)
        add_f64(x, y, out);
}
```

Compile it into a PIXIE library.

```{code-cell} ipython3
src = "simple_add_loop.c"
tus = [
    TranslationUnit.from_c_source(src),
]
export_config = ExportConfiguration()
# Note that both C symbols are stored into the same Python name 
# such that it is building an overloaded function (like C++).
export_config.add_symbol(python_name='loop_add',
                         symbol_name='loop_add_f64',
                         signature='void(double*, double*, double*)',)
compiler = PIXIECompiler(library_name='my_c_loop',
                         translation_units=tus,
                         export_configuration=export_config,
                         **get_default_configuration(),
                         python_cext=True,     # True to make a Python C-extension 
                         output_dir='.')
compiler.compile()
```

We will again get the bitcode from the new PIXIE library.

```{code-cell} ipython3
import my_c_loop
mod_loop = llvm.parse_bitcode(my_c_loop.__PIXIE__['bitcode'])
print(mod_loop)
```

Output:

```llvm
source_filename = "simple_add_loop.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nounwind ssp uwtable
define void @loop_add_f64(double* noundef %0, double* noundef %1, double* noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq double* %2, %0
  %6 = icmp eq double* %2, %1
  %7 = or i1 %5, %6
  %8 = icmp sgt i32 %3, 0
  %9 = and i1 %7, %8
  br i1 %9, label %10, label %14

10:                                               ; preds = %10, %4
  %11 = phi i32 [ %12, %10 ], [ 0, %4 ]
  call void @add_f64(double* noundef %0, double* noundef %1, double* noundef %2) #2
  %12 = add nuw nsw i32 %11, 1
  %13 = icmp eq i32 %12, %3
  br i1 %13, label %14, label %10, !llvm.loop !10

14:                                               ; preds = %10, %4
  ret void
}

declare void @add_f64(double* noundef, double* noundef, double* noundef) local_unnamed_addr #1

attributes #0 = { nounwind ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"branch-target-enforcement", i32 0}
!2 = !{i32 1, !"sign-return-address", i32 0}
!3 = !{i32 1, !"sign-return-address-all", i32 0}
!4 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 1, !"Code Model", i32 1}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{!"clang version 14.0.6"}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.unroll.disable"}
```

+++

We can then link the two LLVM modules. Observe that the post-link module contains a reference to `add_f64()` in `loop_add_f64()` but it doesn't inline the definition.

```{code-cell} ipython3
mod_loop.link_in(mod_add)
print(mod_loop)  # Print the post-link module
```

Output:

```llvm
source_filename = "simple_add_loop.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nounwind ssp uwtable
define void @loop_add_f64(double* noundef %0, double* noundef %1, double* noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq double* %2, %0
  %6 = icmp eq double* %2, %1
  %7 = or i1 %5, %6
  %8 = icmp sgt i32 %3, 0
  %9 = and i1 %7, %8
  br i1 %9, label %10, label %14

10:                                               ; preds = %10, %4
  %11 = phi i32 [ %12, %10 ], [ 0, %4 ]
  call void @add_f64(double* noundef %0, double* noundef %1, double* noundef %2) #2
  %12 = add nuw nsw i32 %11, 1
  %13 = icmp eq i32 %12, %3
  br i1 %13, label %14, label %10, !llvm.loop !10

14:                                               ; preds = %10, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn
define void @add_f64(double* nocapture noundef readonly %0, double* nocapture noundef readonly %1, double* nocapture noundef writeonly %2) local_unnamed_addr #1 {
  %4 = load double, double* %0, align 8, !tbaa !13
  %5 = load double, double* %1, align 8, !tbaa !13
  %6 = fadd double %4, %5
  store double %6, double* %2, align 8, !tbaa !13
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn
define void @add_f32(float* nocapture noundef readonly %0, float* nocapture noundef readonly %1, float* nocapture noundef writeonly %2) local_unnamed_addr #1 {
  %4 = load float, float* %0, align 4, !tbaa !17
  %5 = load float, float* %1, align 4, !tbaa !17
  %6 = fadd float %4, %5
  store float %6, float* %2, align 4, !tbaa !17
  ret void
}

attributes #0 = { nounwind ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind ssp uwtable willreturn "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9, !9}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"branch-target-enforcement", i32 0}
!2 = !{i32 1, !"sign-return-address", i32 0}
!3 = !{i32 1, !"sign-return-address-all", i32 0}
!4 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 1, !"Code Model", i32 1}
!7 = !{i32 7, !"uwtable", i32 1}
!8 = !{i32 7, !"frame-pointer", i32 1}
!9 = !{!"clang version 14.0.6"}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !15, i64 0}
!15 = !{!"omnipotent char", !16, i64 0}
!16 = !{!"Simple C/C++ TBAA"}
!17 = !{!18, !18, i64 0}
!18 = !{!"float", !15, i64 0}
```

+++

Next, we can run the LLVM optimizer to inline the definition of `add_f64()`.

```{code-cell} ipython3
# Create and populate the optimizer
pm = llvm.create_module_pass_manager()
pmb = llvm.create_pass_manager_builder()
pmb.opt_level = 1
pmb.inlining_threshold = 200  # enable inlining
pmb.populate(pm)
# Run optimizer
pm.run(mod_loop)
# Print the IR of loop_add_f64
print(mod_loop.get_function('loop_add_f64'))
```

Output:

```llvm
; Function Attrs: nofree norecurse nosync nounwind ssp uwtable
define void @loop_add_f64(double* noundef readonly %0, double* noundef readonly %1, double* noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq double* %2, %0
  %6 = icmp eq double* %2, %1
  %7 = or i1 %5, %6
  %8 = icmp sgt i32 %3, 0
  %9 = and i1 %7, %8
  br i1 %9, label %.preheader, label %.loopexit

.preheader:                                       ; preds = %4, %.preheader
  %10 = phi i32 [ %14, %.preheader ], [ 0, %4 ]
  %11 = load double, double* %0, align 8, !tbaa !10
  %12 = load double, double* %1, align 8, !tbaa !10
  %13 = fadd double %11, %12
  store double %13, double* %2, align 8, !tbaa !10
  %14 = add nuw nsw i32 %10, 1
  %15 = icmp eq i32 %14, %3
  br i1 %15, label %.loopexit, label %.preheader, !llvm.loop !14

.loopexit:                                        ; preds = %.preheader, %4
  ret void
}
```

```{code-cell} ipython3

```
