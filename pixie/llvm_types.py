# NOTE: This module is a copy of:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/llvm_types.py

import sys
import ctypes
import struct as struct_
import llvmlite.ir
from llvmlite.ir import Constant


_trace_refs_ = hasattr(sys, 'getobjects')
_plat_bits = struct_.calcsize('@P') * 8

_int8 = llvmlite.ir.IntType(8)
_int16 = llvmlite.ir.IntType(16)
_int32 = llvmlite.ir.IntType(32)
_int64 = llvmlite.ir.IntType(64)

_void_star = llvmlite.ir.PointerType(_int8)

_int8_star = _char_star = _void_star

_sizeof_py_ssize_t = ctypes.sizeof(getattr(ctypes, 'c_size_t'))
_llvm_py_ssize_t = llvmlite.ir.IntType(_sizeof_py_ssize_t * 8)
_llvm_py_ssize_t_star = llvmlite.ir.PointerType(_llvm_py_ssize_t)

if _trace_refs_:
    _pyobject_head = llvmlite.ir.LiteralStructType([_void_star, _void_star,
                                                    _llvm_py_ssize_t,
                                                    _void_star])
    _pyobject_head_init = Constant.literal_struct([
        Constant(_void_star, None),            # _ob_next
        Constant(_void_star, None),            # _ob_prev
        Constant(_llvm_py_ssize_t, 1),         # ob_refcnt
        Constant(_void_star, None),            # ob_type
        ])
else:
    _pyobject_head = llvmlite.ir.LiteralStructType([_llvm_py_ssize_t,
                                                    _void_star])
    _pyobject_head_init = Constant.literal_struct([
        Constant(_llvm_py_ssize_t, 1),    # ob_refcnt
        Constant(_void_star, None),       # ob_type
        ])

_pyobject_head_p = llvmlite.ir.PointerType(_pyobject_head)

_pyvarobject = llvmlite.ir.LiteralStructType([
    _pyobject_head,
    _llvm_py_ssize_t,
    ])

_pyvarobject_p = llvmlite.ir.PointerType(_pyvarobject)

_pyobject_var_head = _pyvarobject

# PyTypeObject is #defined to alias _typeobject in Include/pytypedefs.h
# the struct _typeobject is in include/cpython/object.h
_pytypeobject_head = llvmlite.ir.LiteralStructType([_pyobject_var_head,
                                                    _void_star])

_pytypeobject_head_p = llvmlite.ir.PointerType(_pytypeobject_head)
