from collections import defaultdict, namedtuple
import cloudpickle
import ctypes
import io
import os
import pprint
import sys
import tempfile
import types as pytypes
from dataclasses import dataclass
from llvmlite import binding as llvm
from llvmlite import ir
from llvmlite.binding import Linkage

from pixie.platform import Toolchain
from pixie.codegen_helpers import Context, Codegen
from pixie.compiler_lock import compiler_lock
from pixie import types
from pixie import llvm_types as lt

ll_input = namedtuple('ll_input', 'symbol_name signature llvm_ir')

NULL = ir.Constant(lt._void_star, None)
ZERO = ir.Constant(lt._int32, 0)
ONE = ir.Constant(lt._int32, 1)
METH_VARARGS_AND_KEYWORDS = ir.Constant(lt._int32, 1 | 2)

# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/cc.py#L19

class PIXIECompiler(object):  # cf. CC.

    def __init__(self, extension_name, output_dir=None):
        self._extension_name = extension_name
        self._exported_functions = dict()
        self._basename = extension_name
        # Resolve source module name and directory
        f = sys._getframe(1)
        dct = f.f_globals
        if output_dir is None:
            self._source_path = dct.get('__file__', '')
            # By default, output in directory of caller module
            self._output_dir = os.path.dirname(self._source_path)
        else:
            assert os.path.isdir(output_dir)
            self._output_dir = output_dir
        self._toolchain = Toolchain()
        self._output_file = self._toolchain.get_ext_filename(extension_name)
        self._verbose = False
        self._function_table = defaultdict(list)

        self._target_cpu = ''

    def add_function(self, python_name, symbol_name, signature, llvm_ir):
        # check this signature
        sig = types.Signature(signature)
        data = ll_input(symbol_name,
                        sig,
                        llvm_ir)
        self._function_table[python_name].append(data)
        self._exported_functions[symbol_name] = data

    def __repr__(self):
        with io.StringIO() as buf:
            pprint.pprint(self._function_table, stream=buf)
            return buf.getvalue()

    @property
    def _export_entries(self):
        return sorted(self._exported_functions.values(),
                      key=lambda entry: entry.symbol_name)

    def _get_extra_ldflags(self):
        return ()

    @compiler_lock
    def _compile_object_files(self, build_dir):
        compiler = ModuleCompiler(self._export_entries, self._basename,
                                  self._extension_name, self._function_table,
                                  cpu_name=self._target_cpu)
        temp_obj = os.path.join(build_dir,
                                os.path.splitext(self._output_file)[0] + '.o')
        compiler.write_native_object(temp_obj, wrap=True)
        return [temp_obj], compiler.dll_exports

    @compiler_lock  # cf. CC.compile
    def compile_ext(self):
        """
        Compile the extension module.
        """
        self._toolchain.verbose = True
        prefix = f'pixie-build-{self._basename}-'
        with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:

            # Compile object file
            objects, dll_exports = self._compile_object_files(build_dir)

            # Then create shared library
            extra_ldflags = self._get_extra_ldflags()
            output_dll = os.path.join(self._output_dir, self._output_file)
            libraries = self._toolchain.get_python_libraries()
            library_dirs = self._toolchain.get_python_library_dirs()
            self._toolchain.link_shared(output_dll, objects,
                                        libraries, library_dirs,
                                        export_symbols=dll_exports,
                                        extra_ldflags=extra_ldflags)


# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L80

class _ModuleCompiler(object):
    """A base class to compile Python modules to a single shared library or
    extension module.

    :param export_entries: a list of ExportEntry instances.
    :param module_name: the name of the exported module.
    """

    #: Structure used to describe a method of an extension type.
    #: struct PyMethodDef {
    #:     const char  *ml_name;       /* The name of the built-in
    #:                                    function/method */
    #:     PyCFunction  ml_meth;       /* The C function that implements it */
    #:     int          ml_flags;      /* Combination of METH_xxx flags, which
    #:                                    mostly describe the args expected by
    #:                                    the C func */
    #:     const char  *ml_doc;        /* The __doc__ attribute, or NULL */
    #: };
    method_def_ty = ir.LiteralStructType((lt._int8_star,
                                          lt._void_star,
                                          lt._int32,
                                          lt._int8_star))

    method_def_ptr = ir.PointerType(method_def_ty)

    _DEBUG = False

    def __init__(self, export_entries, module_name, extension_name,
                 _function_table, **kwargs):

        self._extension_name = extension_name
        self._toolchain = Toolchain()
        self._target_cpu = ""  # get from kwargs
        self.dll_exports = []
        self._codegen = Codegen(self._extension_name)
        self.module_name = module_name
        self.export_entries = export_entries
        self.context = Context()
        self._function_table = _function_table

    def _emit_python_wrapper(self, llvm_module):
        """Emit generated Python wrapper and extension module code.
        """
        raise NotImplementedError

    def write_llvm_bitcode(self, output, wrap=False, **kws):
        self.export_python_wrap = wrap
        library = self._perform_export_of_entries()
        with open(output, 'wb') as fout:
            fout.write(library.emit_bitcode())

    def write_native_object(self, output, wrap=False, **kws):
        self.export_python_wrap = wrap
        library = self._perform_export_of_entries()
        with open(output, 'wb') as fout:
            fout.write(library.emit_native_object())

    @compiler_lock
    def _perform_export_of_entries(self):
        """Exports all the defined entries.
        """
        pymod_library = self._codegen.create_library(self.module_name)

        # Generate IR for all exported functions
        for entry in self.export_entries:
            # Outline bitcode as const
            bcmod = self._codegen.create_library("_bitcode" + self.module_name)
            mod = llvm.parse_assembly(entry.llvm_ir)
            bcmod.add_llvm_module(mod)
            bcmod.finalize()
            bitcode = bcmod.emit_bitcode()
            ir_mod = ir.Module()
            bc_name = f'bitcode_for_{entry.symbol_name}'
            bitcode_const_bytes = self.context.insert_const_bytes(ir_mod,
                                                                  bitcode,
                                                                  bc_name)
            fnty = ir.FunctionType(lt._void_star, [])
            getter_name = f'get_bitcode_for_{entry.symbol_name}'
            fn = ir.Function(ir_mod, fnty, getter_name)
            bb = fn.append_basic_block()
            fn_builder = ir.IRBuilder(bb)
            fn_builder.ret(bitcode_const_bytes)

            fnty = ir.FunctionType(lt._int64, [])
            get_sz_name = f'get_bitcode_for_{entry.symbol_name}_size'
            fn = ir.Function(ir_mod, fnty, get_sz_name)
            bb = fn.append_basic_block()
            fn_builder = ir.IRBuilder(bb)
            fn_builder.ret(ir.Constant(lt._int64, len(bitcode)))

            pymod_library.add_ir_module(ir_mod)
            mod = llvm.parse_assembly(entry.llvm_ir)
            pymod_library.add_llvm_module(mod)
            llvm_func = pymod_library.get_function(entry.symbol_name)

            llvm_func.linkage = 'external'
            llvm_func.name = entry.symbol_name
            self.dll_exports.append(entry.symbol_name)

        if self.export_python_wrap:
            wrapper_module = pymod_library.create_ir_module("wrapper")
            self._emit_python_wrapper(wrapper_module)
            pymod_library.add_ir_module(wrapper_module)

        # Hide all functions in the DLL except those explicitly exported
        pymod_library.finalize()
        for fn in pymod_library.get_defined_functions():
            break  # TODO: Make this work correctly.
            if fn.name not in self.dll_exports:
                if fn.linkage in {Linkage.private, Linkage.internal}:
                    # Private/Internal linkage must have "default" visibility
                    fn.visibility = "default"
                else:
                    fn.visibility = 'hidden'
        if self._DEBUG:
            print(pymod_library.get_llvm_str())

        return pymod_library


class PyModuleDef_Slot(object):
    Py_mod_create = ir.IntType(32)(1)
    Py_mod_exec = ir.IntType(32)(2)


@dataclass
class c_signature:
    return_type: None
    argument_types: None


@dataclass
class pixie_info:
    signature: c_signature
    symbol_name: str  # mangled_name
    address: int  # address
    ctypes_wrapper: pytypes.NoneType = None
    c_call_wrapper: pytypes.NoneType = None
    bitcode: pytypes.NoneType = None

# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L308

class ModuleCompiler(_ModuleCompiler):

    def _ptr_fun(ret, *args):
        return ir.PointerType(ir.FunctionType(ret, args))

    #: typedef int (*visitproc)(PyObject *, void *);
    visitproc_ty = _ptr_fun(lt._int8,
                            lt._pyobject_head_p)

    #: typedef int (*inquiry)(PyObject *);
    inquiry_ty = _ptr_fun(lt._int8,
                          lt._pyobject_head_p)

    #: typedef int (*traverseproc)(PyObject *, visitproc, void *);
    traverseproc_ty = _ptr_fun(lt._int8,
                               lt._pyobject_head_p,
                               visitproc_ty,
                               lt._void_star)

    #  typedef void (*freefunc)(void *)
    freefunc_ty = _ptr_fun(lt._int8,
                           lt._void_star)

    # PyObject* (*m_init)(void);
    m_init_ty = _ptr_fun(lt._int8)

    _char_star = lt._int8_star

    #: typedef struct PyModuleDef_Base {
    #:   PyObject_HEAD
    #:   PyObject* (*m_init)(void);
    #:   Py_ssize_t m_index;
    #:   PyObject* m_copy;
    #: } PyModuleDef_Base;
    module_def_base_ty = ir.LiteralStructType(
        (
            lt._pyobject_head,
            m_init_ty,
            lt._llvm_py_ssize_t,
            lt._pyobject_head_p
        ))

    #: typedef struct PyModuleDef_Slot {
    #:    int slot;
    #:    void * value;
    #: } PyModuleDef_Slot
    py_module_def_slot_ty = ir.LiteralStructType(
            (
                lt._int32,
                lt._void_star,
            )
        )
    py_module_def_slot_ty_ptr = ir.PointerType(py_module_def_slot_ty)

    #: This struct holds all information that is needed to create a module
    #: object.
    #: typedef struct PyModuleDef{
    #:   PyModuleDef_Base m_base;
    #:   const char* m_name;
    #:   const char* m_doc;
    #:   Py_ssize_t m_size;
    #:   PyMethodDef *m_methods;
    #:   inquiry m_reload;
    #:   traverseproc m_traverse;
    #:   inquiry m_clear;
    #:   freefunc m_free;
    #: }PyModuleDef;
    module_def_ty = ir.LiteralStructType(
        (
            module_def_base_ty,
            _char_star,
            _char_star,
            lt._llvm_py_ssize_t,
            _ModuleCompiler.method_def_ptr,
            py_module_def_slot_ty_ptr,
            traverseproc_ty,
            inquiry_ty,
            freefunc_ty
        ))

    @property
    def module_create_definition(self):
        """
        Return the signature and name of the Python C API function to
        initialize the module.
        """
        signature = ir.FunctionType(lt._pyobject_head_p,
                                    (ir.PointerType(self.module_def_ty),
                                     lt._int32))

        name = "PyModule_Create2"
        if lt._trace_refs_:
            name += "TraceRefs"

        return signature, name

    @property
    def module_init_definition(self):
        """
        Return the name and signature of the module's initialization function.
        """
        signature = ir.FunctionType(lt._pyobject_head_p, ())

        return signature, "PyInit_" + self.module_name

    def _emit_method_array(self, llvm_module):
        """
        Emits a PyMethodDef array.
        :returns: a pointer to the PyMethodDef array.
        """
        method_defs = []
        sentinel = ir.Constant.literal_struct([NULL, NULL, ZERO, NULL])
        method_defs.append(sentinel)
        method_array_init = self.context.create_constant_array(
            self.method_def_ty, method_defs)
        method_array = self.context.add_global_variable(llvm_module,
                                                        method_array_init.type,
                                                        '.module_methods')
        method_array.initializer = method_array_init
        method_array.linkage = 'internal'
        method_array_ptr = ir.Constant.gep(method_array, [ZERO, ZERO])
        return method_array_ptr

    def _emit_mod_init_payload(self):

        def filter_exports():
            d = dict()
            for name, info in self._function_table.items():
                d[name] = defaultdict(dict)
                for decl in info:
                    table = dict()
                    d[name][decl.signature] = table
                    table['symbol'] = decl.symbol_name
                    table['llvm_ir'] = decl.llvm_ir
            return d

        magic = filter_exports()
        # sanity check
        cloudpickle.loads(cloudpickle.dumps(magic))

        def pixie_module_init_func(obj):

            def fish_bitcode(dso, symbol_name):
                sz_name = f"get_bitcode_for_{symbol_name}_size"
                sz_fptr = getattr(dso, sz_name)
                sz_fptr.argtypes = ()
                sz_fptr.restype = ctypes.c_long
                data_name = f"get_bitcode_for_{symbol_name}"
                data_fptr = getattr(dso, data_name)
                data_fptr.restype = ctypes.c_void_p
                data_fptr.argtypes = []
                bitcode = bytes((ctypes.c_char * sz_fptr()).from_address(
                    data_fptr()))
                return bitcode

            dso = ctypes.CDLL(obj.__file__)
            new_table = dict()
            for name, info in magic.items():
                new_table[name] = defaultdict(dict)
                for index, (sig, v) in enumerate(info.items()):
                    symbol_name = v['symbol']
                    ct_fptr = getattr(dso, symbol_name)
                    ctsig = sig.as_ctypes()
                    ct_fptr.restype = ctsig.return_type
                    ct_fptr.argtypes = ctsig.argument_types
                    address = ctypes.addressof(ct_fptr)
                    bitcode = fish_bitcode(dso, symbol_name)
                    new_table[name][index] = pixie_info(sig,
                                                        symbol_name,
                                                        address,
                                                        ctypes_wrapper=ct_fptr,
                                                        bitcode=bitcode)

            obj.__dict__.update(new_table)
        return pixie_module_init_func

    def _emit_module_bootstrap_function(self, llvm_module):
        def bootstrap_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        (lt._pyobject_head_p,))

            name = "module_bootstrap"
            return signature, name

        bootstrap_fn = ir.Function(llvm_module, *bootstrap_sig_type())
        bootstrap_fn.linkage = 'internal'
        entry = bootstrap_fn.append_basic_block('BootstrapEntry')
        builder = ir.IRBuilder(entry)

        pixie_module_init_func = self._emit_mod_init_payload()
        serialized_mod_init_func = cloudpickle.dumps(pixie_module_init_func)
        ir_mod = builder.module
        serialized_mod_init_func_bytes = self.context.insert_const_bytes(
            ir_mod, serialized_mod_init_func, '.bytes.pixie_module_init_func')

        # import cloudpickle
        def PyImport_ImportModule_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *
                                        (self._char_star,))

            name = "PyImport_ImportModule"
            return signature, name
        PyImport_ImportModule_fn = ir.Function(
            llvm_module, *PyImport_ImportModule_sig_type())
        PyImport_ImportModule_fn.linkage = 'external'
        pickle_str = self.context.insert_const_string(llvm_module,
                                                      'cloudpickle')
        pickle_mod = builder.call(PyImport_ImportModule_fn, (pickle_str,))
        # TODO: check return

        # get "loads" attr on pickle
        def PyObject_GetAttrString_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, char *
                                        (lt._pyobject_head_p,
                                         self._char_star,))
            name = "PyObject_GetAttrString"
            return signature, name

        PyObject_GetAttrString_fn = ir.Function(
            llvm_module, *PyObject_GetAttrString_sig_type())
        PyObject_GetAttrString_fn.linkage = 'external'
        pickle_loads_str = self.context.insert_const_string(
            llvm_module, 'loads')
        pickle_loads = builder.call(PyObject_GetAttrString_fn,
                                    (pickle_mod, pickle_loads_str,))

        # get payload as pybytes
        def PyBytes_FromStringAndSize_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *, size_t
                                        (self._char_star, lt._llvm_py_ssize_t))
            name = "PyBytes_FromStringAndSize"
            return signature, name
        PyBytes_FromStringAndSize_fn = ir.Function(
            llvm_module, *PyBytes_FromStringAndSize_sig_type())
        PyBytes_FromStringAndSize_fn.linkage = 'external'
        payload_bytes = builder.call(
            PyBytes_FromStringAndSize_fn,
            (serialized_mod_init_func_bytes,
             lt._llvm_py_ssize_t(len(serialized_mod_init_func)),))

        # call pickle loads on payload
        def PyObject_CallFunctionObjArgs_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, ...
                                        (lt._pyobject_head_p,), var_arg=True)
            name = "PyObject_CallFunctionObjArgs"
            return signature, name

        PyObject_CallFunctionObjArgs_fn = ir.Function(
            llvm_module, *PyObject_CallFunctionObjArgs_sig_type())
        PyObject_CallFunctionObjArgs_fn.linkage = 'external'
        payload = builder.call(PyObject_CallFunctionObjArgs_fn,
                               (pickle_loads, payload_bytes, NULL))

        # finally, execute the payload
        arg = bootstrap_fn.args[0]
        argslot = builder.alloca(arg.type)
        builder.store(arg, argslot)
        builder.call(PyObject_CallFunctionObjArgs_fn,
                     (payload, builder.load(argslot), NULL))

        # Done with loading
        builder.ret(ZERO)

        return bootstrap_fn

    def _emit_python_wrapper(self, llvm_module):
        # Figure out the Python C API module creation function, and
        # get a LLVM function for it.
        create_module_fn = ir.Function(llvm_module,
                                       *self.module_create_definition)
        create_module_fn.linkage = 'external'

        # Define a constant string for the module name.
        mod_name_const = self.context.insert_const_string(llvm_module,
                                                          self.module_name)

        mod_def_base_init = ir.Constant.literal_struct(
            (
                lt._pyobject_head_init,                        # PyObject_HEAD
                ir.Constant(self.m_init_ty, None),             # m_init
                ir.Constant(lt._llvm_py_ssize_t, None),        # m_index
                ir.Constant(lt._pyobject_head_p, None),        # m_copy
            )
        )
        mod_def_base = self.context.add_global_variable(llvm_module,
                                                        mod_def_base_init.type,
                                                        '.module_def_base')
        mod_def_base.initializer = mod_def_base_init
        mod_def_base.linkage = 'internal'

        # Method array, this isn't used at present but needs to exist.
        method_array = self._emit_method_array(llvm_module)

        # bootstrap function, this deserialises the payload that is used to
        # build the module in the mod_exec part of the init
        bootstrap_fn = self._emit_module_bootstrap_function(llvm_module)

        # put bootstrap function into a slot
        slotbootstrap = ir.Constant.literal_struct(
            [PyModuleDef_Slot.Py_mod_exec,
             ir.Constant.bitcast(bootstrap_fn, lt._void_star)])

        slot_defs = []
        # put the bootstrap slot into the slot definitions
        slot_defs.append(slotbootstrap)

        # sentinel slot
        slot_sentinel = ir.Constant.literal_struct([ZERO, NULL])
        slot_defs.append(slot_sentinel)

        # create slots
        slot_array_init = self.context.create_constant_array(
            self.py_module_def_slot_ty, slot_defs)
        slot_array = self.context.add_global_variable(llvm_module,
                                                      slot_array_init.type,
                                                      '.slots')
        slot_array.initializer = slot_array_init
        slot_array.linkage = 'internal'
        slot_array_ptr = ir.Constant.gep(slot_array, [ZERO, ZERO])

        mod_def_init = ir.Constant.literal_struct(
            (
                mod_def_base_init,                              # m_base
                mod_name_const,                                 # m_name
                ir.Constant(self._char_star, None),             # m_doc
                ir.Constant(lt._llvm_py_ssize_t, 0),            # m_size
                method_array,                                   # m_methods
                slot_array_ptr,                                 # m_slots
                ir.Constant(self.traverseproc_ty, None),        # m_traverse
                ir.Constant(self.inquiry_ty, None),             # m_clear
                ir.Constant(self.freefunc_ty, None)             # m_free
            )
        )

        # Define a constant string for the module name.
        mod_def = self.context.add_global_variable(llvm_module,
                                                   mod_def_init.type,
                                                   '.module_def')
        mod_def.initializer = mod_def_init
        mod_def.linkage = 'internal'

        # Define the module initialization function.
        mod_init_fn = ir.Function(llvm_module, *self.module_init_definition)
        entry = mod_init_fn.append_basic_block('ModInitFnEntry')
        builder = ir.IRBuilder(entry)

        def PyModuleDef_Init_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        (ir.PointerType(self.module_def_ty),))

            name = "PyModuleDef_Init"
            return signature, name

        PyModuleDef_Init_fn = ir.Function(llvm_module,
                                          *PyModuleDef_Init_sig_type())
        PyModuleDef_Init_fn.linkage = 'external'
        ret = builder.call(PyModuleDef_Init_fn, (mod_def,))
        builder.ret(ret)

        self.dll_exports.append(mod_init_fn.name)
