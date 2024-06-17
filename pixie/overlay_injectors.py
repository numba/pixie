# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L308
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L80
from llvmlite import ir
from pixie import llvm_types as lt
from pixie.codegen_helpers import Context, IRGenerator
from pixie import pyapi, types, overlay
from pixie.mcext import c
from types import MappingProxyType
from uuid import uuid4
import zlib
import json
from collections import defaultdict


NULL = ir.Constant(lt._void_star, None)
ZERO = ir.Constant(lt._int32, 0)
ONE = ir.Constant(lt._int32, 1)
MINUS_ONE = ir.Constant(lt._int32, -1)
METH_VARARGS_AND_KEYWORDS = ir.Constant(lt._int32, 1 | 2)


class PyModuleDef_Slot(object):
    Py_mod_create = ir.IntType(32)(1)
    Py_mod_exec = ir.IntType(32)(2)


class _OverlayGeneratorBase(IRGenerator):

    def __init__(self, module_name, uuid=None, syms=MappingProxyType({}),
                 available_isas=()):
        self.module_name = module_name
        self._uuid = uuid
        self._syms = syms
        self._available_isas = available_isas
        super().__init__()

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

    method_def_ptr_ty = ir.PointerType(method_def_ty)

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
    py_module_def_slot_ptr_ty = ir.PointerType(py_module_def_slot_ty)

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
            method_def_ptr_ty,
            py_module_def_slot_ptr_ty,
            traverseproc_ty,
            inquiry_ty,
            freefunc_ty
        ))

    @property
    def module_init_definition(self):
        """
        Return the name and signature of the module's initialization function.
        """
        signature = ir.FunctionType(lt._pyobject_head_p, ())

        return signature, "PyInit_" + self.module_name

    def if_unlikely(self, builder, pred):
        return builder.if_then(pred, likely=False)

    def is_null(self, builder, val):
        null = val.type(None)
        return builder.icmp_unsigned('==', null, val)

    def _create_payload(self):
        tmp = overlay.create_base_payload()
        if (mod_uuid := self._uuid) is None:
            mod_uuid = str(uuid4())
        tmp['__PIXIE__']['uuid'] = mod_uuid
        tmp['__PIXIE__']['available_isas'] = self._available_isas
        symbol_dict = tmp['__PIXIE__']['symbols']
        for python_name, defs in self._syms.items():
            symbol_dict[python_name] = defaultdict(dict)
            for d in defs:
                pixie_sig = types.Signature(d[1])
                cty_str = pixie_sig.as_ctypes_string()
                v = overlay.add_variant(ctypes_func_string=cty_str,
                                        raw_symbol=d[0],
                                        metadata=d[2])
                symbol_dict[python_name][d[1]] = v
        return tmp

    def _emit_mod_init_json_payload(self):
        payload = self._create_payload()
        return zlib.compress(bytes(json.dumps(payload), 'UTF-8'))

    def _emit_module_bootstrap_function(self, llvm_module):
        def bootstrap_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        (lt._pyobject_head_p,))

            name = "module_bootstrap"
            return signature, name

        # Create the bootstrap function, this is the function that does the
        # necessary bootstrapping to execute the function that was serialised
        # at compile time in the DSO.
        bootstrap_fn = ir.Function(llvm_module, *bootstrap_sig_type())
        bootstrap_fn.linkage = 'internal'
        entry = bootstrap_fn.append_basic_block('BootstrapEntry')
        builder = ir.IRBuilder(entry)

        ir_mod = builder.module

        # Write Python C-API functions into the module
        rawpy = pyapi.RawPyAPI
        PyImport_ImportModule_fn = rawpy.PyImport_ImportModule(llvm_module)
        PyObject_GetAttrString_fn = rawpy.PyObject_GetAttrString(llvm_module)
        PyBytes_FromStringAndSize_fn = rawpy.PyBytes_FromStringAndSize(
            llvm_module)
        PyObject_CallFunctionObjArgs_fn = rawpy.PyObject_CallFunctionObjArgs(
            llvm_module)
        Py_BuildValue_fn = rawpy.Py_BuildValue(llvm_module)
        PyDict_GetItemString_fn = rawpy.PyDict_GetItemString(llvm_module)
        PyDict_SetItemString_fn = rawpy.PyDict_SetItemString(llvm_module)
        PyRun_String_fn = rawpy.PyRun_String(llvm_module)
        PyUnicode_AsUTF8AndSize_fn = rawpy.PyUnicode_AsUTF8AndSize(llvm_module)

        def check_null_result(result):
            with self.if_unlikely(builder, self.is_null(builder, result)):
                builder.ret(MINUS_ONE)

        def check_call(function, args):
            tmp = builder.call(function, args)
            if function.return_value.type.is_pointer:
                check_null_result(tmp)
                return tmp
            elif tmp.type == lt._int32:
                # need to check the return type is inst
                pred = builder.icmp_unsigned('!=', tmp, tmp.type(0))
                with self.if_unlikely(builder, pred):
                    builder.ret(tmp)
            else:
                raise ValueError(f"Result type cannot be handled {tmp.type}")

        def const_str(string):
            return self.context.insert_const_string(llvm_module, string)

        # TODO: this section needs references working out and appropriate
        # checks/clean up adding.

        # get "loads" attr on json mmodule
        json_str = const_str('json')
        json_mod = check_call(PyImport_ImportModule_fn, (json_str,))
        json_loads_str = const_str('loads')
        json_loads = check_call(PyObject_GetAttrString_fn,
                                (json_mod, json_loads_str,))

        # decompress attr on zlib
        zlib_str = const_str('zlib')
        zlib_mod = check_call(PyImport_ImportModule_fn, (zlib_str,))
        zlib_decompress_str = const_str('decompress')
        zlib_decompress = check_call(PyObject_GetAttrString_fn,
                                     (zlib_mod, zlib_decompress_str,))

        # get compressed payload function as pybytes
        serialized_mod_init_func = self._emit_mod_init_json_payload()
        serialized_mod_init_func_bytes = self.context.insert_const_bytes(
            ir_mod, serialized_mod_init_func, '.bytes.pixie_module_init_func')
        nbytes = len(serialized_mod_init_func)
        payload_bytes = check_call(PyBytes_FromStringAndSize_fn,
                                   (serialized_mod_init_func_bytes,
                                    lt._llvm_py_ssize_t(nbytes),))

        # call zlib decompress on payload
        decompressed_payload = check_call(PyObject_CallFunctionObjArgs_fn,
                                          (zlib_decompress, payload_bytes,
                                           NULL))

        payload = check_call(PyObject_CallFunctionObjArgs_fn,
                             (json_loads, decompressed_payload, NULL))

        # Run a trivial dict ctor to just get some empty dictionaries for us
        # as globals/locals
        empty_dict_str = const_str('{}')
        ldict = check_call(Py_BuildValue_fn, (empty_dict_str,))
        gdict = check_call(Py_BuildValue_fn, (empty_dict_str,))

        # get out payload['__PIXIE_assemblers__']['main']
        __PIXIE_assemblers__str = const_str('__PIXIE_assemblers__')
        __PIXIE_assemblers__ = check_call(PyDict_GetItemString_fn, (payload,
                                          __PIXIE_assemblers__str))

        main_str = const_str('main')
        main = check_call(PyDict_GetItemString_fn, (__PIXIE_assemblers__,
                                                    main_str,))

        # need main as const char *, see _Py_SourceAsString
        size = builder.alloca(lt._llvm_py_ssize_t)
        builder.store(ir.Constant(lt._llvm_py_ssize_t, None), size)  # null
        main_const_str = check_call(PyUnicode_AsUTF8AndSize_fn, (main, size))

        # Need to exec the string in main
        # From:
        # https://github.com/python/cpython/blob/238efbecab24204f822b1d1611914f5bcb2ae2de/Include/compile.h#L9
        # #define Py_file_input 257
        Py_file_input = lt._int32(257)
        # TODO: what to do with the result of this call?
        main_fn_result = check_call(PyRun_String_fn, (main_const_str,  # noqa: F841, E501
                                                      Py_file_input,
                                                      ldict,
                                                      gdict,))

        # Put the "payload" function into the same dict as function "main"
        payload_str = const_str('payload')
        check_call(PyDict_SetItemString_fn, (gdict, payload_str, payload))

        # Get the bootstrap function string out of the assemblers
        bootstrap_str = const_str('bootstrap')
        bootstrap = check_call(PyDict_GetItemString_fn,
                               (__PIXIE_assemblers__, bootstrap_str,))
        bootstrap_const_str = check_call(PyUnicode_AsUTF8AndSize_fn,
                                         (bootstrap, size))

        # exec the bootstrap function string, the dict above "gdict" has the
        # locals of "main" and "payload" in it which the bootstrap closes over
        # TODO: what to do with the result of this call?
        bootstrap_fn_result = check_call(PyRun_String_fn, (bootstrap_const_str,  # noqa: F841, E501
                                                           Py_file_input,
                                                           gdict,
                                                           ldict,))

        # Get the internal bootstrapping function out of the exec'd env
        internal_bootstrap_fn = check_call(PyDict_GetItemString_fn,
                                           (ldict, bootstrap_str,))

        # finally, execute the payload by calling the internal bootstrapping
        # function
        arg = bootstrap_fn.args[0]
        argslot = builder.alloca(arg.type)
        builder.store(arg, argslot)
        # TODO: what to do with the result of this call?
        result = check_call(PyObject_CallFunctionObjArgs_fn,  # noqa: F841
                            (internal_bootstrap_fn, builder.load(argslot),
                             NULL))

        # Done with loading
        builder.ret(ZERO)

        return bootstrap_fn


class AddPixieDictGenerator(_OverlayGeneratorBase):
    """"""

    _DEBUG = False

    def __init__(self, module_name, syms, uuid=None, available_isas=()):
        super().__init__(module_name, uuid=uuid, syms=syms,
                         available_isas=available_isas)
        self.context = Context()

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

    def generate_ir(self, llvm_module):
        # Figure out the Python C API module creation function, and
        # get a LLVM function for it.

        # Define a constant string for the module name.
        mod_name_const = self.context.insert_const_string(llvm_module,
                                                          self.module_name)

        mod_def_base_init = ir.Constant(self.module_def_base_ty,
                                        # PyObject_HEAD
                                        (lt._pyobject_head_init,
                                         # m_init
                                         ir.Constant(self.m_init_ty, None),
                                         # m_index
                                         ir.Constant(lt._llvm_py_ssize_t, None),
                                         # m_copy
                                         ir.Constant(lt._pyobject_head_p, None),
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
        slotbootstrap = ir.Constant(self.py_module_def_slot_ty,
                                    (PyModuleDef_Slot.Py_mod_exec,
                                     ir.Constant.bitcast(bootstrap_fn,
                                                         lt._void_star)))

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

        mod_def_init = ir.Constant(self.module_def_ty,
                                   (  # m_base
                                    mod_def_base_init,
                                    # m_name
                                    mod_name_const,
                                    # m_doc
                                    ir.Constant(self._char_star, None),
                                    # m_size
                                    ir.Constant(lt._llvm_py_ssize_t, 0),
                                    # m_methods
                                    method_array,
                                    # m_slots
                                    slot_array_ptr,
                                    # m_traverse
                                    ir.Constant(self.traverseproc_ty, None),
                                    # m_clear
                                    ir.Constant(self.inquiry_ty, None),
                                    # m_free
                                    ir.Constant(self.freefunc_ty, None)
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


class AugmentingPyInitGenerator(_OverlayGeneratorBase):

    _DEBUG = False

    def __init__(self, module_name, embedded_libhandle_name, syms, uuid=None,
                 available_isas=()):
        super().__init__(module_name, syms=syms, uuid=uuid,
                         available_isas=available_isas)
        self._embedded_libhandle_name = embedded_libhandle_name
        self.context = Context()

    def generate_ir(self, llvm_module):
        # Create a PyInit_{module_name} function to do the work
        # In this function do the following:
        # dlsym to get PyInit_{module_name}
        # Call the init function
        # See if the thing being returned is a PyModuleDef type
        # if it is then do the m_slots augment thing
        # if it is not, then do the obj thing
        mod_init_fn_sig, mod_init_fn_name = self.module_init_definition
        mod_init_fn = ir.Function(llvm_module, mod_init_fn_sig,
                                  mod_init_fn_name)
        entry = mod_init_fn.append_basic_block('ModInitFnEntry')
        builder = ir.IRBuilder(entry)

        # dlsym to get PyInit_{module_name}
        handle = llvm_module.get_global(self._embedded_libhandle_name)
        dso = builder.load(handle)
        const_symbol_name = self.context.insert_const_string(llvm_module,
                                                             mod_init_fn_name)
        sym = c.dlfcn.dlsym(builder, dso, const_symbol_name)

        # Call the init function
        pymod = builder.call(builder.bitcast(sym, mod_init_fn.type), ())

        # See if the thing being returned is a PyModuleDef type
        PyModuleDef_Type_glbl = \
            self.context.add_global_variable(llvm_module,
                                             lt._pytypeobject_head,
                                             "PyModuleDef_Type")

        ob_type = lt._int32(3) if lt._trace_refs_ else lt._int32(1)
        opaque_pymod_type = builder.load(builder.gep(pymod,
                                                     [lt._int32(0), ob_type]))
        pymod_type = builder.bitcast(opaque_pymod_type, lt._pytypeobject_head_p)
        is_mod_def_type = builder.icmp_unsigned("==",
                                                pymod_type,
                                                PyModuleDef_Type_glbl,
                                                )

        rawpy = pyapi.RawPyAPI
        PyType_IsSubtype_fn = rawpy.PyType_IsSubtype(llvm_module)
        is_subtype_mod_def_type = builder.call(PyType_IsSubtype_fn,
                                               (pymod_type,
                                                PyModuleDef_Type_glbl))

        if self._DEBUG:
            self.context.printf(builder, "is_subtype_mod_def_type = %d\n",
                                is_subtype_mod_def_type)
        is_module_def = builder.or_(is_mod_def_type,
                                    builder.trunc(is_subtype_mod_def_type,
                                                  is_mod_def_type.type))

        bootstrap_fn = self._emit_module_bootstrap_function(llvm_module)

        ret = builder.alloca(pymod.type)

        with builder.if_else(is_module_def) as (then, otherwise):
            with then:
                if self._DEBUG:
                    self.context.printf(builder, "Is module def\n")
                # augment slots
                MAX_SLOTS = 32
                slot_array_ty = ir.ArrayType(self.py_module_def_slot_ty,
                                             MAX_SLOTS)
                new_slots = self.context.add_global_variable(llvm_module,
                                                             slot_array_ty,
                                                             'new_slots')
                # zero init, the sentinel is {0, NULL};
                new_slots.initializer = ir.Constant(slot_array_ty, None)
                # cast the module object * to a mod def * type
                moddef = builder.bitcast(pymod, self.module_def_ty.as_pointer())
                # wire the new slots into the mod def slots location
                m_slots = builder.gep(moddef, [ZERO, ir.Constant(lt._int32, 5)])

                old_slots0 = builder.load(m_slots)

                ll_MAX_SLOTS = ir.Constant(lt._int32, MAX_SLOTS)
                n_used_slots = builder.alloca(lt._int32)
                i32_zero = ir.Constant(lt._int32, 0)
                i32_one = ir.Constant(lt._int32, 1)
                builder.store(i32_zero, n_used_slots)
                # walk the m_slots
                with self.context.for_range(builder, ll_MAX_SLOTS) as loop:
                    # check the slot type of the m_slots.slot, if it's zero,
                    # the sentinel has been hit and all the slots have been
                    # walked
                    old_slots_slot = builder.gep(old_slots0, [loop.index, ZERO])
                    pred = builder.icmp_signed("==",
                                               builder.load(old_slots_slot),
                                               ir.Constant(old_slots_slot.type,
                                                           0))
                    with builder.if_then(pred):
                        loop.do_break()

                    # otherwise, make the copy, this is just direct assignment
                    # from a static array that's a member of a static struct in
                    # the embedded module into a local static array
                    new_slots0 = builder.gep(new_slots, [ZERO, loop.index])
                    new_slots00 = builder.gep(new_slots0, [ZERO, ZERO])
                    new_slots01 = builder.gep(new_slots0, [ZERO, ONE])

                    old_slots00 = builder.gep(old_slots0, [loop.index, ZERO])
                    old_slots01 = builder.gep(old_slots0, [loop.index, ONE])

                    builder.store(builder.load(old_slots00), new_slots00)
                    builder.store(builder.load(old_slots01), new_slots01)

                    n_used_slots_plusplus = builder.add(
                        builder.load(n_used_slots), i32_one)
                    builder.store(n_used_slots_plusplus, n_used_slots)

                # TODO: This needs something like this here:
                # if (used_slots == MAX_SLOTS) {
                #     PyErr_SetString(PyExc_RuntimeError,
                #                     "Embedded module has more than 32 slots");
                #     return NULL;
                # }

                # write to slot in n_used_slots, this is one past the last
                # copied slot
                IDX = builder.load(n_used_slots)
                new_slots2 = builder.gep(new_slots, [ZERO, IDX])
                new_slots20 = builder.gep(new_slots2, [ZERO, ZERO])
                new_slots21 = builder.gep(new_slots2, [ZERO, ONE])
                builder.store(PyModuleDef_Slot.Py_mod_exec, new_slots20)
                builder.store(builder.bitcast(bootstrap_fn, lt._void_star),
                              new_slots21)

                # write over the existing slots in the module def with the
                # local static array
                builder.store(builder.bitcast(new_slots, m_slots.type.pointee),
                              m_slots)
                builder.store(pymod, ret)
            with otherwise:
                if self._DEBUG:
                    self.context.printf(builder, "Is module only\n")
                # its already a module instance, just inject the payload.
                builder.call(bootstrap_fn, (pymod,))
                builder.store(pymod, ret)

        builder.ret(builder.load(ret))
