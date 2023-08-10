from collections import defaultdict, namedtuple
import io
import os
import pprint
import sys
import tempfile
from llvmlite import binding as llvm
from llvmlite import ir
from llvmlite.binding import Linkage
import json
import zlib
import uuid

from pixie.platform import Toolchain
from pixie.codegen_helpers import Context, Codegen, _module_pass_manager
from pixie.compiler_lock import compiler_lock
from pixie.selectors import PyVersionSelector, x86CPUSelector
from pixie.dso_tools import ElfMapper, shmEmbeddedDSOHandler
from pixie import types, cpus
from pixie import llvm_types as lt
from pixie import overlay, pyapi


NULL = ir.Constant(lt._void_star, None)
ZERO = ir.Constant(lt._int32, 0)
ONE = ir.Constant(lt._int32, 1)
MINUS_ONE = ir.Constant(lt._int32, -1)
METH_VARARGS_AND_KEYWORDS = ir.Constant(lt._int32, 1 | 2)


_pixie_export = namedtuple('_pixie_export',
                           'symbol_name python_name signature metadata')


class TranslationUnit():
    # Maps to a LLVM module... Probably needs to to be consistent with C.

    def __init__(self, name, source):
        self._name = name
        self._source = source
        if isinstance(source, str):
            self._mod = llvm.parse_assembly(source)
        elif isinstance(source, bytes):
            self._mod = llvm.parse_bitcode(source)
        else:
            msg = "Expected string or bytes for source, got '{type(source)}'."
            raise TypeError(msg)


class ExportConfiguration():

    def __init__(self, versioning_strategy='embed_dso'):
        if versioning_strategy != 'embed_dso':
            msg = "The only supported versioning strategy is 'embed_dso'"
            raise ValueError(msg)
        # strategy might be one of: 'privatise' | 'embed_dso' | 'trampoline'
        self.versioning_strategy = versioning_strategy
        self._data = []

    def add_symbol(self, symbol_name, python_name, signature, metadata=None):
        self._data.append(_pixie_export(symbol_name, python_name, signature,
                                        metadata))


# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/cc.py#L19
class PIXIECompiler(object):  # cf. CC.

    def __init__(self, library_name, translation_units, export_configuration,
                 baseline_cpu, baseline_features, targets_features=(),
                 python_cext=True, uuid=None, output_dir="."):
        self._extension_name = library_name
        self._exported_functions = dict()
        self._basename = library_name
        self._uuid = uuid
        self._targets_features = targets_features
        if targets_features:
            # baseline must not be greater than any target, as such a target
            # could potentially never be run as there may be no set of
            # instructions capable of link loading it.
            if baseline_features > min(targets_features):
                # TODO check ISA enum is the same for all features, i.e. can't
                # mix x86 and ARM.'
                msg = (f"Baseline feature {baseline_features} exceeds minimum "
                       f"target feature ({min(targets_features)})")
                raise ValueError(msg)

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
        _output_file = self._toolchain.get_ext_filename(self._extension_name)
        self._output_file = _output_file
        self._verbose = False
        self._function_table = defaultdict(list)

        if baseline_features is None:
            self._baseline_features = ()
        else:
            self._baseline_features = baseline_features

        if baseline_cpu is None:
            self._baseline_features = llvm.get_host_cpu_name()
        else:
            self._baseline_cpu = baseline_cpu

        self._exported_symbols = defaultdict(list)
        llvm_irs = [x._source for x in translation_units
                    if isinstance(x._source, str)]
        llvm_bcs = [x._source for x in translation_units
                    if isinstance(x._source, bytes)]
        self._sources = {'llvm_ir': llvm_irs, 'llvm_bc': llvm_bcs}

        for d in export_configuration._data:
            self._exported_symbols[d.python_name].append((d.symbol_name,
                                                          d.signature,
                                                          d.metadata))

        self._python_cext = python_cext

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
    def _compile_feature_specific_dsos(self, build_dir, syms):
        # this returns a map of feature: dso compiled against that feature
        binaries = {}
        all_features = set(self._targets_features) | {self._baseline_features,}
        for feature in all_features:
            cpu_feature = cpus.Features(feature)
            bfeat = self._baseline_features
            compiler = FeatureSpecificCompiler(self._basename,
                                            self._extension_name,
                                            syms,
                                            self._sources,
                                            baseline_cpu=self._baseline_cpu,
                                            baseline_features=bfeat,
                                            features=cpu_feature)
            hashed = hash(tuple([hash(sym) for sym in syms]))
            hashed_name = f'{hashed}_{cpu_feature.as_selected_feature_flags}.o'
            obj_fname = os.path.splitext(self._output_file)[0] + hashed_name
            temp_obj = os.path.join(build_dir, obj_fname)
            compiler.write_native_object(temp_obj, wrap=True)
            # there's now an ISA specific object file at path temp_obj.
            objects = (temp_obj,)
            hashed_libname = f'{hashed}_{cpu_feature.as_selected_feature_flags}.so'
            extra_ldflags = self._get_extra_ldflags()
            output_dll = os.path.join(self._output_dir, hashed_libname)
            libraries = self._toolchain.get_python_libraries()
            library_dirs = self._toolchain.get_python_library_dirs()
            self._toolchain.link_shared(output_dll, objects,
                                        libraries, library_dirs,
                                        export_symbols=compiler.dll_exports,
                                        extra_ldflags=extra_ldflags)
            with open(output_dll, 'rb') as f:
                binaries[str(feature).upper()] = f.read()

        return binaries

    def _generate_dispatch(self, build_dir, syms, selector_class, dso_handler):
        mod = ir.Module()
        mod.triple = llvm.get_process_triple()
        emap = ElfMapper(mod)

        binaries = self._compile_feature_specific_dsos(build_dir, syms)
        binaries['baseline'] = binaries[min(binaries.keys())]
        # create the DSO constructor, it does the select and dispatch
        emap.create_dso_ctor(binaries, selector_class, dso_handler)
        # create the DSO destructor, it cleans up the resources used by the
        # create_dso_ctor.
        emap.create_dso_dtor(dso_handler)

        return mod, emap

    def create_fake_ifuncs(self, mod, embedded_libhandle_name):
        DEBUG = False
        ctx = Context()
        if DEBUG:
            def printf(builder, *args):
                ctx.printf(builder, *args)
        else:
            def printf(builder, *args):
                pass
        # creates the ifunc-like behaviour for the exported symbols
        _handle = mod.get_global(embedded_libhandle_name)
        for python_name, symsigs in self._exported_symbols.items():
            for symbol_name, sig, metadata in symsigs:
                pixie_sig = types.Signature(sig)
                fnty = pixie_sig.as_llvm_function_type()
                # create a global fnptr for the symbol
                fnty_as_ptr = fnty.as_pointer()
                # the dlsym return is just a void *
                void_ptr_ty = ir.IntType(8).as_pointer()
                fnptr_cache = ir.GlobalVariable(mod, void_ptr_ty,
                                                f"_fnptr_cache_for_{symbol_name}")
                # nullify on init, this is important state as the trampoline
                # function branches on the NULL.
                fnptr_cache.initializer = ir.Constant(fnptr_cache.type.pointee, None)

                # create a function that will trampoline
                trampoline_fn = ir.Function(mod, fnty,
                                            name=symbol_name)
                block = trampoline_fn.append_basic_block(name="entry")
                builder = ir.IRBuilder(block)

                fnptr_local_ref = builder.alloca(fnty_as_ptr)
                pred = builder.icmp_unsigned("==", builder.load(fnptr_cache), fnptr_cache.type(None))
                printf(builder, "predicate %d\n", pred)
                with builder.if_else(pred) as (then, otherwise):
                    with then:
                        printf(builder, "calling dlsym\n")
                        # find the symbol
                        dso = builder.load(_handle)
                        printf(builder, "dso is at %d\n", dso)
                        from pixie.mcext import c
                        const_symbol_name = ctx.insert_const_string(mod, symbol_name)
                        sym = c.dlfcn.dlsym(builder, dso, const_symbol_name)
                        printf(builder, "called dlsym, %d\n", sym)
                        builder.store(sym, fnptr_cache)
                        builder.store(builder.bitcast(sym, fnty_as_ptr), fnptr_local_ref)
                    with otherwise:
                        printf(builder, "replay from cache\n")
                        # store the dso global value into the slot
                        builder.store(builder.bitcast(builder.load(fnptr_cache), fnty_as_ptr), fnptr_local_ref)
                fn = builder.load(fnptr_local_ref)
                builder.call(fn, trampoline_fn.args)
                builder.ret_void()

    @compiler_lock
    def _compile_object_files(self, build_dir):
        # This compiles the numerous variants.
        objects = []
        dll_exports = []


        syms = []
        for python_name, symsigs in self._exported_symbols.items():
            for symbol_name, sig, metadata in symsigs:
                syms.append(symbol_name)

        selector_class = x86CPUSelector
        dso_handler = shmEmbeddedDSOHandler()
        mod, emap = self._generate_dispatch(build_dir, syms, selector_class, dso_handler)
        self.create_fake_ifuncs(mod, emap._embedded_libhandle_name)

        # Currently set this to empty
        feature = cpus.Features(())
        bfeat = self._baseline_features
        compiler = FeatureSpecificCompiler(self._basename,
                                           self._extension_name,
                                           syms,
                                           # self._sources,
                                           {'llvm_ir':[str(mod)],
                                            'llvm_bc':[]},
                                           baseline_cpu=self._baseline_cpu,
                                           baseline_features=bfeat,
                                           features=feature)
        hashed = hash(tuple([hash(sym) for sym in syms]))
        hashed_name = f'{hashed}_{feature.as_selected_feature_flags}.o'
        obj_fname = os.path.splitext(self._output_file)[0] + hashed_name
        temp_obj = os.path.join(build_dir, obj_fname)
        compiler.write_native_object(temp_obj, wrap=True)
        objects.append(temp_obj), dll_exports.extend(compiler.dll_exports)

        return objects, dll_exports

    @compiler_lock
    def _compile_cext_object_file(self, build_dir, baseline_cpu,
                                  baseline_features):

        compiler = ModuleCompiler(self._basename, self._extension_name,
                                  self._exported_symbols, self._sources,
                                  baseline_cpu=baseline_cpu,
                                  baseline_features=baseline_features,
                                  features=cpus.Features(()),
                                  uuid=self._uuid)
        obj_fname = os.path.splitext(self._output_file)[0] + '_meta.o'
        temp_obj = os.path.join(build_dir, obj_fname)
        compiler.write_native_object(temp_obj, wrap=True)
        return [temp_obj], compiler.dll_exports

    @compiler_lock  # cf. CC.compile
    def compile(self):
        """
        Compile the extension module.
        """
        # Set this to true to see the link line(s)
        self._toolchain.verbose = False

        prefix = f'pixie-build-{self._basename}-'
        with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:

            objects = []
            dll_exports = []

            # Compile object files for each feature set
            feature_objects, feature_dll_exports = \
                self._compile_object_files(build_dir)
            objects.extend(feature_objects)
            dll_exports.extend(feature_dll_exports)

            # Compile the PIXIE meta object that adds all the PIXIE specific
            # data into the DSO if the library needs to be a Python
            # C-Extension.
            if self._python_cext:
                meta_obj, meta_dll_exports = self._compile_cext_object_file(
                    build_dir, self._baseline_cpu, self._baseline_features)
                objects.extend(meta_obj)
                dll_exports.extend(meta_dll_exports)

            # Then create shared library
            extra_ldflags = self._get_extra_ldflags()
            output_dll = os.path.join(self._output_dir, self._output_file)
            libraries = self._toolchain.get_python_libraries()
            library_dirs = self._toolchain.get_python_library_dirs()
            self._toolchain.link_shared(output_dll, objects,
                                        libraries, library_dirs,
                                        export_symbols=dll_exports,
                                        extra_ldflags=extra_ldflags)


class _FeatureSpecificCompilerBase(object):
    _DEBUG = False

    def __init__(self, module_name, extension_name, syms, sources, **kwargs):

        self._extension_name = extension_name
        self._toolchain = Toolchain()
        self._target_cpu = kwargs.pop('baseline_cpu')  # get from kwargs
        self._target_features = kwargs.pop('features')  # get from kwargs
        self.dll_exports = []
        self._codegen = Codegen(self._extension_name,
                                cpu_name=self._target_cpu,
                                target_features=self._target_features)
        self.module_name = module_name
        self.context = Context()
        self._syms = syms
        self._sources = sources

    def write_llvm_bitcode(self, output, **kws):
        library = self._perform_export_of_entries()
        with open(output, 'wb') as fout:
            fout.write(library.emit_bitcode())

    def write_native_object(self, output, **kws):
        library = self._perform_export_of_entries()
        with open(output, 'wb') as fout:
            fout.write(library.emit_native_object())

    def _perform_export_of_entries(self):
        raise NotImplementedError

    def _combine_sources(self, lib):
        # TODO: add other handlers here
        for src in self._sources['llvm_ir']:
            mod = llvm.parse_assembly(src)
            lib.add_llvm_module(mod)

        for src in self._sources['llvm_bc']:
            mod = llvm.parse_bitcode(src)
            lib.add_llvm_module(mod)


class FeatureSpecificCompiler(_FeatureSpecificCompilerBase):
    def __init__(self, *args, **kwargs):
        super(FeatureSpecificCompiler, self).__init__(*args, **kwargs)

    @compiler_lock
    def _perform_export_of_entries(self):
        """Exports all the defined entries as mangled versions.
        """
        debug = False
        tmp_lib = self._codegen.create_library(f"_tmp_lib_{str(uuid.uuid4())}")
        # Combine the sources
        self._combine_sources(tmp_lib)

        pixie_lib = self._codegen.create_library(self.module_name)
        pixie_lib.add_llvm_module(tmp_lib._final_module)

        # mark the exports as external
        for name in self._syms:
            llvm_func = pixie_lib.get_function(name)
            llvm_func.linkage = 'external'
            self.dll_exports.append(name)

        if debug:
            print(pixie_lib.get_llvm_str())
        # run opt!
        mpm = _module_pass_manager(self._codegen._tm)
        mpm.run(pixie_lib._final_module)
        return pixie_lib


class PyModuleDef_Slot(object):
    Py_mod_create = ir.IntType(32)(1)
    Py_mod_exec = ir.IntType(32)(2)


# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L308
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/pycc/compiler.py#L80

class ModuleCompiler(_FeatureSpecificCompilerBase):
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
            method_def_ptr,
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

    _DEBUG = False

    def __init__(self, *args, **kwargs):
        self._uuid = kwargs.pop('uuid', None)
        super(ModuleCompiler, self).__init__(*args, **kwargs)

    @compiler_lock
    def _perform_export_of_entries(self):
        """Exports all the defined meta entries.
        """
        pymod_library = self._codegen.create_library(self.module_name)

        # sort out the bitcode for storage
        tmp_lib = self._codegen.create_library("tmp")
        self._combine_sources(tmp_lib)

        # force exported symbols as external linkage
        for python_name, symsigs in self._syms.items():
            for sym, sig, metadata in symsigs:
                llvm_func = tmp_lib.get_function(sym)
                llvm_func.linkage = 'external'

        # finalize
        tmp_lib.finalize()

        # Outline bitcode as const
        bitcode = tmp_lib.emit_bitcode()
        ir_mod = ir.Module()
        bc_name = f'bitcode_for_{self.module_name}'
        bitcode_const_bytes = self.context.insert_const_bytes(ir_mod,
                                                              bitcode,
                                                              bc_name)
        fnty = ir.FunctionType(lt._void_star, [])
        getter_name = f'get_bitcode_for_{self.module_name}'
        fn = ir.Function(ir_mod, fnty, getter_name)
        bb = fn.append_basic_block()
        fn_builder = ir.IRBuilder(bb)
        fn_builder.ret(bitcode_const_bytes)

        fnty = ir.FunctionType(lt._int64, [])
        get_sz_name = f'get_bitcode_for_{self.module_name}_size'
        fn = ir.Function(ir_mod, fnty, get_sz_name)
        bb = fn.append_basic_block()
        fn_builder = ir.IRBuilder(bb)
        fn_builder.ret(ir.Constant(lt._int64, len(bitcode)))

        pymod_library.add_ir_module(ir_mod)

        # Create the bootstrap wrapper
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

    def if_unlikely(self, builder, pred):
        return builder.if_then(pred, likely=False)

    def is_null(self, builder, val):
        null = val.type(None)
        return builder.icmp_unsigned('==', null, val)

    def _create_payload(self):
        tmp = overlay.create_base_payload()
        if (mod_uuid := self._uuid) is None:
            mod_uuid = str(uuid.uuid4())
        tmp['__PIXIE__']['uuid'] = mod_uuid
        symbol_dict = tmp['__PIXIE__']['symbols']
        for python_name, defs in self._syms.items():
            symbol_dict[python_name] = defaultdict(dict)
            for d in defs:
                pixie_sig = types.Signature(d[1])
                cty_str = pixie_sig.as_ctypes_string()
                v = overlay.add_variant(ctypes_func_string=cty_str,
                                        raw_symbol=d[0],
                                        # TODO remove hardcoded feature
                                        baseline='sse3',
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
