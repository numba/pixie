from abc import abstractmethod
from collections import namedtuple

from llvmlite import ir

from pixie.codegen_helpers import Context
from pixie.mcext import c, langref


class Selector():

    _DEBUG = False

    def __init__(self, mod, name, data):
        self._name = name
        self._mod = mod
        self._data = data
        self._ctx = Context()
        self._embedded_data = self._embed_data()
        self._debug_selector_str = self._debug_selector(mod)

    def _debug_selector(self, mod):
        # add global
        _selected_dso = self._ctx.add_global_variable(mod, c.types.charptr,
                                                      "_selected_dso")
        _selected_dso.linkage = "linkonce_odr"
        _selected_dso.initializer = _selected_dso.type.pointee(None)

        fnty = ir.FunctionType(c.types.charptr, ())
        get_selected_fn = ir.Function(mod, fnty, "get_selected_dso")
        fn_builder = ir.IRBuilder(get_selected_fn.append_basic_block())
        fn_builder.ret(fn_builder.load(_selected_dso))
        return _selected_dso

    def debug_print(self, builder, *args):
        if self._DEBUG:
            self._ctx.printf(builder, *args)

    def _embed_data(self):
        embedded_data = namedtuple('embedded_data', 'nbytes_fn get_bytes_fn')
        data_map = {}

        for k, v in self._data.items():
            mod = self._mod
            bytes_name = f'bytes_for_{k}'
            const_bytes = self._ctx.insert_const_bytes(mod, v, bytes_name)

            fnty = ir.FunctionType(c.types.voidptr, ())
            getter_name = f'get_bytes_for_{k}'
            get_bytes_fn = ir.Function(mod, fnty, getter_name)
            bb = get_bytes_fn.append_basic_block()
            fn_builder = ir.IRBuilder(bb)
            fn_builder.ret(const_bytes)

            fnty = ir.FunctionType(c.stdint.uint64_t, ())
            get_sz_name = f'get_sizeof_{k}_bytes'
            get_size_fn = ir.Function(mod, fnty, get_sz_name)
            bb = get_size_fn.append_basic_block()
            fn_builder = ir.IRBuilder(bb)
            fn_builder.ret(ir.Constant(c.stdint.uint64_t, len(v)))

            data_map[k] = embedded_data(get_size_fn, get_bytes_fn)

        return data_map

    def _select(self, builder, entry):
        # This is purely for debug/testing
        str_entry = self._ctx.insert_const_string(builder.module, str(entry))
        builder.store(str_entry, self._debug_selector_str)

        # this does the wiring for the selected embedded data
        nbytes_fn = self._embedded_data[entry].nbytes_fn
        get_bytes_fn = self._embedded_data[entry].get_bytes_fn
        builder.store(builder.bitcast(nbytes_fn,
                                      self._nbytes_fn_ptr.type.pointee),
                      self._nbytes_fn_ptr)
        builder.store(builder.bitcast(get_bytes_fn,
                                      self._get_bytes_fn_ptr.type.pointee),
                      self._get_bytes_fn_ptr)

    def generate_selector(self,):
        voidptrptr = c.types.voidptr.as_pointer()
        disp_fn_ty = ir.FunctionType(c.types.void,
                                     (voidptrptr, voidptrptr))
        selector_fn = ir.Function(self._mod, disp_fn_ty, name=self._name)
        selector_fn.linkage = "internal"
        self._nbytes_fn_ptr, self._get_bytes_fn_ptr = selector_fn.args
        entry_block = selector_fn.append_basic_block('entry_block')
        builder = ir.IRBuilder(entry_block)
        self.selector_impl(builder)
        return selector_fn

    @abstractmethod
    def selector_impl(self, builder):
        raise NotImplementedError("This method must be implemented.")

    def _generate_env_var_check(self, builder, found, variant_order):
        i8 = langref.types.i8

        # This is an "escape hatch" for debug etc whereby the ISA is picked from
        # the environment variable PIXIE_USE_ISA (if it has been set).
        # TODO: use secure_getenv if available
        PIXIE_USE_ISA_env_var_str = self._ctx.insert_const_string(
            builder.module, "PIXIE_USE_ISA")
        # this returns a char * to the env var contents, or NULL if no match
        PIXIE_USE_ISA_value = c.stdlib.getenv(builder,
                                                PIXIE_USE_ISA_env_var_str)
        envvar_set_pred = builder.not_(
            self._ctx.is_null(builder, PIXIE_USE_ISA_value))
        self.debug_print(builder, "Testing for env var dispatch.\n")
        with builder.if_else(envvar_set_pred, likely=False) as (then,
                                                                otherwise):
            with then:
                self.debug_print(builder, "Using env var dispatch.\n")
                # TODO: this is like the env var selector impl, could that be
                # used?
                for specific_feature in variant_order:
                    zero = ir.Constant(c.types.int, 0)
                    max_len = ir.Constant(c.stddef.size_t, 255)
                    feature_name_str = self._ctx.insert_const_string(
                        builder.module, specific_feature)
                    strcmp_res = c.string.strncmp(builder, PIXIE_USE_ISA_value,
                                                    feature_name_str,
                                                    max_len)
                    pred = builder.icmp_signed("==", strcmp_res, zero)
                    with builder.if_then(pred):
                        msg = "Using version from env var: PIXIE_USE_ISA=%s\n"
                        self.debug_print(builder, msg, PIXIE_USE_ISA_value)
                        self._select(builder, specific_feature)
                        # mark having found a suitable match
                        builder.store(ir.Constant(i8, 1), found)
                        builder.ret_void()

                # check that a match was found and abort otherwise
                pred = builder.icmp_unsigned("==", builder.load(found),
                                                ir.Constant(i8, 0))
                with builder.if_then(pred, likely=False):
                    message = ("No matching library is available for ISA "
                                "\"%s\" supplied via environment variable "
                                "PIXIE_USE_ISA.\n"
                                "\nThis error is unrecoverable and the program "
                                "will now exit. Try checking that the supplied "
                                "ISA is valid and then rerun.\n")
                    error_message = self._ctx.insert_const_string(
                        builder.module, message)
                    c.stdio.printf(builder, error_message, PIXIE_USE_ISA_value)
                    # call sigabrt.
                    self.debug_print(builder, "calling exit\n")
                    c.stdlib.exit(builder, c.sysexits.EX_SOFTWARE)
                    builder.ret_void()
            with otherwise:
                self.debug_print(builder, "No env var set.\n")


class PyVersionSelector(Selector):

    def selector_impl(self, builder):
        # query python version from in process
        NULL = langref.types.i8.as_pointer()(None)
        dlself = c.dlfcn.dlopen(builder, NULL, c.dlfcn.RTLD_NOW)
        Py_GetVersion_sym = self._ctx.insert_const_string(builder.module,
                                                          "Py_GetVersion")
        sym_addr = c.dlfcn.dlsym(builder, dlself, Py_GetVersion_sym)

        # call sym_addr to get const char * string of version
        Py_GetVersion_fnty = ir.FunctionType(c.types.charptr, ())
        fnptr = builder.bitcast(sym_addr, Py_GetVersion_fnty.as_pointer())
        py_str = builder.call(fnptr, ())

        # strncmp first 4 bytes against known python versions
        size_t_four = ir.Constant(c.stddef.size_t, 4)
        zero = ir.Constant(c.types.int, 0)
        for k in reversed(self._embedded_data.keys()):
            str_pyver = self._ctx.insert_const_string(builder.module, k)
            strcmp_res = c.string.strncmp(builder, py_str, str_pyver,
                                          size_t_four)
            pred = builder.icmp_signed("==", strcmp_res, zero)
            with builder.if_then(pred):
                self.debug_print(builder, f"Version: {k}\n")
                self._select(builder, k)
                builder.ret_void()
        builder.ret_void()


class EnvVarSelector(Selector):

    def __init__(self, *args, **kwargs):
        assert "envvar_name" in kwargs
        self._envvar_name = kwargs.pop("envvar_name")
        super().__init__(*args, **kwargs)

    def selector_impl(self, builder):
        # query an environment variable from in process
        zero = ir.Constant(c.types.int, 0)
        _env_var_name = self._ctx.insert_const_string(builder.module,
                                                      self._envvar_name)
        for k in self._embedded_data.keys():
            str_envvar = self._ctx.insert_const_string(builder.module, k)
            lenk = c.stddef.size_t(len(k))
            str_envvar_len = self._ctx.insert_unique_const(builder.module,
                                                           f"_len_{k}", lenk)
            env_var_value = c.stdlib.getenv(builder, _env_var_name)
            self.debug_print(builder, "env_var_value %s\n", env_var_value)
            strcmp_res = c.string.strncmp(builder, str_envvar, env_var_value,
                                          builder.load(str_envvar_len))
            pred = builder.icmp_signed("==", strcmp_res, zero)
            with builder.if_then(pred):
                self.debug_print(builder, f"Using version from env var: {k}\n")
                self._select(builder, k)
                builder.ret_void()
        builder.ret_void()


class NoSelector(Selector):

    def __init__(self, mod, name, data, **kwargs):
        assert len(data) == 1
        super().__init__(mod, name, data, **kwargs)

    def selector_impl(self, builder):
        # Only one key in dict so always return that entry
        key = next(iter(self._data.keys()))
        self._select(builder, key)
        builder.ret_void()
