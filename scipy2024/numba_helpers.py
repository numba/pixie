from collections import namedtuple
from collections.abc import Iterable
import json
import pickle
import textwrap
from numba import njit, types
from numba.extending import overload, intrinsic
from numba.core import cgutils
from llvmlite import binding as llvm
from llvmlite import ir as llvmir
from numba.core.callconv import excinfo_t
NumbaFunctions = namedtuple('NumbaFunctions', 'jit aot')


# The following `_generate*` functions are derived from the retired
# `@overload_glue` decorator present in Numba 0.56 as found in this file:
# https://github.com/numba/numba/blob/2e833f7eb00375329f76a60cbcd9ebb32c2c0fb6/numba/core/overload_glue.py#L131

def _generate_wrapper(name, python_signature, intrin):
    # This is from Numba's retired @overload_glue decorator.
    # It generates a function with arguments that match the supplied
    # python_signature wired through to a call to "intrin".
    pysig_params = python_signature.parameters
    buf = []
    for k, v in pysig_params.items():
        if v.default is v.empty:  # no default ~= positional arg
            buf.append(k)
        else:  # is kwarg, wire in default
            buf.append(f'{k} = {v.default}')
    call_str_specific = ', '.join(buf)
    call_str = ', '.join(pysig_params.keys())
    gen = textwrap.dedent(("""
    def {}({}):
        return intrin({})
    """)).format(name, call_str_specific, call_str)
    lcls = {}
    g = {'intrin': intrin}
    exec(gen, g, lcls)
    return lcls[f'{name}']


def generate_intrinsic_wrapper(name, python_signature, return_type, codegen):
    # This is derived from Numba's retired @overload_glue decorator.
    # It Generates a function with arguments that match the arg names of the
    # supplied python_signature wired through to a call to "intrin".
    pysig_params = python_signature.parameters
    call_str = ', '.join(pysig_params.keys())
    gen = textwrap.dedent(("""
    def {}(tyctx, {}):
        sig = return_type({})
        return sig, codegen
    """)).format(name, call_str, call_str)
    lcls = {}
    g = {'codegen': codegen, 'return_type': return_type, 'types': types}
    exec(gen, g, lcls)
    fn = lcls[f'{name}']
    return fn


double = llvmir.DoubleType()
dble_ptr = double.as_pointer()
float = llvmir.FloatType()
float_ptr = float.as_pointer()
i64 = llvmir.IntType(64)
i32 = llvmir.IntType(32)
i16 = llvmir.IntType(16)
i8 = llvmir.IntType(18)
void = llvmir.VoidType()
voidptr = i8.as_pointer()

legal_llvm_value_types = set((double, float, i64, i32, i16, i8))
legal_llvm_pointer_types = set([x.as_pointer() for x in
                                legal_llvm_value_types])


# This is based on Numba's call convention, specifically:
# https://github.com/numba/numba/blob/7f056946c7a69f8739c07ef5a1bdb4b4b5be72cd/numba/core/callconv.py#L871-L910


class CallConv():

    def __init__(self, numba_cc):
        self.numba_cc = numba_cc

    def call_function(self, builder, callee, resty, argtys, args, attrs=None):
        """
        Call the Numba-compiled *callee*.
        Parameters:
        -----------
        attrs: LLVM style string or iterable of individual attributes, default
                is None which specifies no attributes. Examples:
                LLVM style string: "noinline fast"
                Equivalent iterable: ("noinline", "fast")
        """
        # XXX better fix for callees that are not function values
        #     (pointers to function; thus have no `.args` attribute)
        ll_resty = self.numba_cc.context.get_value_type(resty)
        retvaltmp = cgutils.alloca_once(builder, ll_resty)
        # initialize return value to zeros
        builder.store(cgutils.get_null_value(ll_resty), retvaltmp)

        excinfoptr = cgutils.alloca_once(builder,
                                         llvmir.PointerType(excinfo_t),
                                         name="excinfo")

        arginfo = self.numba_cc._get_arg_packer(argtys)
        args = list(arginfo.as_arguments(builder, args))
        converted = [retvaltmp, excinfoptr]

        for arg in args:
            if arg.type.is_pointer and arg.type in legal_llvm_pointer_types:
                converted.append(arg)
            elif (not arg.type.is_pointer and
                  arg.type in legal_llvm_value_types):
                slot = builder.alloca(arg.type)
                builder.store(arg, slot)
                converted.append(slot)
            else:
                converted.append(builder.bitcast(arg, voidptr))

        # deal with attrs, it's fine to specify a load in a string like
        # "noinline fast" as per LLVM or equally as an iterable of individual
        # attributes.
        if attrs is None:
            _attrs = ()
        elif isinstance(attrs, Iterable) and not isinstance(attrs, str):
            _attrs = tuple(attrs)
        else:
            raise TypeError("attrs must be an iterable of strings or None")

        # retcode_t (<Python return type>*, excinfo **, ... <Python arguments>)
        if isinstance(callee, llvmir.CastInstr):  # fnptr cast from AOT
            return_code = builder.alloca(callee.type.pointee.args[0].pointee)
        else:  # bitcode callsite
            return_code = builder.alloca(callee.args[0].type.pointee)
        converted.insert(0, return_code)

        builder.call(callee, converted, attrs=_attrs)
        status = self.numba_cc._get_return_status(builder,
                                                  builder.load(return_code),
                                                  builder.load(excinfoptr))
        casted_ret = builder.bitcast(retvaltmp, ll_resty.as_pointer())
        retval = builder.load(casted_ret)
        out = self.numba_cc.context.get_returned_value(builder, resty, retval)
        return status, out


def generate_overload_wiring(overload_trampoline, python_signature, nb_str_sig,
                             ol_type, codegen):
    # This basically does the following:

    # an intrinsic with appropriate args and signature is generated and wired
    # through to a call to the supplied codegen.
    # @intrinsic
    # def intrinsic_wrapper(...):
    #     return codegen

    # A jit function with a concrete signature (nb_str_sig) is created to call
    # the above intrinsic. This is to trivivally block further compilation as
    # there no actual implementation available other than that present.
    # @njit(concrete_signature)
    # def concrete_wrapper(...):
    #     return intrinsic_wrapper(...)

    # An overload for the overload_trampoline function which calls the
    # concrete_wrapper.
    # @overload(overload_trampoline):
    # def ol_trampoline(...):
    #     return concrete_wrapper

    # Technically, the concrete wrapper is all that's needed, but in the case
    # of wanting to recompile from e.g. NIL having the overload mech in place
    # makes it more easily possible to do this. Further, the NumbaNamespace
    # could set the CPython AOT binding as .jit and .aot so it works from the
    # interpreter, then this impl just needs to overload those functions.
    # However this would require Numba to have type inference adding for
    # "NumbaNamespace".

    # intrinsic for this impl
    bind_call = intrinsic(generate_intrinsic_wrapper(f'bind_{ol_type}_call',
                                                     python_signature,
                                                     nb_str_sig.return_type,
                                                     codegen))

    # This blocks further compilation attempts
    wrapper = _generate_wrapper(f"_{ol_type}_concrete_wrapper",
                                python_signature, bind_call)
    _concrete_wrapper = njit(nb_str_sig,
                             forceinline=True,
                             no_cpython_wrapper=True)(wrapper)

    # wire it all up
    impl = _generate_wrapper("_trampoline_impl", python_signature,
                             _concrete_wrapper)
    ol_trampoline = _generate_wrapper(f"ol_{ol_type}_trampoline",
                                      python_signature, lambda *args: impl)
    overload(overload_trampoline)(ol_trampoline)


def pixie_converter(pixie_lib):
    syms = pixie_lib.__PIXIE__['symbols'].keys()

    tmp_namespace = namedtuple(f'NumbaNamespace_{pixie_lib.__name__}',
                               ' '.join(syms))
    kwargs = {}
    for k in syms:

        bindings = pixie_lib.__PIXIE__['symbols'][k]

        def jit_trampoline(*args):
            print("wire in CPython/PyObject variant here")

        def aot_trampoline(*args):
            print("wire in CPython/PyObject variant here")

        for pixie_sig, data in bindings.items():
            md = data['metadata']
            if md is None:
                continue
            if 'numba_sig' not in md:
                continue

            if not md.get('pixie_wrapper', False):
                continue

            nb_str_sig = md['numba_sig']
            nrm_sig = pickle.loads(eval(json.loads(nb_str_sig)))
            nb_type_sig = nrm_sig[1](*nrm_sig[0])

            # This needs some attention
            pysig_bytestr = json.loads(md['numba_py_sig'])
            python_signature = pickle.loads(eval(pysig_bytestr))

            # The JIT implementation:
            def jit_codegen(cgctx, builder, sig, llargs):
                bitcode = pixie_lib.__PIXIE__['bitcode']
                mod = llvm.parse_bitcode(bitcode)
                cgctx.active_code_library.add_llvm_module(mod)
                sym_name = data['symbol']
                fnty = cgctx.call_conv.get_function_type(sig.return_type,
                                                         sig.args)

                newargs = [*fnty.args[:2]]
                for arg in fnty.args[2:]:
                    if arg.is_pointer and arg in legal_llvm_pointer_types:
                        newargs.append(arg)
                    elif arg in legal_llvm_value_types:
                        newargs.append(arg.as_pointer())
                    elif arg.is_pointer:
                        newargs.append(voidptr)
                    else:
                        raise RuntimeError("unknown arg type")
                # inject the return value pointer slot at the front
                return_value_ty = fnty.return_type.as_pointer()
                newargs.insert(0, return_value_ty)
                newfnty = llvmir.FunctionType(void, newargs)
                fn = cgutils.get_or_insert_function(builder.module, newfnty,
                                                    sym_name)
                fn.attributes.add('alwaysinline')
                cc = CallConv(cgctx.call_conv)
                stat, out = cc.call_function(builder, fn, sig.return_type,
                                             sig.args, llargs)
                # the PIXIE function won't have incref'd this
                cgctx.nrt.incref(builder, sig.return_type, out)
                # not sure this is quite right yet
                with cgutils.if_unlikely(builder, stat.is_error):
                    cgctx.call_conv.return_status_propagate(builder, stat)
                return out

            generate_overload_wiring(jit_trampoline, python_signature,
                                     nb_type_sig, "jit", jit_codegen)

            # The AOT implementation
            def aot_codegen(cgctx, builder, sig, llargs):
                addressable_name = f"AOT_BINDING_{data['symbol']}"
                ptr_val = cgctx.add_dynamic_addr(builder, data['address'],
                                                 addressable_name)
                fnty = cgctx.call_conv.get_function_type(sig.return_type,
                                                         sig.args)

                newargs = [*fnty.args[:2]]
                for arg in fnty.args[2:]:
                    if arg.is_pointer and arg in legal_llvm_pointer_types:
                        newargs.append(arg)
                    elif arg in legal_llvm_value_types:
                        newargs.append(arg.as_pointer())
                    elif arg.is_pointer:
                        newargs.append(voidptr)
                    else:
                        raise RuntimeError("unknown arg type")

                return_value_ty = fnty.return_type.as_pointer()
                newargs.insert(0, return_value_ty)
                newfnty = llvmir.FunctionType(void, newargs)

                cc = CallConv(cgctx.call_conv)
                casted = builder.bitcast(ptr_val, newfnty.as_pointer())
                stat, out = cc.call_function(builder, casted, sig.return_type,
                                             sig.args, llargs)
                # the PIXIE function won't have incref'd this
                cgctx.nrt.incref(builder, sig.return_type, out)
                # not sure this is quite right yet
                with cgutils.if_unlikely(builder, stat.is_error):
                    cgctx.call_conv.return_status_propagate(builder, stat)
                return out

            generate_overload_wiring(aot_trampoline, python_signature,
                                     nb_type_sig, "aot", aot_codegen)

        @njit(forceinline=True)
        def jit_wrapper(*args):
            return jit_trampoline(*args)

        @njit(forceinline=True)
        def aot_wrapper(*args):
            return aot_trampoline(*args)

        kwargs[k] = NumbaFunctions(jit_wrapper, aot_wrapper)

    return tmp_namespace(**kwargs)


def gen_pixie_raw_callsite(pixie_lib, pysym, pixie_sig):
    @intrinsic
    def bind_call(tyctx, ty_x):

        sig = ty_x(ty_x)

        def codegen(cgctx, builder, sig, llargs):
            bitcode = pixie_lib.__PIXIE__['bitcode']
            mod = llvm.parse_bitcode(bitcode)
            # Prevent e.g. cython globals from being multiply defined.
            for v in mod.global_variables:
                if v.linkage == llvm.Linkage.external:
                    v.linkage = llvm.Linkage.linkonce_odr
            cgctx.active_code_library.add_llvm_module(mod)
            foo_sym = pixie_lib.__PIXIE__['symbols'][pysym]
            sym_name = foo_sym[pixie_sig]['symbol']
            double_ptr = llvmir.DoubleType().as_pointer()
            fnty = llvmir.FunctionType(llvmir.VoidType(),
                                       (double_ptr, double_ptr))
            fn = cgutils.get_or_insert_function(builder.module, fnty,
                                                sym_name)
            fn.attributes.add('alwaysinline')
            x_ptr = cgutils.alloca_once_value(builder, llargs[0])
            r_ptr = cgutils.alloca_once(builder, llargs[0].type)
            builder.call(fn, (x_ptr, r_ptr))
            return builder.load(r_ptr)
        return sig, codegen

    @njit(forceinline=True, no_cpython_wrapper=True)
    def pixie_trampoline(x):
        return bind_call(x)
    return pixie_trampoline
