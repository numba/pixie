from numba_aot_impl import aot  # noqa: F401 # re-export
import json
import pickle
import inspect
from numba.core import sigutils
from pixie.compiler import PIXIECompiler, ExportConfiguration
from pixie.compiler import TranslationUnit as pixie_TU
from pixie.targets import get_default_configuration
from numba.core.typing import Context as tyctx
from numba.core import cpu
from llvmlite import ir

double = ir.DoubleType()
dble_ptr = double.as_pointer()
float = ir.FloatType()
float_ptr = float.as_pointer()
i64 = ir.IntType(64)
i32 = ir.IntType(32)
i16 = ir.IntType(16)
i8 = ir.IntType(8)
void = ir.VoidType()
voidptr = i8.as_pointer()

legal_llvm_value_types = set((double, float, i64, i32, i16, i8))
legal_llvm_pointer_types = set([x.as_pointer() for x in
                                legal_llvm_value_types])


def gen_pixie_callsite(sig, numba_func):
    tmpctx = cpu.CPUContext(tyctx())
    numba_ll_sig = tmpctx.call_conv.get_function_type(sig.return_type,
                                                      sig.args)
    pixie_args = []
    for arg in numba_ll_sig.args:
        if arg.is_pointer and arg in legal_llvm_pointer_types:
            pixie_args.append(arg)
        elif not arg.is_pointer and arg in legal_llvm_value_types:
            pixie_args.append(arg.as_pointer())
        else:
            pixie_args.append(voidptr)
    # This pixie_args now need the return status adding, this depends on the
    # CC.
    # For the PIXIE<->Numba CC adaptation purposes, add this as first arg
    pixie_args.insert(0, numba_ll_sig.return_type.as_pointer())
    pixie_fnty = ir.FunctionType(void, pixie_args)
    module = ir.Module(name=__file__)
    # NOTE: this needs a proper mangle
    wrapper_func_name = f"pixie_wrapper_{numba_func}"
    pixie_func = ir.Function(module, pixie_fnty, name=wrapper_func_name)
    block = pixie_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    fnty = ir.FunctionType(numba_ll_sig.return_type, numba_ll_sig.args)
    func = ir.Function(module, fnty, numba_func)
    func.linkage = 'external'  # export this

    converted_args = []
    for numba_arg, pixie_arg in zip(numba_ll_sig.args, pixie_func.args[1:]):
        if numba_arg.is_pointer and numba_arg in legal_llvm_pointer_types:
            converted_args.append(pixie_arg)
        elif not numba_arg.is_pointer and numba_arg in legal_llvm_value_types:
            converted_args.append(builder.load(pixie_arg))
        else:
            converted_args.append(builder.bitcast(pixie_arg, numba_arg))

    ret_status = builder.call(func, converted_args)
    # store numba's return status into the first arg
    builder.store(ret_status, pixie_func.args[0])
    builder.ret_void()
    return str(module), wrapper_func_name, pixie_fnty._to_string()


class Library():
    def __init__(self, name, translation_units, outdir=None, libdir=None):
        self._name = name
        self._tus = translation_units

    def compile(self):
        export_config = ExportConfiguration()
        ptus = []
        fnames = ('_limited_helpermod.ll',)
        for name in fnames:
            with open(name, 'rt') as f:
                tmp = f.read()
            ptus.append(pixie_TU("name", tmp))

        for tu in self._tus:

            for src in tu.sources:
                for sig in src._sigs:
                    src._dispatcher.compile(sig)
                    nrm_sig = sigutils.normalize_signature(sig)
                    llvm_ir = src._dispatcher.inspect_llvm(nrm_sig[0])
                    ptu = pixie_TU("tu_name", llvm_ir)
                    ptus.append(ptu)
                    overload = src._dispatcher.overloads[nrm_sig[0]]
                    lfunc_sym = overload.fndesc.llvm_func_name

                    # This needs some attention
                    def encode(thing):
                        return json.dumps(str(pickle.dumps(thing)))

                    pysig = inspect.signature(src._dispatcher.py_func)
                    encoded_py_sig = encode(pysig)
                    encoded_nrm_sig = encode(nrm_sig)

                    # Generate the pixie binding
                    pixie_wrapper, pixie_wrapper_fn_name, pixie_sig = \
                        gen_pixie_callsite(sig, lfunc_sym)
                    ptu = pixie_TU("pixie_wrapper", pixie_wrapper)
                    ptus.append(ptu)
                    qualname = overload.fndesc.qualname
                    md = {'numba_sig': encoded_nrm_sig,
                          'numba_py_sig': encoded_py_sig,
                          'pixie_wrapper': True}
                    export_config.add_symbol(python_name=qualname,
                                             symbol_name=pixie_wrapper_fn_name,
                                             signature=pixie_sig,
                                             metadata=md)

        compiler = PIXIECompiler(library_name=self._name,
                                 translation_units=ptus,
                                 export_configuration=export_config,
                                 **get_default_configuration(),
                                 python_cext=True,
                                 output_dir='.')
        compiler.compile()


class TranslationUnit():
    def __init__(self):
        self._sources = []

    def add(self, function):
        self._sources.append(function)

    @property
    def sources(self):
        return self._sources
