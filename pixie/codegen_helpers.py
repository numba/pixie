from llvmlite import ir
from llvmlite import binding as llvm


# NOTE: methods of this class are based on those in:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/base.py#L170
# and those in
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/cgutils.py

class Context():

    GENERIC_POINTER = ir.PointerType(ir.IntType(8))

    def add_global_variable(self, module, ty, name, addrspace=0):
        unique_name = module.get_unique_name(name)
        return ir.GlobalVariable(module, ty, unique_name, addrspace)

    def global_constant(self, builder_or_module, name, value,
                        linkage='internal'):
        """
        Get or create a (LLVM module-)global constant with *name* or *value*.
        """
        if isinstance(builder_or_module, ir.Module):
            module = builder_or_module
        else:
            module = builder_or_module.module
        data = self.add_global_variable(module, value.type, name)
        data.linkage = linkage
        data.global_constant = True
        data.initializer = value
        return data

    def insert_unique_const(self, mod, name, val):
        """
        Insert a unique internal constant named *name*, with LLVM value
        *val*, into module *mod*.
        """
        try:
            gv = mod.get_global(name)
        except KeyError:
            return self.global_constant(mod, name, val)
        else:
            return gv

    def make_bytearray(self, buf):
        """
        Make a byte array constant from *buf*.
        """
        b = bytearray(buf)
        n = len(b)
        return ir.Constant(ir.ArrayType(ir.IntType(8), n), b)

    def insert_const_string(self, mod, string):
        """
        Insert constant *string* (a str object) into module *mod*.
        """
        stringtype = self.GENERIC_POINTER
        name = ".const.%s" % string
        text = self.make_bytearray(string.encode("utf-8") + b"\x00")
        gv = self.insert_unique_const(mod, name, text)
        return ir.Constant.bitcast(gv, stringtype)

    def insert_const_bytes(self, mod, bytes, name=None):
        """
        Insert constant *byte* (a `bytes` object) into module *mod*.
        """
        stringtype = self.GENERIC_POINTER
        name = ".bytes.%s" % (name or hash(bytes))
        text = self.make_bytearray(bytes)
        gv = self.insert_unique_const(mod, name, text)
        return ir.Constant.bitcast(gv, stringtype)

    def create_constant_array(self, ty, val):
        """
        Create an LLVM-constant of a fixed-length array from Python values.

        The type provided is the type of the elements.
        """
        return ir.Constant(ir.ArrayType(ty, len(val)), val)

    def is_null(self, builder, val):
        null = self.get_null_value(val.type)
        return builder.icmp_unsigned('==', null, val)

    def is_not_null(self, builder, val):
        null = self.get_null_value(val.type)
        return builder.icmp_unsigned('!=', null, val)

    def get_null_value(self, ltype):
        return ltype(None)


# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/codegen.py#L518


class CodeLibrary(object):

    _finalized = False
    _object_caching_enabled = False
    _disable_inspection = False

    def __init__(self, codegen, name):
        self._linking_libraries = []   # maintain insertion order
        self._codegen = codegen
        self._name = name
        self._final_module = llvm.parse_assembly(
            str(self._codegen._create_empty_module(self.name)))
        self._final_module.name = normalize_ir_text(self.name)
        self._shared_module = None

    @property
    def name(self):
        return self._name

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("operation impossible on finalized object %r"
                               % (self,))

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, ir.Module)
        nrmir = normalize_ir_text(str(ir_module))
        ll_module = llvm.parse_assembly(nrmir)
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._final_module.link_in(ll_module)

    def create_ir_module(self, name):
        """
        Create an LLVM IR module for use by this library.
        """
        self._raise_if_finalized()
        ir_module = self._codegen._create_empty_module(name)
        return ir_module

    def _optimize_final_module(self):
        pass

    def finalize(self):
        self._raise_if_finalized()

        # Link libraries for shared code
        seen = set()
        for library in self._linking_libraries:
            if library not in seen:
                seen.add(library)
                self._final_module.link_in(
                    library._get_module_for_linking(), preserve=True,
                )

        self._optimize_final_module()
        self._final_module.verify()
        self._finalize_final_module()

    def _finalize_final_module(self):
        self._finalized = True

    def get_llvm_str(self):
        return str(self._final_module)

    def emit_native_object(self):
        """
        Return this library as a native object (a bytestring) -- for example
        ELF under Linux.

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._codegen._tm.emit_object(self._final_module)

    def emit_bitcode(self):
        """
        Return this library as LLVM bitcode (a bytestring).

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._final_module.as_bitcode()

    def get_function(self, name):
        return self._final_module.get_function(name)

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        mod = self._final_module
        for fn in mod.functions:
            if not fn.is_declaration:
                yield fn


# NOTE: This is from:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/codegen.py#L1411

def initialize_llvm():
    """Safe to use multiple times.
    """
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


# NOTE: This is from:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/cgutils.py#L1147

def normalize_ir_text(text):
    """
    Normalize the given string to latin1 compatible encoding that is
    suitable for use in LLVM IR.
    """
    # Just re-encoding to latin1 is enough
    return text.encode('utf8').decode('latin1')

# NOTE: This is based on:
# https://github.com/numba/numba/blob/04ebc63fe1dd1efd5a68cc9caf8f245404d99fa7/numba/core/codegen.py#L1161

class Codegen():

    _library_class = CodeLibrary

    def __init__(self, module_name, cpu_name=None):
        initialize_llvm()

        self._cpu_name = cpu_name or ''
        self._data_layout = None
        self._llvm_module = llvm.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._llvm_module.name = "global_codegen_module"

        target = llvm.Target.from_triple(llvm.get_process_triple())
        tm_options = dict(opt=3)
        self._tm_features = self._customize_tm_features()
        self._customize_tm_options(tm_options)
        tm = target.create_target_machine(**tm_options)
        # Need to bind this to self and keep it alive
        self._engine = llvm.create_mcjit_compiler(self._llvm_module, tm)
        self._tm = tm

    def create_library(self, name, **kwargs):
        """
        Create a :class:`CodeLibrary` object for use with this codegen
        instance.
        """
        return self._library_class(self, name, **kwargs)

    def _create_empty_module(self, name):
        ir_module = ir.Module(normalize_ir_text(name))
        ir_module.triple = llvm.get_process_triple()
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    def _customize_tm_options(self, options):
        cpu_name = self._cpu_name
        if cpu_name == 'host':
            cpu_name = self._get_host_cpu_name()
        options['cpu'] = cpu_name
        options['reloc'] = 'pic'
        options['codemodel'] = 'default'
        options['features'] = self._tm_features

    def _customize_tm_features(self):
        # ISA features are selected according to the requested CPU model
        # in _customize_tm_options()
        return ''
