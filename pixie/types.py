import ctypes
import re
from collections import namedtuple
from llvmlite import ir


class Type(object):
    def __init__(self, name, llvm_type, numpy_shortcode=None):
        self._name = name
        self._numpy_shortcode = numpy_shortcode
        self._llvm_type = llvm_type

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Integer(Type):

    def __init__(self, bits, signed, llvm_type, numpy_shortcode=None):
        assert isinstance(bits, int)
        assert bits > 0
        self._bits = bits
        char = 'i' if signed else 'u'
        super(Integer, self).__init__(f'{char}{bits}', llvm_type,
                                      numpy_shortcode=numpy_shortcode)


class Float(Type):

    def __init__(self, bits, llvm_type, numpy_shortcode):
        assert isinstance(bits, int)
        assert bits > 0
        self._bits = bits
        super(Float, self).__init__(llvm_type, llvm_type, numpy_shortcode)


class Void(Type):
    def __init__(self):
        super(Void, self).__init__('void', 'void', numpy_shortcode='V')


class Pointer(Type):
    def __init__(self, pointee_type):
        self._pointee_type = pointee_type
        super(Pointer, self).__init__(f'{pointee_type}*',
                                      f'{pointee_type._llvm_type}*')

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return other._pointee_type == self._pointee_type

    def __hash__(self):
        return hash(("pointer to", self._pointee_type._name))


# Stand up specific types

int8 = Integer(8, True, 'i8', 'b')
uint8 = Integer(8, False, 'i8', 'B')

int16 = Integer(16, True, 'i16', 'h')
uint16 = Integer(16, False, 'i16', 'H')

int32 = Integer(32, True, 'i32', 'i')
uint32 = Integer(32, False, 'i32', 'I')

int64 = Integer(64, True, 'i32', 'l')
uint64 = Integer(64, False, 'i32', 'L')

float32 = Float(32, 'float', 'f')
float64 = Float(64, 'double', 'd')

byte = Integer(8, True, 'i8', 'b')
void = Void()

# Complex is a structure, typically packed floats, so is not present.

# Pointers
int8p = Pointer(int8)
uint8p = Pointer(uint8)

int16p = Pointer(int16)
uint16p = Pointer(uint16)

int32p = Pointer(int32)
uint32p = Pointer(uint32)

int64p = Pointer(int64)
uint64p = Pointer(uint64)

float32p = Pointer(float32)
float64p = Pointer(float64)

bytep = Pointer(byte)
charp = Pointer(byte)
voidp = Pointer(void)

# Maps

_llvm_ptr_map = {'i8*': int8p,
                 'i16*': int16p,
                 'i32*': int32p,
                 'i64*': int64p,
                 'float*': float32p,
                 'double*': float64p,
                 'char*': charp,
                 'void*': voidp,
                 }


_llvm_ir_ptr_map = {
                 int8p: ir.IntType(8).as_pointer(),
                 uint8p: ir.IntType(8).as_pointer(),
                 int16p: ir.IntType(16).as_pointer(),
                 uint16p: ir.IntType(16).as_pointer(),
                 int32p: ir.IntType(32).as_pointer(),
                 uint32p: ir.IntType(32).as_pointer(),
                 int64p: ir.IntType(64).as_pointer(),
                 uint64p: ir.IntType(64).as_pointer(),
                 float32p: ir.FloatType().as_pointer(),
                 float64p: ir.DoubleType().as_pointer(),
                 charp: ir.IntType(8).as_pointer(),
                 voidp: ir.IntType(8).as_pointer(),
                 }

_ctypes_ptr_map = {int8p: ctypes.POINTER(ctypes.c_int8),
                   uint8p: ctypes.POINTER(ctypes.c_uint8),
                   int16p: ctypes.POINTER(ctypes.c_int16),
                   uint16p: ctypes.POINTER(ctypes.c_uint16),
                   int32p: ctypes.POINTER(ctypes.c_int32),
                   uint32p: ctypes.POINTER(ctypes.c_uint32),
                   int64p: ctypes.POINTER(ctypes.c_int64),
                   uint64p: ctypes.POINTER(ctypes.c_uint64),
                   float32p: ctypes.POINTER(ctypes.c_float),
                   float64p: ctypes.POINTER(ctypes.c_double),
                   bytep: ctypes.POINTER(ctypes.c_int8),
                   charp: ctypes.POINTER(ctypes.c_int8),
                   voidp: ctypes.c_void_p,
                   }

_ctypes_str_ptr_map = {int8p: 'ctypes.POINTER(ctypes.c_int8)',
                       uint8p: 'ctypes.POINTER(ctypes.c_uint8)',
                       int16p: 'ctypes.POINTER(ctypes.c_int16)',
                       uint16p: 'ctypes.POINTER(ctypes.c_uint16)',
                       int32p: 'ctypes.POINTER(ctypes.c_int32)',
                       uint32p: 'ctypes.POINTER(ctypes.c_uint32)',
                       int64p: 'ctypes.POINTER(ctypes.c_int64)',
                       uint64p: 'ctypes.POINTER(ctypes.c_uint64)',
                       float32p: 'ctypes.POINTER(ctypes.c_float)',
                       float64p: 'ctypes.POINTER(ctypes.c_double)',
                       bytep: 'ctypes.POINTER(ctypes.c_int8)',
                       charp: 'ctypes.POINTER(ctypes.c_int8)',
                       voidp: 'ctypes.c_void_p',
                       }

# Parsers
_parse_llvm_encoding = re.compile(r'void\s*\((.*)\)')
_parse_numpy_encoding = re.compile(r'V\((.*)\)')

_parse_pointer = re.compile(r'([\S\d]+)\s*\*')

canonical_signature = namedtuple('canonical_signature',
                                 'return_type argument_types')


class Signature():

    def __init__(self, typestr, encoding='llvm'):
        if encoding == 'llvm':
            argmatch = _parse_llvm_encoding.match(typestr)
        elif encoding == 'numpy':
            argmatch = _parse_numpy_encoding.match(typestr)

        if argmatch is None or len(argmatch.groups()) > 1:
            raise ValueError("Type string signature is not compliant.")

        argpart = argmatch.groups()[0]
        args = [x.strip() for x in argpart.split(',')]

        arg_types = []

        for arg in args:
            iarg_match = _parse_pointer.match(arg)
            if iarg_match is None or len(iarg_match.groups()) > 1:
                msg = f"Argument {arg} is not compliant pointer declaration."
                raise ValueError(msg)
            arg_types.append(f'{iarg_match.groups()[0]}*')

        if encoding == 'llvm':
            type_map = _llvm_ptr_map

        canonical_types = [type_map[arg] for arg in arg_types]

        self._canonical_signature = canonical_signature(void, canonical_types)

    def __repr__(self):
        retty = self._canonical_signature.return_type
        argty = ', '.join(map(str, self._canonical_signature.argument_types))
        return f'{retty}({argty})'

    def as_ctypes(self):
        argtys = self._canonical_signature.argument_types
        ctargs = [_ctypes_ptr_map[x] for x in argtys]
        return canonical_signature(None, ctargs)

    def as_ctypes_string(self):
        argtys = self._canonical_signature.argument_types
        ctargs = [_ctypes_str_ptr_map[x] for x in argtys]
        fn = f"ctypes.CFUNCTYPE(None, {','.join(ctargs)})"
        return fn

    def as_llvm_types(self):
        argtys = self._canonical_signature.argument_types
        llargs = [_llvm_ir_ptr_map[x] for x in argtys]
        return canonical_signature(ir.VoidType(), llargs)

    def as_llvm_function_type(self):
        s = self.as_llvm_types()
        return ir.FunctionType(s.return_type, s.argument_types)

    def __hash__(self):
        return hash((tuple(self._canonical_signature.argument_types),
                    self._canonical_signature.return_type))

    def __eq__(self, other):
        return self._canonical_signature == other._canonical_signature
