from types import SimpleNamespace
from llvmlite import ir

# ------------------------------------------------------------------------------
# From:
# https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/core/cgutils.py#L417-L425


def get_or_insert_function(module, fnty, name):
    """
    Get the function named *name* with type *fnty* from *module*, or insert it
    if it doesn't exist.
    """
    fn = module.globals.get(name, None)
    if fn is None:
        fn = ir.Function(module, fnty, name)
    return fn

# ------------------------------------------------------------------------------


langref = SimpleNamespace()
langref.types = SimpleNamespace()
langref.types.void = ir.VoidType()
langref.types.i1 = ir.IntType(1)
langref.types.i8 = ir.IntType(8)
langref.types.i16 = ir.IntType(16)
langref.types.i32 = ir.IntType(32)
langref.types.i64 = ir.IntType(64)


c = SimpleNamespace()
# need to declare this early as other things rely on it
_va_list_ty = ir.LiteralStructType([langref.types.i32,
                                    langref.types.i32,
                                    langref.types.i8.as_pointer(),
                                    langref.types.i8.as_pointer()])
c.stdarg = SimpleNamespace(va_list=_va_list_ty)


def _generate_standard_binding(builder, libname, function_name, resty, argtys,
                               fnargs):
    fnty = ir.FunctionType(resty, argtys)
    libandfn = f"{libname}.{function_name}"
    if builder.module.globals.get(libandfn, None) is None:
        fn = get_or_insert_function(builder.module, fnty, libandfn)
        fn.linkage = "internal"
        block = fn.append_basic_block(f"{function_name}_impl")
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, function_name)
        # alloc, store, load for deferred args
        local_slots = []
        for ty, arg in zip(argtys, fn.args):
            slot = local_builder.alloca(ty)
            local_slots.append(slot)
        for slot, arg in zip(local_slots, fn.args):
            local_builder.store(arg, slot)
        loaded_args = [local_builder.load(x) for x in local_slots]
        ret_val = local_builder.call(c_fn, *[loaded_args])
        if isinstance(resty, ir.VoidType):
            local_builder.ret_void()
        else:
            local_builder.ret(ret_val)
        return builder.call(fn, fnargs)
    else:
        fn = get_or_insert_function(builder.module, fnty, libandfn)
        return builder.call(fn, fnargs)


# ------------------------------------------------------------------------------
# fcntl.h

def _open(builder, _path, _oflag, *varargs):
    # note varargs is just `mode_t mode`, it can be supplied if O_CREAT is
    # present in `flags`.
    assert _path.type == c.types.charptr, _path.type
    assert _oflag.type == c.types.int
    _mode = None
    if varargs:
        assert len(varargs) == 1
        (_mode,) = varargs
        assert _mode.type == c.sys.types.mode_t
    fnty = ir.FunctionType(c.types.int, (_path.type, _oflag.type,),
                           var_arg=True)
    fn = get_or_insert_function(builder.module, fnty, "libc.open")
    fn.linkage = "internal"
    block = fn.append_basic_block('open_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "open")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_path, _oflag] + list(varargs))


def _shm_open(builder, _name, _oflag, _mode):
    assert _name.type == c.types.charptr, _name.type
    assert _oflag.type == c.types.int, _oflag.type
    assert _mode.type == c.sys.types.mode_t, _mode.type
    return _generate_standard_binding(builder=builder,
                                      libname="librt",
                                      function_name="shm_open",
                                      resty=c.types.int,
                                      argtys=(c.types.charptr, c.types.int,
                                              c.sys.types.mode_t,),
                                      fnargs=(_name, _oflag, _mode,))


def _shm_unlink(builder, _name,):
    assert _name.type == c.types.charptr, _name.type
    return _generate_standard_binding(builder=builder,
                                      libname="librt",
                                      function_name="shm_unlink",
                                      resty=c.types.int,
                                      argtys=(c.types.charptr,),
                                      fnargs=(_name,))

# stdlib


def _malloc(builder, _size):
    assert _size.type == c.stddef.size_t, _size.type
    return _generate_standard_binding(builder=builder,
                                      libname="libc",
                                      function_name="malloc",
                                      resty=c.types.voidptr,
                                      argtys=(c.stddef.size_t,),
                                      fnargs=(_size,))


def _free(builder, _ptr):
    assert _ptr.type == c.types.voidptr, _ptr.type
    return _generate_standard_binding(builder=builder,
                                      libname="libc",
                                      function_name="free",
                                      resty=c.types.void,
                                      argtys=(c.types.voidptr,),
                                      fnargs=(_ptr,))


def _exit(builder, _status):
    assert _status.type == c.types.int, _status.type
    fnty = ir.FunctionType(c.types.void, (_status.type,))
    fn = get_or_insert_function(builder.module, fnty, "libc.exit")
    fn.linkage = "internal"
    block = fn.append_basic_block('exit_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "exit")
    local_builder.call(c_fn, *[fn.args,])
    local_builder.unreachable()
    return builder.call(fn, [_status,])


def _getenv(builder, _name):
    assert _name.type == c.types.charptr, _name.type
    fnty = ir.FunctionType(c.types.charptr, (_name.type,))

    if builder.module.globals.get('libc.getenv', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.getenv")
        fn.linkage = "internal"
        block = fn.append_basic_block('getenv_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "getenv")
        retval = local_builder.call(c_fn, *[fn.args,])
        local_builder.ret(retval)
        return builder.call(fn, [_name,])
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.getenv")
        return builder.call(fn, (_name,))


def _mkstemp(builder, _template):
    # Note that the _template cannot be a string constant as it will be
    # modified, it should be declared as a char array.
    assert _template.type == c.types.charptr, _template.type
    fnty = ir.FunctionType(c.types.int, (_template.type,))

    if builder.module.globals.get('libc.mkstemp', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.mkstemp")
        fn.linkage = "internal"
        block = fn.append_basic_block('mkstemp_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "mkstemp")
        retval = local_builder.call(c_fn, *[fn.args,])
        local_builder.ret(retval)
        return builder.call(fn, [_template,])
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.mkstemp")
        return builder.call(fn, (_template,))


def _abort(builder,):
    fnty = ir.FunctionType(c.types.void, ())
    if builder.module.globals.get('libc.abort', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.abort")
        block = fn.append_basic_block('abort_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "abort")
        local_builder.call(c_fn, *[fn.args,])
        local_builder.unreachable()
        return builder.call(fn, ())
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.abort")
        return builder.call(fn, ())


# unistd.h

def _read(builder, ):
    pass


def _write(builder, _fd, _buf, _count):
    assert _fd.type == c.types.int, _fd.type
    assert _buf.type == c.types.voidptr, _buf.type
    assert _count.type == c.stddef.size_t, _count.type

    fnty = ir.FunctionType(c.stddef.size_t, (_fd.type, _buf.type, _count.type))
    fn = get_or_insert_function(builder.module, fnty, "libc.write")
    fn.linkage = "internal"
    block = fn.append_basic_block('write_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "write")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_fd, _buf, _count])


def _getpid(builder,):
    fnty = ir.FunctionType(c.sys.types.pid_t, ())
    fn = get_or_insert_function(builder.module, fnty, "libc.getpid")
    fn.linkage = "internal"
    block = fn.append_basic_block('getpid_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "getpid")
    ret_val = local_builder.call(c_fn, *[fn.args,])
    local_builder.ret(ret_val)
    return builder.call(fn, [])


def _ftruncate(builder, _flides, _length):
    assert _flides.type == c.types.int, _flides.type
    assert _length.type == c.sys.types.off_t, _length.type

    fnty = ir.FunctionType(c.types.int, (_flides.type, _length.type,))
    fn = get_or_insert_function(builder.module, fnty, "libc.ftruncate")
    fn.linkage = "internal"
    block = fn.append_basic_block('ftruncate_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "ftruncate")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_flides, _length,])


def _readlink(builder, _path, _buf, _bufsize):
    assert _path.type == c.types.charptr, _path.type
    assert _buf.type == c.types.charptr, _buf.type
    assert _bufsize.type == c.stddef.size_t, _bufsize.type

    fnty = ir.FunctionType(c.stddef.size_t, (_path.type, _buf.type,
                                             _bufsize.type))
    fn = get_or_insert_function(builder.module, fnty, "libc.readlink")
    fn.linkage = "internal"
    block = fn.append_basic_block('readlink_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "readlink")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_path, _buf, _bufsize])


def _unlink(builder, _path):
    assert _path.type == c.types.charptr, _path.type

    fnty = ir.FunctionType(c.stddef.size_t, (_path.type,))
    fn = get_or_insert_function(builder.module, fnty, "libc.unlink")
    fn.linkage = "internal"
    block = fn.append_basic_block('unlink_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "unlink")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_path,])


# stdio.h

def _printf(builder, _fmt, *varargs):
    # This goes via vprintf and needs va_args
    assert _fmt.type == c.types.charptr, _fmt.type
    outer_fnty = ir.FunctionType(c.types.void, (_fmt.type,), var_arg=True)

    if builder.module.globals.get('libc.printf', None) is None:
        fn = get_or_insert_function(builder.module, outer_fnty, "libc.printf")
        fn.linkage = "internal"
        block = fn.append_basic_block('printf_impl')
        local_builder = ir.IRBuilder(block)

        # need to use va_args as this is forwarding variadic args to an inner
        # function
        i8_ptr = langref.types.i8.as_pointer()
        va_list_ptr = local_builder.alloca(c.stdarg.va_list)
        va_list = local_builder.bitcast(va_list_ptr, i8_ptr)

        llvm_va_start_fnty = ir.FunctionType(langref.types.void, (i8_ptr,))
        llvm_va_start_fn = get_or_insert_function(builder.module,
                                                  llvm_va_start_fnty,
                                                  "llvm.va_start")

        local_builder.call(llvm_va_start_fn, (va_list,))

        vprintf_fnty = ir.FunctionType(c.types.void,
                                       (_fmt.type,
                                        c.stdarg.va_list.as_pointer()))
        c_fn = get_or_insert_function(builder.module, vprintf_fnty, "vprintf")

        local_builder.call(c_fn, (fn.args[0], va_list_ptr))

        llvm_va_end_fnty = ir.FunctionType(langref.types.void, (i8_ptr,))
        llvm_va_end_fn = get_or_insert_function(builder.module,
                                                llvm_va_end_fnty, "llvm.va_end")

        local_builder.call(llvm_va_end_fn, (va_list,))

        local_builder.ret_void()
        return builder.call(fn, [_fmt] + list(varargs))
    else:
        fn = get_or_insert_function(builder.module, outer_fnty, "libc.printf")
        return builder.call(fn, [_fmt] + list(varargs))


def _snprintf(builder,  _str, _size, _format, *varargs):
    # This goes via vsnprintf and needs va_args
    assert _str.type == c.types.charptr, _str.type
    assert _size.type == c.stddef.size_t
    assert _format.type == c.types.charptr
    outer_fnty = ir.FunctionType(c.types.int,
                                 (_str.type, _size.type, _format.type),
                                 var_arg=True)

    if builder.module.globals.get('libc.snprintf', None) is None:
        fn = get_or_insert_function(builder.module, outer_fnty, "libc.snprintf")
        fn.linkage = "internal"
        block = fn.append_basic_block('snprintf_impl')
        local_builder = ir.IRBuilder(block)

        # need to use va_args as this is forwarding variadic args to an inner
        # function
        i8_ptr = langref.types.i8.as_pointer()
        va_list_ptr = local_builder.alloca(c.stdarg.va_list)
        va_list = local_builder.bitcast(va_list_ptr, i8_ptr)

        llvm_va_start_fnty = ir.FunctionType(langref.types.void, (i8_ptr,))
        llvm_va_start_fn = get_or_insert_function(builder.module,
                                                  llvm_va_start_fnty,
                                                  "llvm.va_start")

        local_builder.call(llvm_va_start_fn, (va_list,))

        vsnprintf_fnty = ir.FunctionType(c.types.int,
                                         (_str.type, _size.type, _format.type,
                                          c.stdarg.va_list.as_pointer()))
        c_fn = get_or_insert_function(builder.module, vsnprintf_fnty,
                                      "vsnprintf")

        retval = local_builder.call(c_fn, fn.args + (va_list_ptr,))

        llvm_va_end_fnty = ir.FunctionType(langref.types.void, (i8_ptr,))
        llvm_va_end_fn = get_or_insert_function(builder.module,
                                                llvm_va_end_fnty, "llvm.va_end")

        local_builder.call(llvm_va_end_fn, (va_list,))
        local_builder.ret(retval)
        return builder.call(fn, [_str, _size, _format] + list(varargs))
    else:
        fn = get_or_insert_function(builder.module, outer_fnty, "libc.snprintf")
        return builder.call(fn, [_str, _size, _format] + list(varargs))


def _perror(builder, _s):
    assert _s.type == c.types.charptr, _s.type
    fnty = ir.FunctionType(c.types.void, (_s.type,))
    if builder.module.globals.get('libc.perror', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.perror")
        fn.linkage = "internal"
        block = fn.append_basic_block('perror_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "perror")
        local_builder.call(c_fn, *[fn.args,])
        local_builder.ret_void()
        return builder.call(fn, [_s,])
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.perror")
        return builder.call(fn, [_s,])


# string.h
def _memset(builder, _b, _c, _len):
    # C   : void * memset(void *b, int c, size_t len)
    # enforce types
    assert _b.type == c.types.voidptr
    assert _c.type == c.types.int
    assert _len.type == c.stddef.size_t
    fnty = ir.FunctionType(c.types.voidptr, (_b.type, _c.type, _len.type))
    # LLVM: void memset(ptr <dest>, i8 <val>, i{32,64} <len>, i1 <isvolatile>)
    # llvmlite binds this via 2 args, the pointer and the type of the length
    if builder.module.globals.get('libc.memset', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.memset")
        fn.linkage = "internal"
        block = fn.append_basic_block('memset_impl')
        local_builder = ir.IRBuilder(block)
        intrin_fn = local_builder.module.declare_intrinsic('llvm.memset',
                                                           (c.types.voidptr,
                                                            _len.type,))
        new_c = local_builder.trunc(fn.args[1], langref.types.i8)
        local_builder.call(intrin_fn, [fn.args[0], new_c, fn.args[2]] +
                           [langref.types.i1(0),])
        builder.call(fn, [_b, _c, _len])
        return local_builder.ret(fn.args[0])
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.memset")
        return builder.call(fn, (_b, _c, _len))


def _strncmp(builder, _s1, _s2, _n):
    assert _s1.type == c.types.charptr
    assert _s2.type == c.types.charptr
    assert _n.type == c.stddef.size_t, _n.type

    fnty = ir.FunctionType(c.types.int, (_s1.type, _s2.type, _n.type))
    if builder.module.globals.get('libc.strncmp', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.strncmp")
        fn.linkage = "internal"
        block = fn.append_basic_block('strncmp_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "strncmp")
        retval = local_builder.call(c_fn, *[fn.args,])
        local_builder.ret(retval)
        return builder.call(fn, (_s1, _s2, _n))
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.strncmp")
        return builder.call(fn, (_s1, _s2, _n))


def _strncpy(builder, _dst, _src, _len):
    assert _dst.type == c.types.charptr
    assert _src.type == c.types.charptr
    assert _len.type == c.stddef.size_t, _len.type

    fnty = ir.FunctionType(c.types.int, (_dst.type, _src.type, _len.type))
    if builder.module.globals.get('libc.strncpy', None) is None:
        fn = get_or_insert_function(builder.module, fnty, "libc.strncpy")
        fn.linkage = "internal"
        block = fn.append_basic_block('strncpy_impl')
        local_builder = ir.IRBuilder(block)
        c_fn = get_or_insert_function(builder.module, fnty, "strncpy")
        retval = local_builder.call(c_fn, *[fn.args,])
        local_builder.ret(retval)
        return builder.call(fn, (_dst, _src, _len))
    else:
        fn = get_or_insert_function(builder.module, fnty, "libc.strncpy")
        return builder.call(fn, (_dst, _src, _len))


# stdint.h

_int8_t = ir.IntType(8)
_int16_t = ir.IntType(16)
_int32_t = ir.IntType(32)
_int64_t = ir.IntType(64)
_uint8_t = ir.IntType(8)
_uint16_t = ir.IntType(16)
_uint32_t = ir.IntType(32)
_uint64_t = ir.IntType(64)


# dlfcn.h

def _dlopen(builder, _path, _mode):
    assert _path.type == c.types.charptr, _path.type
    assert _mode.type == c.types.int, _mode.type
    fnty = ir.FunctionType(c.types.voidptr, (_path.type, _mode.type))
    fn = get_or_insert_function(builder.module, fnty, "libdl.dlopen")
    fn.linkage = "internal"
    block = fn.append_basic_block('dlopen_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "dlopen")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_path, _mode])


def _dlerror(builder,):
    fnty = ir.FunctionType(c.types.charptr, ())
    fn = get_or_insert_function(builder.module, fnty, "libdl.dlerror")
    fn.linkage = "internal"
    block = fn.append_basic_block('dlerror_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "dlerror")
    local_builder.call(c_fn, [])
    return builder.call(fn, [])


def _dlsym(builder, _handle, _symbol):
    assert _handle.type == c.types.voidptr, _handle.type
    assert _symbol.type == c.types.charptr, _symbol.type
    fnty = ir.FunctionType(c.types.voidptr, (_handle.type, _symbol.type))
    fn = get_or_insert_function(builder.module, fnty, "libdl.dlsym")
    fn.linkage = "internal"
    block = fn.append_basic_block('dlsym_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "dlsym")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_handle, _symbol])


def _dlclose(builder, _handle):
    assert _handle.type == c.types.voidptr, _handle.type
    fnty = ir.FunctionType(c.types.int, (_handle.type,))
    fn = get_or_insert_function(builder.module, fnty, "libdl.dlclose")
    fn.linkage = "internal"
    block = fn.append_basic_block('dlclose_impl')
    local_builder = ir.IRBuilder(block)
    c_fn = get_or_insert_function(builder.module, fnty, "dlclose")
    ret_val = local_builder.call(c_fn, *[fn.args])
    local_builder.ret(ret_val)
    return builder.call(fn, [_handle])

# ------------------------------------------------------------------------------


c.types = SimpleNamespace(int=_int32_t,
                          float=ir.FloatType(),
                          double=ir.DoubleType(),
                          void=ir.VoidType(),
                          voidptr=_int8_t.as_pointer(),
                          char=_int8_t,
                          charptr=_int8_t.as_pointer(),
                          )


c.stddef = SimpleNamespace(size_t=_int64_t,
                           ptrdiff_t=_int64_t,)

c.sys = SimpleNamespace()
# TODO: check these types for linux and OSX.
c.sys.types = SimpleNamespace(pid_t=_int32_t,
                              mode_t=_int32_t,
                              off_t=_int32_t)

# Reference for constants:
# https://github.com/llvm/llvm-project/blob/a131525908a908baff4cd01140dae158a307dc9e/libc/include/llvm-libc-macros/linux/sys-stat-macros.h#L33-L36
c.sys.stat = SimpleNamespace(S_IRUSR=ir.Constant(c.sys.types.mode_t, 0o00400),
                             S_IWUSR=ir.Constant(c.sys.types.mode_t, 0o00200),
                             S_IXUSR=ir.Constant(c.sys.types.mode_t, 0o00100),)

c.stdint = SimpleNamespace(int8_t=_int8_t,
                           int16_t=_int16_t,
                           int32_t=_int32_t,
                           int64_t=_int64_t,
                           uint8_t=_uint8_t,
                           uint16_t=_uint16_t,
                           uint32_t=_uint32_t,
                           uint64_t=_uint64_t,)

# Reference for constants:
# https://github.com/llvm/llvm-project/blob/a131525908a908baff4cd01140dae158a307dc9e/libc/include/llvm-libc-macros/linux/fcntl-macros.h#L14
# https://github.com/llvm/llvm-project/blob/a131525908a908baff4cd01140dae158a307dc9e/libc/include/llvm-libc-macros/linux/fcntl-macros.h#L46
c.fcntl = SimpleNamespace(open=_open,
                          shm_open=_shm_open,
                          shm_unlink=_shm_unlink,
                          # consts, in octal
                          O_RDWR=ir.Constant(c.types.int, 0o02),
                          O_CREAT=ir.Constant(c.types.int, 0o100),
                          )


c.stdlib = SimpleNamespace(malloc=_malloc,
                           free=_free,
                           exit=_exit,
                           getenv=_getenv,
                           mkstemp=_mkstemp,
                           abort=_abort,)


c.unistd = SimpleNamespace(read=_read,
                           write=_write,
                           getpid=_getpid,
                           ftruncate=_ftruncate,
                           readlink=_readlink,
                           unlink=_unlink)


c.stdio = SimpleNamespace(printf=_printf,
                          snprintf=_snprintf,
                          perror=_perror,)


c.string = SimpleNamespace(memset=_memset,
                           strncmp=_strncmp,
                           strncpy=_strncpy)


# Reference for constants:
# https://git.musl-libc.org/cgit/musl/plain/include/dlfcn.h
c.dlfcn = SimpleNamespace(dlopen=_dlopen,
                          dlclose=_dlclose,
                          dlsym=_dlsym,
                          dlerror=_dlerror,
                          # hex consts
                          RTLD_LAZY=ir.Constant(c.types.int, 0x00001),
                          RTLD_NOW=ir.Constant(c.types.int, 0x00002),
                          RTLD_LOCAL=ir.Constant(c.types.int, 0x0),
                          RTLD_GLOBAL=ir.Constant(c.types.int, 0x00100),
                          )

# Reference for constants:
# https://git.musl-libc.org/cgit/musl/plain/include/sysexits.h
c.sysexits = SimpleNamespace(EX_OK=ir.Constant(c.types.int, 0),
                             EX_SOFTWARE=ir.Constant(c.types.int, 70))

# ------------------------------------------------------------------------------
# This is libm

m = SimpleNamespace()

# ------------------------------------------------------------------------------
# DEBUG
# ------------------------------------------------------------------------------
# def _main():
#     class _tmp():
#         def module(self):
#             return ir.Module()
#
#         def function(self, module=None, name='my_func'):
#             module = module or self.module()
#             fnty = ir.FunctionType(c.types.int, ())
#             return ir.Function(module, fnty, name)
#
#         def block(self, func=None, name=''):
#             func = func or self.function()
#             return func.append_basic_block(name)
#
#     helper = _tmp()
#     fn = helper.function()
#     block = fn.append_basic_block('a_block')
#     builder = ir.IRBuilder(block)
#
#     # test memset
#     _s = builder.alloca(c.types.int, name="s")
#     _c = builder.alloca(c.types.int, name="c")
#     _n = builder.alloca(c.stddef.size_t, name="n")
#     c.string.memset(builder, builder.bitcast(_s, c.types.voidptr),
#                     builder.load(_c), builder.load(_n))
#
#     # test snprintf
#     _str = builder.alloca(c.types.char, name="str")
#     _size = builder.alloca(c.stddef.size_t, name="size")
#     _format = builder.alloca(c.types.char, name="format")
#     _some_int = builder.alloca(c.types.int, name="some_int")
#     c.stdio.snprintf(builder, _str, builder.load(_size), _format,
#                      builder.load(_some_int))
#
#
#     # test perror
#     _s = builder.alloca(c.types.char, name="s")
#     c.stdio.perror(builder, _s)
#
#     # test exit
#     _status = builder.alloca(c.types.int, name="status")
#     c.stdlib.exit(builder, builder.load(_status))
#
#     # test shm_open
#     _name = builder.alloca(c.types.char, name="name")
#     _oflag = builder.alloca(c.types.int, name="oflag")
#     builder.store(builder.or_(c.fcntl.O_RDWR, c.fcntl.O_CREAT), _oflag)
#     _mode = builder.alloca(c.sys.types.mode_t, name="mode")
#     builder.store(builder.or_(c.sys.stat.S_IRUSR, c.sys.stat.S_IWUSR), _mode)
#     stat = c.fcntl.shm_open(builder, _name, builder.load(_oflag),
#                             builder.load(_mode))
#
#     # test open
#     _name = builder.alloca(c.types.char, name="name")
#     _oflag = builder.alloca(c.types.int, name="oflag")
#     builder.store(builder.or_(c.fcntl.O_RDWR, c.fcntl.O_CREAT), _oflag)
#     _mode = builder.alloca(c.sys.types.mode_t, name="mode")
#     builder.store(builder.or_(c.sys.stat.S_IRUSR, c.sys.stat.S_IWUSR), _mode)
#     stat = c.fcntl.open(builder, _name, builder.load(_oflag),
#                         builder.load(_mode))
#
#     # test ftruncate
#     _flides = builder.alloca(c.types.int, name="flides")
#     _length = builder.alloca(c.sys.types.off_t, name="length")
#     stat = c.unistd.ftruncate(builder, builder.load(_flides),
#                               builder.load(_length))
#
#     # test write
#     _fd = builder.alloca(c.types.int, name="fd")
#     _buf = builder.alloca(c.types.char, name="buf")
#     _count = builder.alloca(c.stddef.size_t, name="count")
#     stat = c.unistd.write(builder, builder.load(_fd), _buf,
#                           builder.load(_count))
#
#     # test free
#     _ptr = builder.alloca(c.types.int, name="ptr")
#     c.stdlib.free(builder, builder.bitcast(_ptr, c.types.voidptr))
#
#     # test getpid
#     c.unistd.getpid(builder)
#
#     # test readlink
#     _path = builder.alloca(c.types.char, name="path")
#     _buf = builder.alloca(c.types.char, name="buf")
#     _count = builder.alloca(c.stddef.size_t, name="count")
#     stat = c.unistd.readlink(builder, _path, _buf, builder.load(_count))
#
#     # test dlopen
#     _path = builder.alloca(c.types.char, name="path")
#     _mode = builder.alloca(c.types.int, name="mode")
#     builder.store(c.dlfcn.RTLD_NOW, _mode)
#     stat = c.dlfcn.dlopen(builder, _path, builder.load(_mode))
#
#     # test dlclose
#     _handle = builder.alloca(c.types.char, name="handle")
#     stat = c.dlfcn.dlclose(builder, _handle)
#
#     # test dlsym
#     _handle = builder.alloca(c.types.char, name="handle")
#     _symbol = builder.alloca(c.types.char, name="symbol")
#     stat = c.dlfcn.dlsym(builder, _handle, _symbol)
#
#     # test dlerror
#     stat = c.dlfcn.dlerror(builder,)
#
#     # test printf
#     _str = builder.alloca(c.types.char, name="str")
#     _size = builder.alloca(c.stddef.size_t, name="size")
#     _format = builder.alloca(c.types.char, name="format")
#     _some_int = builder.alloca(c.types.int, name="some_int")
#     c.stdio.printf(builder, _str, _size, _format, _some_int)
#
#
#     builder.ret_void()
#     print(builder.module)
#     import pdb; pdb.set_trace()
#
#
# if __name__ == "__main__":
#     _main()
