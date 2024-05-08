from abc import abstractmethod
import uuid

from llvmlite import ir

from pixie.codegen_helpers import Context
from pixie.mcext import c, langref


class ElfMapper(object):

    def __init__(self, module):
        self._mod = module
        self._embedded_libhandle_name = "_libhandle" # f"_libhandle_{uuid.uuid4().hex}"
        self._elf_filepath = None
        self._ctx = Context()

    def create_loader(self, binaries, selector_class, embedder):

        # This creates function that loads and wires in the _mod_init_func_ptr
        # global.
        do_load_fn = ir.Function(self._mod,
                                ir.FunctionType(ir.VoidType(), ()),
                                name="_do_load")
        do_load_entry_block = do_load_fn.append_basic_block('entry_block')
        do_load_builder = ir.IRBuilder(do_load_entry_block)

        # select impl
        nbytes, thebytes = apply_selector(do_load_builder, binaries,
                                          selector_class)

        # get the filepath to the extracted embedded binary
        self._elf_filepath = embedder._create_internal(do_load_builder,
                                                       thebytes, nbytes)

        # TODO: Move this.
        # make sure the handle is saved
        langref_i8_ptr = langref.types.i8.as_pointer()
        handle = ir.GlobalVariable(self._mod, langref_i8_ptr,
                                   self._embedded_libhandle_name)
        handle.initializer = ir.Constant(handle.type.pointee, None)

        load_library(do_load_builder, self._ctx, self._elf_filepath, handle)

        do_load_builder.ret_void()
        return do_load_fn

    def create_dso_ctor(self, binaries, selector_class, embedder):
        # Create the interposing shared library constructor function
        # "_dso_ctor", it will:
        # 1. load an embedded library appropriate to the selection method used.
        # 2. write the address of the embedded library's handle into a global.
        dso_ctor_fn = ir.Function(self._mod, ir.FunctionType(c.types.void, ()),
                                name="_dso_ctor")
        dso_ctor_entry_block = dso_ctor_fn.append_basic_block('entry_block')
        dso_ctor_builder = ir.IRBuilder(dso_ctor_entry_block)
        # create the function that will do the loading of the embedded module.
        do_load_fn = self.create_loader(binaries, selector_class, embedder)
        dso_ctor_builder.call(do_load_fn, ())
        dso_ctor_builder.ret_void()

        # Shared library constructor declaration cf.
        # __attribute__((constructor (priority)))
        generate_dso_ctor_dtor(self._mod, "constructor", {dso_ctor_fn: 101})

    def create_dso_dtor(self, dso_handler):
        """
        Does clean up of the loaded embedded DSO
        * dlclose()s it
        * cleans up the file system resources
        """
        # Shared library constructor function:
        dso_dtor_fn = ir.Function(self._mod, ir.FunctionType(c.types.void, ()),
                                name="_dso_dtor")
        dso_dtor_entry_block = dso_dtor_fn.append_basic_block('entry_block')
        dso_dtor_builder = ir.IRBuilder(dso_dtor_entry_block)
        _handle = self._mod.get_global(self._embedded_libhandle_name)
        # In the common case, the handle needs closing as the embedded library
        # successfully loaded. However, there is the possibility that the
        # constructor failed to load the embedded library and the handle is
        # NULL, in this case, do not attempt to dlclose the library.

        # dlclose the handle
        deref_handle = dso_dtor_builder.load(_handle)
        pred = dso_dtor_builder.not_(self._ctx.is_null(dso_dtor_builder,
                                                       deref_handle))
        with dso_dtor_builder.if_then(pred, likely=True):
            c.dlfcn.dlclose(dso_dtor_builder, deref_handle)

        # destroy the file system resources
        _EXTRACTED_FILEPATH_as_charptr = dso_dtor_builder.bitcast(
            dso_handler.EXTRACTED_FILEPATH, c.types.charptr)
        dso_handler._destroy_internal(dso_dtor_builder,
                                    _EXTRACTED_FILEPATH_as_charptr)
        dso_dtor_builder.ret_void()

        # Shared library destructor declaration cf.
        # __attribute__((destructor (priority)))
        generate_dso_ctor_dtor(self._mod, "destructor", {dso_dtor_fn: 101})


def generate_dso_ctor_dtor(mod, ctor_or_dtor, func_to_priority_map):
    """
    Shared library constructor/destructor declaration cf.
     __attribute__(({con,de}structor (priority)))
    * mod is the current module
    * ctor_or_dtor is the string "constructor" or "destructor" to select what
      to generate.
    * func_to_priority_map is a map of llvm functions to call in the ctor/dtor
      to the "priority" (an int) associated with them.
    """

    # use the variable name `etor` as ctor and dtor are the same structs
    assert ctor_or_dtor in ("constructor", "destructor")
    etor_fnty = ir.FunctionType(c.types.void, ()).as_pointer()
    etor_struct = ir.LiteralStructType([ir.IntType(32), etor_fnty,
                                        ir.IntType(8).as_pointer()])

    # need as much array as there are things in the func_to_priority_map
    nfuncs = len(func_to_priority_map)
    etor_arr = ir.ArrayType(etor_struct, nfuncs)
    special_name = f"llvm.global_{ctor_or_dtor[0]}tors"
    _enstructor_arr = ir.GlobalVariable(mod, etor_arr, special_name)

    # init list
    init_list = []
    data = ir.Constant(ir.IntType(8).as_pointer(), None)
    for func, priority in func_to_priority_map.items():
        ll_priority = ir.Constant(ir.IntType(32), priority)
        init_list.append(etor_struct([ll_priority, func, data]))
    _enstructor_arr.initializer = etor_arr(init_list)
    _enstructor_arr.linkage = 'appending'


def load_library(builder, ctx, file_path, handle_cache):
    """Loads the library at file_path into the handle_cache, if handle_cache
    is not null, returns the result of loading handle_cache"""
    mod = builder.module
    is_null = ctx.is_null(builder, builder.load(handle_cache))
    with builder.if_else(is_null) as (then, otherwise):
        with then:
            # find out if the handle is null, if so dlopen the library and store
            # the handle
            libhandle = c.dlfcn.dlopen(builder, file_path,
                                    builder.or_(c.dlfcn.RTLD_NOW,
                                                c.dlfcn.RTLD_LOCAL))
            builder.store(libhandle, handle_cache)
        with otherwise:
            libhandle = builder.load(handle_cache)
    return libhandle

def load_symbol(builder, ctx, file_path, symbol_name, handle_cache):
    """Open library at absolute file system path "file_path" and fetch address
    of symbol named "symbol_name" from the loaded library. `handle_cache` is a
    "file handle cache" to store the handle from the dlopen of file_path, if it
    is NULL then the library at file_path will be dlopen'd and the handled
    stored into the cache and the lookup will then be performed against the
    cache, if it's non-null the lookup will be performed against file handle
    cache. Returns the address of symbol_name in file_path.
    """
    mod = builder.module
    libhandle = load_library(builder, ctx, file_path, handle_cache)
    # dlsym
    # *(void **)(&tmp) = dlsym(_libhandle, symbol_name);
    const_symbol_name = ctx.insert_const_string(mod, symbol_name)
    return c.dlfcn.dlsym(builder, libhandle, const_symbol_name)


def apply_selector(builder, binaries, selector_class):
    """In the current block write out the code for a selector 'selector_class'
    operating on 'binaries', the resulting selected binary will be returned
    as a tuple of (number of bytes, the bytes)."""
    # START THE selector PART
    # create the selector site
    ctx = Context()
    disp = selector_class(builder.module,
                          f"pixie_selector_{selector_class.__name__}",
                          binaries)
    disp_fn = disp.generate_selector()

    # create a couple of pointers that will be written to with the functions
    # that return the size and the data from the selected embedded module.
    nbytes_fn_ptr = builder.alloca(c.types.voidptr)
    ctx.init_alloca(builder, nbytes_fn_ptr)
    get_bytes_fn_ptr = builder.alloca(c.types.voidptr)
    ctx.init_alloca(builder, get_bytes_fn_ptr)

    # call the selector
    builder.call(disp_fn, (nbytes_fn_ptr, get_bytes_fn_ptr))

    # cast the functions and call them to get the number of bytes and the bytes
    # themselves.
    size_t_void_fnty = ir.FunctionType(c.stddef.size_t, ())
    casted = builder.bitcast(nbytes_fn_ptr,
                                     size_t_void_fnty.as_pointer().as_pointer())
    nbytes = builder.call(builder.load(casted), ())

    voidptr_void_fnty = ir.FunctionType(c.types.voidptr, ())
    voidptr_void_fnty_ptr_ptr = voidptr_void_fnty.as_pointer().as_pointer()
    casted = builder.bitcast(get_bytes_fn_ptr,
                                     voidptr_void_fnty_ptr_ptr)
    thebytes = builder.call(builder.load(casted), ())
    return nbytes, thebytes
    # END selector PART


class EmbeddedDSOHandler():
    """
    Handles the conversion of a DSO as embedded bytes into a discoverable
    file name, and the clean up of the file on exit.
    """

    def __init__(self,):
        self._ctx = Context()
        self._EXTRACTED_FILEPATH = None
        self._NAME_MAX = 255
        self._DEBUG = False

    def debug_print(self, builder, *args):
        if self._DEBUG:
            self._ctx.printf(builder, *args)

    @property
    def NAME_MAX(self):
        return self._NAME_MAX

    @property
    def EXTRACTED_FILEPATH(self):
        if self._EXTRACTED_FILEPATH is None:
            msg = ("The .create() method must be called to populate the "
                   "extracted filepath.")
            raise RuntimeError(msg)
        else:
            return self._EXTRACTED_FILEPATH

    def _create_internal(self, builder, thebytes, nbytes):
        # call the create() method to get the file path specific to this loader
        mapped_file_path = self.create(builder, thebytes, nbytes)

        # create a global name for the file path and write the generated
        # file path into the global.
        mod = builder.module
        fpath = f"_EXTRACTED_FILEPATH_{uuid.uuid4().hex}"
        _EXTRACTED_FILEPATH = ir.GlobalVariable(mod, ir.ArrayType(c.types.char,
                                                self.NAME_MAX),
                                                fpath)
        _EXTRACTED_FILEPATH.initializer = ir.Constant(
            _EXTRACTED_FILEPATH.type.pointee, None)
        _EXTRACTED_FILEPATH.align = 16
        count = c.stdio.snprintf(builder,
                                 builder.bitcast(_EXTRACTED_FILEPATH,
                                                 c.types.charptr),
                                 ir.Constant(ir.IntType(64), (self.NAME_MAX)),
                                 mapped_file_path)
        self.debug_print(builder,
                         "snprintf _EXTRACTED_FILEPATH count = %d\n",
                         count)
        self.debug_print(builder,
                         "snprintf of filename to _EXTRACTED_FILEPATH %s\n",
                         _EXTRACTED_FILEPATH)

        self._EXTRACTED_FILEPATH = _EXTRACTED_FILEPATH
        return builder.bitcast(_EXTRACTED_FILEPATH, c.types.charptr)

    def _destroy_internal(self, builder, filepath):
        self.destroy(builder, filepath)
        # zero out global
        fname_as_charptr = builder.bitcast(self._EXTRACTED_FILEPATH,
                                           c.types.charptr)
        c.string.memset(builder, fname_as_charptr,
                        ir.Constant(c.types.int, 0),
                        ir.Constant(c.stddef.size_t, self.NAME_MAX))

    @abstractmethod
    def create(self, builder, thebytes, nbytes):
        """This needs to take bytes `thebytes` and number of bytes `nbytes` and
        write them into an actual file that dlopen() can use. i.e. it has to be
        a file system accessible file."""
        pass

    @abstractmethod
    def destroy(self, filepath):
        """This needs to appropriately remove the resources created in `create`
        from the file system. The `file_path` argument is the file path as
        returned by the call to `self.create()`."""
        pass


class shmEmbeddedDSOHandler(EmbeddedDSOHandler):
    """
    Use SHM based mapping to provide embedded DSO handling.
    """

    def create(self, builder, thebytes, nbytes):
        mod = builder.module
        size_t_name_max = ir.Constant(c.stddef.size_t, self.NAME_MAX)
        shm_name = f"/{uuid.uuid4().hex}.so"
        shm_file_name = self._ctx.insert_const_string(mod, shm_name)

        self._shm_file_name = shm_file_name

        # do shm_open
        _oflag = builder.alloca(c.types.int, name="oflag")
        builder.store(builder.or_(c.fcntl.O_RDWR, c.fcntl.O_CREAT), _oflag)
        _mode = builder.alloca(c.sys.types.mode_t, name="mode")
        mode_flags = builder.or_(c.sys.stat.S_IRUSR, c.sys.stat.S_IWUSR)
        builder.store(mode_flags, _mode)
        shm_fd = c.fcntl.shm_open(builder,
                                  builder.bitcast(shm_file_name,
                                                  c.types.charptr),
                                  builder.load(_oflag),
                                  builder.load(_mode))
        self.debug_print(builder, "done shm_open on\n", shm_file_name)

        # ftruncate the shm_fd to the length of the bytes
        c.unistd.ftruncate(builder, shm_fd, builder.trunc(nbytes, c.types.int))
        self.debug_print(builder, "done truncate\n")
        # write bytes into the shm_fd
        c.unistd.write(builder, shm_fd, thebytes, nbytes)

        # get the pid
        pid = c.unistd.getpid(builder)
        self.debug_print(builder, "done getpid %d\n", pid)
        # allocate proc_path memory
        proc_path = builder.alloca(c.types.char, self.NAME_MAX)

        # memset the proc_path to nul
        c.string.memset(builder, proc_path, ir.Constant(c.types.int, 0),
                        size_t_name_max)
        self.debug_print(builder, "done memset %s\n", proc_path)
        # do this:
        # written = snprintf(proc_path, str_size, "/proc/%d/fd/%d", pid,
        #                    shm_fd);
        fmt = self._ctx.insert_const_string(mod, "/proc/%d/fd/%d")
        written = c.stdio.snprintf(builder,
                                   proc_path,
                                   ir.Constant(ir.IntType(64), (self.NAME_MAX)),
                                   builder.bitcast(fmt, c.types.charptr),
                                   pid,
                                   shm_fd)
        self.debug_print(builder, "proc_path written = %d\n", written)
        self.debug_print(builder, "proc_path should be = /proc/%d/fd/%d\n", pid,
                         shm_fd)
        self.debug_print(builder, "proc_path = %s\n", proc_path)

        # allocate shm_file_path
        shm_file_path = builder.alloca(c.types.char, self.NAME_MAX)

        # memset shm_file_path to nul
        c.string.memset(builder, shm_file_path, ir.Constant(c.types.int, 0),
                        size_t_name_max)
        self.debug_print(builder, "done memset on shm_file_path %s\n",
                         shm_file_path)

        # readlink
        # size_t readlink_result = readlink(proc_path, shm_file_path, str_size);
        readlink_result = c.unistd.readlink(builder, proc_path,
                                            shm_file_path,
                                            size_t_name_max)
        self.debug_print(builder, "done readlink %d\n", readlink_result)

        self.debug_print(builder, "shm_file_path %s\n", shm_file_path)

        return shm_file_path

    def destroy(self, builder, file_path):
        # NOTE file_path is not used. shm_unlink needs the "portable" use name
        # given to `shm_open`, which is stored in `self._shm_file_name`.
        rval = c.fcntl.shm_unlink(builder, self._shm_file_name)
        self.debug_print(builder, "shm_unlink on %s: stat = %d\n",
                         self._shm_file_name, rval)


class mkstempEmbeddedDSOHandler(EmbeddedDSOHandler):
    """
    Use mkstemp based mapping to provide embedded DSO handling.
    """
    def create(self, builder, thebytes, nbytes):
        # allocate a filepath slot
        file_path = builder.alloca(c.types.char, self.NAME_MAX)
        self._ctx.init_alloca(builder, file_path)

        # memset filepath slot to nul
        c.string.memset(builder, file_path, ir.Constant(c.types.int, 0),
                        ir.Constant(c.stddef.size_t, self.NAME_MAX))
        self.debug_print(builder, "done memset on file_path %s\n", file_path)
        # copy in the template
        template = "/tmp/template_name_XXXXXX"
        template_str = self._ctx.insert_const_string(builder.module,
                                                     template)
        template_str_len = ir.Constant(c.stddef.size_t, len(template))
        # prevent overflow at compile time
        msg = "temporary file name template will overflow buffer"
        assert len(template) < self.NAME_MAX, msg
        c.string.strncpy(builder, file_path, template_str, template_str_len)
        fd = c.stdlib.mkstemp(builder, file_path)
        c.unistd.write(builder, fd, thebytes, nbytes)
        return file_path

    def destroy(self, builder, file_path):
        c.unistd.unlink(builder, file_path)
