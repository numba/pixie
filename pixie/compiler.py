from collections import defaultdict, namedtuple
import uuid
import tempfile
import os
import re
import sys
import types as pytypes
import subprocess
import sysconfig

from llvmlite import ir
from llvmlite.ir.values import GlobalValue
from llvmlite import binding as llvm
from pixie.compiler_lock import compiler_lock
from pixie.targets.common import Features, TargetDescription
from pixie.codegen_helpers import (Codegen, Context, IRGenerator,
                                   module_pass_manager, function_pass_manager)
from pixie.platform import Toolchain
from pixie.dso_tools import (ElfMapper, shmEmbeddedDSOHandler,  # noqa: F401
                             mkstempEmbeddedDSOHandler)  # noqa: F401
from pixie.mcext import c
from pixie.overlay_injectors import (AddPixieDictGenerator,
                                     AugmentingPyInitGenerator)
from pixie import llvm_types as lt


IS_LINUX = sys.platform.startswith('linux')

# TODO: fix for x-compile
defaultDSOHandler = (shmEmbeddedDSOHandler if IS_LINUX
                     else mkstempEmbeddedDSOHandler)


class SimpleCompiler():
    # takes llvm_ir, compiles it to an object file
    def __init__(self, target_cpu, target_features):
        self._target_cpu = target_cpu
        self._target_features = target_features

    def compile(self, sources, opt=0, loop_vectorize=False,
                slp_vectorize=False):
        # takes sources, returns object files
        objects = []
        codegen = Codegen(str(uuid.uuid4().hex),
                          cpu_name=str(self._target_cpu),
                          target_features=self._target_features)

        if isinstance(sources, (str, bytes, llvm.module.ModuleRef)):
            sources = (sources,)

        for source in sources:
            codelibrary = codegen.create_library(uuid.uuid4().hex)
            if isinstance(source, str):
                mod = llvm.parse_assembly(source)
            elif isinstance(source, bytes):
                mod = llvm.parse_bitcode(source)
            elif isinstance(source, llvm.module.ModuleRef):
                mod = source
            else:
                assert 0, f"Unknown source type {type(source)}"
            codelibrary.add_llvm_module(mod)

            with function_pass_manager(codegen._tm,
                                       codelibrary._final_module) as fpm:
                for func in codelibrary._final_module.functions:
                    fpm.initialize()
                    fpm.run(func)
                    fpm.finalize()

            mpm = module_pass_manager(codegen._tm,
                                      opt=opt,
                                      loop_vectorize=loop_vectorize,
                                      slp_vectorize=slp_vectorize)

            remarks_filter = os.environ.get("PIXIE_LLVM_REMARKS_FILTER", "")
            remarks_file = os.environ.get("PIXIE_LLVM_REMARKS_FILE", None)
            if remarks_file is not None:
                (status, remarks) = mpm.run_with_remarks(
                    codelibrary._final_module, remarks_filter=remarks_filter)
                with open(remarks_file, 'at') as f:
                    f.write(remarks)
            else:
                mpm.run(codelibrary._final_module)

            objects.append(codelibrary.emit_native_object())
            del codelibrary
        return tuple(objects)


class SimpleLinker():

    def __init__(self):
        self._toolchain = Toolchain()

    # takes object files and links them into a binary
    def link(self, objects, outfile='a.out'):
        # linker requires objects serialisd onto disk
        with tempfile.TemporaryDirectory() as build_dir:
            objfiles = []
            for obj in objects:
                ntf = os.path.join(build_dir, f"{uuid.uuid4().hex}.o")
                with open(ntf, 'wb') as f:
                    f.write(obj)
                objfiles.append(ntf)

            libraries = self._toolchain.get_python_libraries()
            library_dirs = self._toolchain.get_python_library_dirs()
            self._toolchain.link_shared(outfile,
                                        objfiles,
                                        libraries,
                                        library_dirs,)


class SimpleCompilerDriver():
    # like e.g. clang or gcc, compiles and links source translation units to
    # a DSO.
    def __init__(self, target_cpu, target_features):
        self._compiler = SimpleCompiler(target_cpu, target_features)
        self._linker = SimpleLinker()

    def compile_and_link(self, sources, opt=3,
                         opt_flags=pytypes.MappingProxyType({}),
                         outfile='a.out'):
        objects = self._compiler.compile(sources, opt=opt, **opt_flags)
        return self._linker.link(objects, outfile=outfile)


_pixie_export = namedtuple('_pixie_export',
                           'symbol_name python_name signature metadata')


class TranslationUnit():
    # Maps to a LLVM module... Probably needs to to be consistent with C.

    def __init__(self, name, source):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")
        self._name = name
        self._source = source
        if isinstance(source, str):
            self._mod = llvm.parse_assembly(source)
        elif isinstance(source, bytes):
            self._mod = llvm.parse_bitcode(source)
        else:
            msg = ("Expected string or bytes for source, "
                   f"got {type(source).__name__}.")
            raise TypeError(msg)

    @classmethod
    def from_c_source(cls, path_to_c_file, name="", extra_flags=()):
        # Takes a C-language source file at path `path_to_c_file` and produces a
        # translation unit from it via a call to clang.
        with tempfile.NamedTemporaryFile(suffix=".ll") as ntf:
            # NOTE: This needs to be -O1 or great, -O0 adds `optnone` to
            # function attributes which then prevents optimisation by the PIXIE
            # toolchain.
            cmd = ('clang', '-x', 'c', '-O1',
                   '-I', sysconfig.get_path("include"),
                   '-fPIC', '-mcmodel=small',
                   *extra_flags,
                   '-emit-llvm', path_to_c_file, '-o', ntf.name, '-S')
            try:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                # print here, in case another exception interrupts the handling
                # of this one and the output is masked.
                print(err.stdout.decode())
                raise err

            ntf.flush()
            with open(ntf.name, 'rt') as f:
                data = f.read()
        _name = name or path_to_c_file
        return TranslationUnit(_name, data)

    @classmethod
    def from_cython_source(cls, path_to_cython_file, name="",
                           extra_clang_flags=(), extra_cython_flags=()):
        # convert cython to C, then pass that to `from_c_source`.
        with tempfile.NamedTemporaryFile(suffix='.c') as ntf:
            cmd = ('cython', '-3', *extra_cython_flags, path_to_cython_file,
                   '-o', ntf.name)
            subprocess.check_output(cmd)
            _name = name or path_to_cython_file
            return TranslationUnit.from_c_source(ntf.name, _name,
                                                 extra_flags=extra_clang_flags)


class ExportConfiguration():

    def __init__(self):
        self._data = []

    def add_symbol(self, symbol_name, python_name, signature, metadata=None):
        self._data.append(_pixie_export(symbol_name, python_name, signature,
                                        metadata))


class PIXIECompiler():

    def __init__(self,
                 library_name=None,
                 translation_units=(),
                 export_configuration=None,
                 baseline_cpu="",
                 baseline_features=(),
                 targets_features=(),
                 python_cext=True,
                 uuid=None,
                 opt=3,
                 opt_flags=pytypes.MappingProxyType({}),
                 output_dir='.'):
        self._library_name = library_name
        self._translation_units = translation_units
        self._export_configuration = export_configuration
        self._python_cext = python_cext
        self._uuid = uuid
        self._opt = opt
        self._opt_flags = opt_flags
        self._output_dir = output_dir

        if not isinstance(library_name, str):
            msg = ("kwarg library_name should be a string, got "
                   f"{type(library_name).__name__}")
            raise TypeError(msg)

        if library_name == "":
            msg = ("kwarg library_name cannot be an empty string")
            raise ValueError(msg)

        if not isinstance(translation_units, (tuple, list)):
            msg = ("kwarg translation_units should be a tuple or list, got "
                   f"{type(translation_units).__name__}")
            raise TypeError(msg)

        if len(translation_units) == 0:
            raise ValueError("kwarg translation_units is an empty tuple, no "
                             "translation units were given, there is "
                             "nothing to compile!")

        if export_configuration is not None:
            if not isinstance(export_configuration, ExportConfiguration):
                msg = ("kwarg export_configuration must be of type None or a "
                       "pixie.ExportConfiguration instance, got "
                       f"{type(export_configuration).__name__}")
                raise TypeError(msg)

        if baseline_cpu == "":
            msg = ("The baseline_cpu kwarg must be supplied and also be a "
                   "valid cpu name.\nFor a list of valid CPU names for the "
                   "current process, try running:\n\n"
                   "'pixie.targets.common.display_cpu_names()'")
            raise ValueError(msg)

        if not isinstance(python_cext, bool):
            msg = ("kwarg python_cext should be a bool type, got "
                   f"{type(python_cext).__name__}")
            raise TypeError(msg)

        if isinstance(uuid, (str,)):
            from uuid import UUID
            # make sure that the string will instantiate as a UUID
            UUID(uuid)
        elif uuid is not None:
            msg = ("kwarg uuid must be a string representation of a uuid4 or "
                   "None")
            raise TypeError(msg)

        if opt not in (0, 1, 2, 3):
            msg = ("kwarg opt must be an integer value in (0, 1, 2, 3), got "
                   f"{opt}")
            raise ValueError(msg)

        if not isinstance(opt_flags, (pytypes.MappingProxyType, dict)):
            msg = ("kwarg opt_flags must be a dictionary")
            raise TypeError(msg)

        for k, v in opt_flags.items():
            if k not in ("slp_vectorize", "loop_vectorize"):
                msg = f"kwarg opt_flags contains an invalid key: {k}"
                raise ValueError(msg)
            if not isinstance(v, bool):
                msg = (f"kwarg opt_flags key '{k}' has an invalid value type "
                       f"'{type(v).__name__}' (expected bool).")
                raise TypeError(msg)

        if not isinstance(output_dir, (str, bytes, os.PathLike)):
            msg = ("kwarg output_dir should be a string, bytes or os.PathLike, "
                   f"got {type(output_dir).__name__}")
            raise TypeError(msg)

        # TODO: Triple should be an option to help with cross compile?
        triple = llvm.get_process_triple()
        self._target_descr = TargetDescription(triple,
                                               baseline_cpu,
                                               baseline_features,
                                               targets_features)

    def compile(self):
        ir_mod = ir.Module()
        # TODO: sort this out for x-compile.
        ir_mod.triple = llvm.get_process_triple()
        pixie_mod = PIXIEModule(self._library_name, self._translation_units,
                                self._export_configuration,
                                self._target_descr,
                                opt=self._opt,
                                opt_flags=self._opt_flags,
                                uuid=self._uuid)
        # always set the bitcode to be emitted
        # set the wiring method based on platform
        target_system = self._target_descr.target_triple.sys
        wiring = "ifunc" if "linux" in target_system else "trampoline"
        pixie_mod.generate_ir(ir_mod, python_cext=self._python_cext,
                              wiring=wiring)

        bfeat = self._target_descr.baseline_target.features
        compiler = SimpleCompilerDriver(self._target_descr.baseline_target.cpu,
                                        Features(bfeat),)
        output_file = compiler._linker._toolchain.get_ext_filename(
            self._library_name)
        outpath = os.path.join(self._output_dir, output_file)
        compiler.compile_and_link(str(ir_mod), opt=self._opt,
                                  opt_flags=self._opt_flags, outfile=outpath)


class PIXIEModule(IRGenerator):

    def __init__(self, library_name, translation_units, export_configuration,
                 target_descr, opt, opt_flags, uuid=None):
        self._library_name = library_name
        self._target_descr = target_descr
        self._opt = opt
        self._opt_flags = opt_flags
        self._uuid = uuid

        # convert translation units to a single module
        llvm_irs = [x._source for x in translation_units
                    if isinstance(x._source, str)]
        llvm_bcs = [x._source for x in translation_units
                    if isinstance(x._source, bytes)]
        self._user_source = {'llvm_ir': llvm_irs, 'llvm_bc': llvm_bcs}

        def filter_features(insrc):
            # the LLVM input source may have "target-cpu", "target-features" or
            # "tune-cpu" attributes present on things like functions. These need
            # to be removed as they influence the compilation and seem to
            # override features supplied as flags. Without doing this, it might
            # be possible to compile some C code with clang on a host machine
            # with AVX512F, and then those features impact the instructions in
            # the final binary, which is a SIGILL waiting to happen if the
            # target features were specified as e.g. SSE4.2.
            #
            # For reference, attributes look like:
            # attributes #18 = { "target-cpu"="x86-64" "target-features"="+sse,+x87" } # noqa: E501
            # it's a space separated string with a mixure of single arguments
            # and kwarg pairs of form "key"="comma-separated-values".
            if '"target-' in insrc:
                re_attrs = re.compile(r'(.*){(.*)}')
                buf = []
                for line in insrc.splitlines():
                    if (line.lstrip().startswith("attributes") and
                            '"target-' in line):
                        attrs = re_attrs.match(line).groups()[1]
                        new_attrs = []
                        for attr in attrs.split(' '):
                            if "target-features" in attr:
                                continue
                            if "target-cpu" in attr:
                                continue
                            if "tune-cpu" in attr:
                                continue
                            new_attrs.append(attr)
                        attr_idx = re_attrs.match(line).groups()[0]
                        new_line = attr_idx + "{" + ' '.join(new_attrs) + "}"
                        buf.append(new_line)
                    else:
                        buf.append(line)
                return "\n".join(buf)
            else:
                return insrc

        def _combine_sources():
            # don't create a new module for single source, this matters,
            # particularly for bitcode where a respecialization of a module
            # should embed the exact same bitcode as used to create the
            # original. A reparse of the bitcode can produce a valid but not
            # identical bitcode as to what was inputted.
            if (len(self._user_source['llvm_ir']) == 1 and
                    len(self._user_source['llvm_bc']) == 0):
                llvm_ir = self._user_source['llvm_ir'][0]
                # filter
                filtered_llvm_ir = filter_features(llvm_ir)
                self._single_mod = llvm.parse_assembly(filtered_llvm_ir)
                self._single_source = str(self._single_mod)
                self._single_bitcode = self._single_mod.as_bitcode()
                return
            elif (len(self._user_source['llvm_ir']) == 0 and
                    len(self._user_source['llvm_bc']) == 1):
                bc = self._user_source['llvm_bc'][0]
                llvm_ir = str(llvm.parse_bitcode(bc))
                filtered_llvm_ir = filter_features(llvm_ir)
                if filtered_llvm_ir == llvm_ir:
                    # nothing to filter, so can just wire stuff in, this
                    # satisfies the necessary conditions for respecialization.
                    self._single_mod = llvm.parse_bitcode(bc)
                    self._single_source = str(self._single_mod)
                    self._single_bitcode = bc
                else:
                    self._single_mod = llvm.parse_assembly(filtered_llvm_ir)
                    self._single_source = str(self._single_mod)
                    self._single_bitcode = self._single_mod.as_bitcode()
                return
            else:
                tmp_mod = ir.Module("tmp")
                # TODO: fix for x-compile
                tmp_mod.triple = llvm.get_process_triple()
                ir_module = llvm.parse_assembly(str(tmp_mod))

                mods = []
                for src in self._user_source['llvm_bc']:
                    # bitcode has to go through string form to be filtered
                    tmp = filter_features(str(llvm.parse_bitcode(src)))
                    mods.append(llvm.parse_assembly(tmp))
                for src in self._user_source['llvm_ir']:
                    mods.append(llvm.parse_assembly(filter_features(src)))

                # link into main module
                for mod in mods:
                    ir_module.link_in(mod)

                self._single_mod = ir_module
                self._single_source = str(self._single_mod)
                self._single_bitcode = self._single_mod.as_bitcode()

        _combine_sources()

        # get exports
        self._exported_symbols = defaultdict(list)
        if export_configuration is not None:
            for d in export_configuration._data:
                self._exported_symbols[d.python_name].append((d.symbol_name,
                                                              d.signature,
                                                              d.metadata))

    @compiler_lock
    def _compile_feature_specific_dsos(self,):
        # this returns a map of feature: dso compiled against that feature
        binaries = {}
        all_features = (set(self._target_descr.additional_targets) |
                        {self._target_descr.baseline_target})
        for cpu_descr in all_features:
            cpu_name = str(cpu_descr.cpu)
            features = cpu_descr.features
            cpu_feature = Features(features)
            compiler = SimpleCompilerDriver(cpu_name, cpu_feature)
            with tempfile.TemporaryDirectory() as build_dir:
                outfile = os.path.join(build_dir, str(uuid.uuid4().hex))
                compiler.compile_and_link(self._single_source, opt=self._opt,
                                          opt_flags=self._opt_flags,
                                          outfile=outfile)
                with open(outfile, 'rb') as f:
                    binaries[str(max(features))] = f.read()
        return binaries

    def create_real_ifuncs(self, mod, embedded_libhandle_name, python_cext):

        class IFunc(GlobalValue):
            """
            An IFunc
            """

            def __init__(self, module, name, IFuncTy, resolver_name):
                assert isinstance(IFuncTy, ir.types.FunctionType)
                super(IFunc, self).__init__(module, IFuncTy.as_pointer(),
                                            name=name)
                self.value_type = IFuncTy
                self.initializer = None
                resolver_ty = ir.FunctionType(IFuncTy.as_pointer(), ())
                self.resolver_value_type = resolver_ty
                self.resolver_name = resolver_name
                self.linkage = None
                self.parent.add_global(self)

            def descr(self, buf):

                if self.linkage is None:
                    # Default to no dso_local linkage
                    linkage = 'dso_local'
                else:
                    linkage = self.linkage
                if linkage:
                    buf.append(linkage + " ")
                buf.append("ifunc ")
                buf.append(f"{self.value_type}, ")
                buf.append(f"{self.resolver_value_type}* ")
                buf.append(f"@{self.resolver_name}")
                buf.append("\n")

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
        for llfunc in self._single_mod.functions:
            # Only create a trampoline for functions which have external linkage
            # and are not declarations (i.e. they are defined opposed to
            # linked in).
            if (not llfunc.is_declaration and
                    llfunc.linkage == llvm.Linkage.external):
                symbol_name = llfunc.name
                # if this is declared as a c-extension, do not forward the
                # init function as PIXIE needs to intercept it and inject
                # __PIXIE__ dictionary
                if python_cext and symbol_name.startswith("PyInit_"):
                    continue
                fnty = llfunc.global_value_type.as_ir(mod.context)
                resolver_function_name = f"resolver_for_{symbol_name}"
                ifunc = IFunc(mod, symbol_name, fnty, resolver_function_name)
                resolver = ir.Function(mod, ifunc.resolver_value_type,
                                       resolver_function_name)
                resolver.linkage = 'internal'
                resolver_entry_block = resolver.append_basic_block(
                    'entry_block')
                resolver_builder = ir.IRBuilder(resolver_entry_block)
                # This needs to dlsym
                dso = resolver_builder.load(_handle)
                printf(resolver_builder, "dso is at %d\n", dso)
                const_symbol_name = ctx.insert_const_string(mod, symbol_name)
                sym = c.dlfcn.dlsym(resolver_builder, dso, const_symbol_name)
                printf(resolver_builder, "called dlsym, %d\n", sym)
                # cast void * to fnptr type
                casted_sym = resolver_builder.bitcast(sym, fnty.as_pointer())
                resolver_builder.ret(casted_sym)

    def create_fake_ifuncs(self, mod, embedded_libhandle_name, python_cext):
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
        for llfunc in self._single_mod.functions:
            # Only create a trampoline for functions which have external linkage
            # and are not declarations (i.e. they are defined opposed to
            # linked in).
            if (not llfunc.is_declaration and
                    llfunc.linkage == llvm.Linkage.external):
                symbol_name = llfunc.name
                # if this is declared as a c-extension, do not forward the
                # init function as PIXIE needs to intercept it and inject
                # __PIXIE__ dictionary
                if python_cext and symbol_name.startswith("PyInit_"):
                    continue
                fnty = llfunc.global_value_type.as_ir(mod.context)
                # create a global fnptr for the symbol
                fnty_as_ptr = fnty.as_pointer()
                # the dlsym return is just a void *
                void_ptr_ty = ir.IntType(8).as_pointer()
                fnptr_cache_name = f"_fnptr_cache_for_{symbol_name}"
                fnptr_cache = ir.GlobalVariable(mod, void_ptr_ty,
                                                fnptr_cache_name)
                # nullify on init, this is important state as the trampoline
                # function branches on the NULL.
                fnptr_cache.initializer = ir.Constant(fnptr_cache.type.pointee,
                                                      None)

                # create a function that will trampoline
                trampoline_fn = ir.Function(mod, fnty,
                                            name=symbol_name)
                block = trampoline_fn.append_basic_block(name="entry")
                builder = ir.IRBuilder(block)

                fnptr_local_ref = builder.alloca(fnty_as_ptr)
                pred = builder.icmp_unsigned("==", builder.load(fnptr_cache),
                                             fnptr_cache.type(None))
                printf(builder, "predicate %d\n", pred)
                with builder.if_else(pred) as (then, otherwise):
                    with then:
                        printf(builder, "calling dlsym\n")
                        # find the symbol
                        dso = builder.load(_handle)
                        printf(builder, "dso is at %d\n", dso)
                        const_symbol_name = ctx.insert_const_string(mod,
                                                                    symbol_name)
                        sym = c.dlfcn.dlsym(builder, dso, const_symbol_name)
                        printf(builder, "called dlsym, %d\n", sym)
                        builder.store(sym, fnptr_cache)
                        builder.store(builder.bitcast(sym, fnty_as_ptr),
                                      fnptr_local_ref)
                    with otherwise:
                        printf(builder, "replay from cache\n")
                        # store the dso global value into the slot
                        builder.store(builder.bitcast(
                            builder.load(fnptr_cache), fnty_as_ptr),
                            fnptr_local_ref)
                fn = builder.load(fnptr_local_ref)
                ret = builder.call(fn, trampoline_fn.args)
                if ret.type == ir.VoidType():
                    builder.ret_void()
                else:
                    builder.ret(ret)

    def _embed_bitcode(self, mod):
        ctx = Context()
        bitcode = self._single_bitcode

        bc_name = 'bitcode_for_self'
        bitcode_const_bytes = ctx.insert_const_bytes(mod, bitcode, bc_name)

        fnty = ir.FunctionType(lt._void_star, [])
        getter_name = 'get_bitcode_for_self'
        fn = ir.Function(mod, fnty, getter_name)
        bb = fn.append_basic_block()
        fn_builder = ir.IRBuilder(bb)
        fn_builder.ret(bitcode_const_bytes)

        fnty = ir.FunctionType(lt._int64, [])
        get_sz_name = 'get_sizeof_bitcode_for_self'
        fn = ir.Function(mod, fnty, get_sz_name)
        bb = fn.append_basic_block()
        fn_builder = ir.IRBuilder(bb)
        fn_builder.ret(ir.Constant(lt._int64, len(bitcode)))

    def generate_ir(self, mod, python_cext=True, wiring='trampoline',
                    embed_bitcode=True):

        binaries = self._compile_feature_specific_dsos()
        binaries['baseline'] = binaries[min(binaries.keys())]

        # create the DSO constructor, it does the select and dispatch
        selector_class = self._target_descr.arch.CPUSelector
        dso_handler = defaultDSOHandler()
        emap = ElfMapper(mod)
        emap.create_dso_ctor(binaries, selector_class, dso_handler)
        # create the DSO destructor, it cleans up the resources used by the
        # create_dso_ctor.
        emap.create_dso_dtor(dso_handler)

        if wiring == "trampoline":
            self.create_fake_ifuncs(mod, emap._embedded_libhandle_name,
                                    python_cext)
        elif wiring == "ifunc":
            self.create_real_ifuncs(mod, emap._embedded_libhandle_name,
                                    python_cext)
        else:
            raise ValueError("wiring should be one of 'trampoline' or 'ifunc'")

        if embed_bitcode:
            self._embed_bitcode(mod)

        # If this is targetting a PIXIE C-ext then either create a PyInit
        # function or augment the existing one.
        if python_cext:
            def augment_existing():
                for func in self._single_mod.functions:
                    if func.name == f"PyInit_{self._library_name}":
                        return True
                return False

            have_isas = tuple(binaries.keys())
            if augment_existing():
                # Write a new PyInit which calls up the existing one and then
                # shoves the PIXIE dictionary into it
                # TODO: handle case where multiple PyInit_ exist in the same
                # source.
                gen = AugmentingPyInitGenerator(self._library_name,
                                                emap._embedded_libhandle_name,
                                                self._exported_symbols,
                                                uuid=self._uuid,
                                                available_isas=have_isas)
            else:
                gen = AddPixieDictGenerator(self._library_name,
                                            self._exported_symbols,
                                            uuid=self._uuid,
                                            available_isas=have_isas)
            gen.generate_ir(mod)

        return mod
