from collections import defaultdict, namedtuple
import uuid
import tempfile
import os
import sys

from llvmlite import ir
from llvmlite.ir.values import GlobalValue
from llvmlite import binding as llvm
from pixie.compiler_lock import compiler_lock
from pixie.targets.common import Features, TargetDescription
from pixie.codegen_helpers import (Codegen, Context, _module_pass_manager,
                                   IRGenerator)
from pixie.platform import Toolchain
from pixie.dso_tools import (ElfMapper, shmEmbeddedDSOHandler,  # noqa: F401
                             mkstempEmbeddedDSOHandler)  # noqa: F401
from pixie.mcext import c
from pixie.overlay_injectors import (AddPixieDictGenerator,
                                     AugmentingPyInitGenerator)
from pixie import llvm_types as lt


IS_LINUX = sys.platform.startswith('linux')


class SimpleCompiler():
    # takes llvm_ir, compiles it to an object file
    def __init__(self, target_cpu, target_features):
        self._target_cpu = target_cpu
        self._target_features = target_features

    def compile(self, sources, opt=0):
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
            # TODO: wire in loop and slp vectorize
            mpm = _module_pass_manager(codegen._tm,
                                       opt=opt,
                                       loop_vectorize=False,
                                       slp_vectorize=False)
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

    def compile_and_link(self, sources, opt=0, outfile='a.out'):
        objects = self._compiler.compile(sources, opt=opt)
        return self._linker.link(objects, outfile=outfile)


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
                 output_dir='.'):
        self._library_name = library_name
        self._translation_units = translation_units
        self._export_configuration = export_configuration
        self._python_cext = python_cext
        self._uuid = uuid
        self._opt = opt
        self._output_dir = output_dir

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
        compiler.compile_and_link(str(ir_mod), opt=self._opt, outfile=outpath)


class PIXIEModule(IRGenerator):

    def __init__(self, library_name, translation_units, export_configuration,
                 target_descr, uuid=None):
        self._library_name = library_name
        self._target_descr = target_descr
        self._uuid = uuid

        # convert translation units to a single module
        llvm_irs = [x._source for x in translation_units
                    if isinstance(x._source, str)]
        llvm_bcs = [x._source for x in translation_units
                    if isinstance(x._source, bytes)]
        self._user_source = {'llvm_ir': llvm_irs, 'llvm_bc': llvm_bcs}

        def _combine_sources():
            # don't create a new module for single source, this matters,
            # particularly for bitcode where a respecialization of a module
            # should embed the exact same bitcode as used to create the
            # original. A reparse of the bitcode can produce a valid but not
            # identical bitcode as to what was inputted.
            if (len(self._user_source['llvm_ir']) == 1 and
                    len(self._user_source['llvm_bc']) == 0):
                llvm_ir = self._user_source['llvm_ir'][0]
                self._single_mod = llvm.parse_assembly(llvm_ir)
                self._single_source = str(self._single_mod)
                self._single_bitcode = self._single_mod.as_bitcode()
            if (len(self._user_source['llvm_ir']) == 0 and
                    len(self._user_source['llvm_bc']) == 1):
                bc = self._user_source['llvm_bc'][0]
                self._single_mod = llvm.parse_bitcode(bc)
                self._single_source = str(self._single_mod)
                self._single_bitcode = bc
                return

            tmp_mod = ir.Module("tmp")
            # TODO: fix for x-compile
            tmp_mod.triple = llvm.get_process_triple()
            ir_module = llvm.parse_assembly(str(tmp_mod))
            mods = []
            for src in self._user_source['llvm_ir']:
                mods.append(llvm.parse_assembly(src))
            for src in self._user_source['llvm_bc']:
                mods.append(llvm.parse_bitcode(src))
            # link into main module
            for mod in mods:
                ir_module.link_in(mod)

            self._single_mod = ir_module
            self._single_source = str(self._single_mod)
            self._single_bitcode = self._single_mod.as_bitcode()

        _combine_sources()

        # get exports
        self._exported_symbols = defaultdict(list)
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
                compiler.compile_and_link(self._single_source, outfile=outfile)
                with open(outfile, 'rb') as f:
                    binaries[str(max(features)).upper()] = f.read()
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
                builder.call(fn, trampoline_fn.args)
                builder.ret_void()

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
        get_sz_name = 'get_bitcode_for_self_size'
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
        dso_handler = shmEmbeddedDSOHandler()
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

            if augment_existing():
                # Write a new PyInit which calls up the existing one and then
                # shoves the PIXIE dictionary into it
                # TODO: handle case where multiple PyInit_ exist in the same
                # source.
                gen = AugmentingPyInitGenerator(self._library_name,
                                                emap._embedded_libhandle_name)
            else:
                gen = AddPixieDictGenerator(self._library_name,
                                            self._exported_symbols,
                                            uuid=self._uuid)
            gen.generate_ir(mod)

        return mod
