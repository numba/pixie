from pixie import cpus
from pixie.codegen_helpers import Codegen
from pixie.dso_tools import ElfMapper, shmEmbeddedDSOHandler
from pixie.mcext import c
from pixie.platform import Toolchain
from pixie.selectors import PyVersionSelector, x86CPUSelector
from pixie.tests.support import PixieTestCase
from llvmlite import ir
from llvmlite import binding as llvm
import ctypes
import os
import random
import sys
import tempfile
import uuid

# TODO: use the compiler from pixie/compiler.py directly


class SimpleCompiler():
    # takes llvm_ir, compiles it to an object file
    def __init__(self, target_cpu, target_features):
        self._target_cpu = target_cpu
        self._target_features = target_features

    def compile(self, sources):
        # takes sources, returns object files
        objects = []
        codegen = Codegen(str(uuid.uuid4().hex),
                          cpu_name=self._target_cpu,
                          target_features=self._target_features)
        for source in sources:
            codelibrary = codegen.create_library(uuid.uuid4().hex)
            if isinstance(source, str):
                mod = llvm.parse_assembly(source)
            elif isinstance(source, bytes):
                mod = llvm.parse_bitcode(source)
            else:
                assert 0, f"Unknown source type {type(source)}"
            codelibrary.add_llvm_module(mod)
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
            self._toolchain.link_shared(outfile, objfiles)


class SimpleCompilerDriver():
    # like e.g. clang or gcc, compiles and links source translation units to
    # a DSO.
    def __init__(self, target_cpu, target_features):
        self._compiler = SimpleCompiler(target_cpu, target_features)
        self._linker = SimpleLinker()

    def compile_and_link(self, sources, outfile='a.out'):
        return self._linker.link(self._compiler.compile(sources),
                                 outfile=outfile)


class TestSelectors(PixieTestCase):

    def gen_dispatch_and_expected(self, dispatch_keys):
        expected = {}
        dispatch_data = {}
        for dispatchable in dispatch_keys:
            value, dso = self.generate_embeddable_dso()
            expected[dispatchable] = value
            dispatch_data[dispatchable] = dso
        return expected, dispatch_data

    def gen_mod(self, binaries, selector_class, dso_handler):
        mod = ir.Module()
        mod.triple = llvm.get_process_triple()
        emap = ElfMapper(mod)

        # create the DSO constructor, it does the select and dispatch
        emap.create_dso_ctor(binaries, selector_class, dso_handler)
        # create the DSO destructor, it cleans up the resources used by the
        # create_dso_ctor.
        emap.create_dso_dtor(dso_handler)

        return str(mod)

    def generate_embeddable_dso(self):
        """This needs to produce a DSO as bytes. In the DSO there needs to be a
        function foo that returns an int value."""
        mod = ir.Module()
        mod.triple = llvm.get_process_triple()
        fn = ir.Function(mod, ir.FunctionType(c.types.int, ()),
                         name="foo")
        block = fn.append_basic_block("entry_block")
        builder = ir.IRBuilder(block)
        value = random.randint(0, 2**31-1)  # C-int range
        builder.ret(ir.Constant(c.types.int, value))

        dso = os.path.join(self.tmpdir.name, uuid.uuid4().hex)
        target_features = cpus.Features((cpus.x86.sse2,))
        compiler_driver = SimpleCompilerDriver(target_cpu='nocona',
                                               target_features=target_features)
        compiler_driver.compile_and_link(sources=(str(mod),), outfile=dso)
        with open(dso, 'rb') as f:
            dso_bytes = f.read()

        return value, dso_bytes

    def test_pyversion_selector(self):

        dispatch_keys = ('3.9', '3.10', '3.11', '3.12')
        expected, dispatch_data = self.gen_dispatch_and_expected(dispatch_keys)

        selector_class = PyVersionSelector
        dso_handler = shmEmbeddedDSOHandler()
        llvm_ir = self.gen_mod(dispatch_data, selector_class, dso_handler)

        dso = os.path.join(self.tmpdir.name, uuid.uuid4().hex)
        target_features = cpus.Features((cpus.x86.sse2,))
        compiler_driver = SimpleCompilerDriver(target_cpu='nocona',
                                               target_features=target_features)
        compiler_driver.compile_and_link(sources=(llvm_ir,), outfile=dso)

        # check the DSO loads appropriately.
        binding = ctypes.CDLL(dso)

        uniq_filepath = dso_handler._EXTRACTED_FILEPATH
        uniq_filepath_global = getattr(binding, uniq_filepath.name)
        extracted_embedded_dso_path_bytes = ctypes.cast(uniq_filepath_global,
                                                        ctypes.c_char_p).value
        extracted_embedded_dso_path = extracted_embedded_dso_path_bytes.decode()
        extracted_embedded_dso = ctypes.CDLL(extracted_embedded_dso_path)
        extracted_embedded_dso.foo.restype = ctypes.c_int
        extracted_embedded_dso.foo.argtypes = ()

        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert extracted_embedded_dso.foo() == expected[pyver]

    def test_x86_isa_selector(self):

        dispatch_keys = ('baseline', 'SSE2', 'SSE3', 'SSE42', 'AVX')
        expected, dispatch_data = self.gen_dispatch_and_expected(dispatch_keys)

        selector_class = x86CPUSelector
        dso_handler = shmEmbeddedDSOHandler()
        llvm_ir = self.gen_mod(dispatch_data, selector_class, dso_handler)

        # Compile into DSO
        dso = os.path.join(self.tmpdir.name, uuid.uuid4().hex)
        target_features = cpus.Features((cpus.x86.sse2,))
        compiler_driver = SimpleCompilerDriver(target_cpu='nocona',
                                               target_features=target_features)
        compiler_driver.compile_and_link(sources=(llvm_ir,), outfile=dso)

        # check the DSO loads appropriately.
        binding = ctypes.CDLL(dso)

        uniq_filepath = dso_handler._EXTRACTED_FILEPATH
        uniq_filepath_global = getattr(binding, uniq_filepath.name)
        extracted_embedded_dso_path_bytes = ctypes.cast(uniq_filepath_global,
                                                        ctypes.c_char_p).value
        extracted_embedded_dso_path = extracted_embedded_dso_path_bytes.decode()
        extracted_embedded_dso = ctypes.CDLL(extracted_embedded_dso_path)
        extracted_embedded_dso.foo.restype = ctypes.c_int
        extracted_embedded_dso.foo.argtypes = ()

        # look for the highest available feature that is also in the dispatch
        # list.
        from numpy.core._multiarray_umath import __cpu_features__
        highest_feature = None
        for isa, present in __cpu_features__.items():
            if present and isa in expected:
                highest_feature = isa

        assert highest_feature is not None
        assert extracted_embedded_dso.foo() == expected[highest_feature]
