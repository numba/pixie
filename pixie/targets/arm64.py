# arm64 - Apple silicon target
from enum import auto, Enum, IntEnum
from types import SimpleNamespace

from llvmlite import ir

from pixie.targets.common import (
    create_cpu_enum_for_target,
    FeaturesEnum,
    CPUDescription,
)
from pixie.selectors import Selector
from pixie.mcext import langref

# re-export
from .bsd_utils import sysctlbyname   # noqa: F401

from . import darwin_info


cpus = create_cpu_enum_for_target("arm64-unknown-unknown")


class features(FeaturesEnum):
    NONE = 0

    # features
    neon = 4   # same as fp_armv8
    dotprod = 5
    fullfp16 = 6
    fp16fml = 7
    sha3 = 8
    i8mm = 9
    bf16 = 10

    # microarch profile
    v8a = 11
    v8_1a = 12
    v8_2a = 13
    v8_3a = 14
    v8_4a = 15
    v8_5a = 16
    v8_6a = 17

    # TODO: implement
    feature_max       = auto()  # noqa: E221

    def as_feature_str(self):
        return self.name.replace('_', '.')


class cpu_features(IntEnum):
    NONE = 0

    # features
    neon = 4   # same as fp_armv8
    dotprod = 5
    fullfp16 = 6
    fp16fml = 7
    sha3 = 8
    i8mm = 9
    bf16 = 10

    # microarch profile
    v8a = 11
    v8_1a = 12
    v8_2a = 13
    v8_3a = 14
    v8_4a = 15
    v8_5a = 16
    v8_6a = 17


class cpu_dispatchable(IntEnum):
    neon = (1 << cpu_features.neon)
    sha3 = (1 << cpu_features.sha3)

    v8a = (1 << cpu_features.v8a) | neon
    v8_1a = (1 << cpu_features.v8_1a) | v8a
    v8_2a = (1 << cpu_features.v8_2a) | v8_1a
    v8_3a = (1 << cpu_features.v8_3a) | v8_2a

    dotprod = (1 << cpu_features.dotprod)
    fullfp16 = (1 << cpu_features.fullfp16)
    fp16fml = (1 << cpu_features.fp16fml)
    v8_4a = (1 << cpu_features.v8_4a) | v8_3a | fullfp16 | fp16fml | dotprod

    v8_5a = (1 << cpu_features.v8_5a) | v8_4a

    i8mm = (1 << cpu_features.i8mm)
    v8_6a = (1 << cpu_features.v8_6a) | v8_5a | i8mm

    bf16 = (1 << cpu_features.bf16)

    v8_6a_bf16 = v8_6a | bf16


_cd = cpu_dispatchable


class cpu_family_features(Enum):
    # M1: is +8.4a       +fp-armv8 +fp16fml +fullfp16 +sha3 +ssbs +sb +fptoint
    APPLE_M1 = _cd.v8_4a | _cd.sha3
    # M2: is +8.4a +8.6a +fp-armv8 +fp16fml +fullfp16 +sha3 +ssbs +sb +fptoint
    #        +bti +predres +i8mm +bf16
    APPLE_M2 = _cd.v8_6a | _cd.sha3 | _cd.bf16


_apple_m1 = CPUDescription(cpus.generic, (features.v8_4a,
                                          features.sha3,))
_apple_m2 = CPUDescription(cpus.generic, (features.v8_6a,
                                          features.sha3,
                                          features.bf16,))

# a set of predefined targets
predefined = SimpleNamespace(
    apple_m1=_apple_m1,
    apple_m2=_apple_m2,
)


class arm64CPUSelector(Selector):
    def selector_impl(self, builder):
        # Check the keys supplied are valid
        def check_keys():
            supplied_variants = set(self._data.keys())
            assert 'baseline' in supplied_variants, supplied_variants
            supplied_variants.remove('baseline')
            memb = cpu_dispatchable.__members__

            for k in supplied_variants:
                assert k in memb, f"{k} not in {memb.keys()}"

        check_keys()

        i32 = langref.types.i32
        i32_ptr = i32.as_pointer()
        i64 = langref.types.i64

        # commpage address
        commpage_addr = ir.Constant(i64, darwin_info.commpage_addr)
        cpu_family_offset = ir.Constant(
            i64, darwin_info.commpage_cpu_family_offset,
        )

        cpu_families = darwin_info.cpu_families

        def gen_cpu_family_probe(module):
            """
            Probe commpage cpu-family
            """
            ftype = ir.FunctionType(i32, ())
            fn = ir.Function(module, ftype,
                             name="_pixie_darwin_arm64_cpu_family_probe")
            PREFIX = f"[{fn.name}]"
            fn.linkage = 'internal'
            builder = ir.IRBuilder(fn.append_basic_block('entry'))
            cpu_fam_sel = builder.alloca(i32, name='cpu_fam_sel')
            builder.store(i32(0), cpu_fam_sel)
            commpage_cpu_fam_ptr = builder.inttoptr(
                commpage_addr.add(cpu_family_offset), i32_ptr,
                name="commpage_cpu_fam_ptr",
            )
            cpu_fam = builder.load(commpage_cpu_fam_ptr, name='cpu_fam')
            self.debug_print(builder, "{PREFIX} commpage value = %llu\n",
                             cpu_fam)

            for i, (name, cpuid) in enumerate(cpu_families.items()):
                matched = builder.icmp_unsigned('==', i32(cpuid), cpu_fam)
                with builder.if_then(matched):
                    builder.store(i32(i), cpu_fam_sel)
                    self.debug_print(builder, f"{PREFIX} matched {name}\n")

            output = builder.load(cpu_fam_sel, name='output')

            message = f"{PREFIX} output=%d\n"
            self.debug_print(builder, message, output)

            builder.ret(output)
            return fn

        supplied_variants = set(self._data.keys()) ^ {'baseline'}

        def cpu_release_order(*args):
            (feat,) = args
            return tuple(cpu_dispatchable.__members__.keys()).index(feat)

        variant_order = sorted(list(supplied_variants), key=cpu_release_order)

        fn_cpu_family_probe = gen_cpu_family_probe(builder.module)
        cpu_sel = builder.call(fn_cpu_family_probe, ())

        bb_default = builder.append_basic_block()
        swt = builder.switch(cpu_sel, bb_default)
        with builder.goto_block(bb_default):
            self.debug_print(builder, '[selector] select baseline\n')
            self._select(builder, 'baseline')  # should it be an error?
            builder.ret_void()

        def choose_variant(cpu_name) -> str | None:
            """Given the CPU family name, choose the dispatchable variant
            that matches the features of that CPU family.
            """
            features = cpu_family_features.__members__[cpu_name]
            for variant in reversed(variant_order):
                variant_feats = cpu_dispatchable[variant]
                anded = features.value & variant_feats.value
                if anded == variant_feats.value:
                    return variant

        # Match against CPU family and select the
        for i, name in enumerate(cpu_families):
            bb = builder.append_basic_block()
            with builder.goto_block(bb):
                self.debug_print(builder, f'[selector] cpu is {name}\n')
                variant = choose_variant(name)
                if variant is not None:
                    # matches a variant
                    self.debug_print(builder, f'[selector] select {variant}\n')
                    self._select(builder, variant)
                else:
                    # doesn't match any variant. choose baseline
                    self._select(builder, "baseline")
                builder.ret_void()
            swt.add_case(i, bb)

CPUSelector = arm64CPUSelector
