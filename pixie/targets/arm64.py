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
from pixie.mcext import c, langref, get_or_insert_function


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


class cpu_dispatchable(IntEnum):
    neon = (1 << features.neon)
    sha3 = (1 << features.sha3)

    v8a = (1 << features.v8a) | neon
    v8_1a = (1 << features.v8_1a) | v8a
    v8_2a = (1 << features.v8_2a) | v8_1a
    v8_3a = (1 << features.v8_3a) | v8_2a

    dotprod = (1 << features.dotprod)
    fullfp16 = (1 << features.fullfp16)
    fp16fml = (1 << features.fp16fml)
    v8_4a = (1 << features.v8_4a) | v8_3a | fullfp16 | fp16fml | dotprod

    v8_5a = (1 << features.v8_5a) | v8_4a

    i8mm = (1 << features.i8mm)
    v8_6a = (1 << features.v8_6a) | v8_5a | i8mm

    bf16 = (1 << features.bf16)

    v8_6a_bf16 = v8_6a | bf16


_cd = cpu_dispatchable


class cpu_family_features(Enum):
    # M1: is +8.4a       +fp-armv8 +fp16fml +fullfp16 +sha3 +ssbs +sb +fptoint
    APPLE_M1 = _cd.V8_4A | _cd.SHA3
    # M2: is +8.4a +8.6a +fp-armv8 +fp16fml +fullfp16 +sha3 +ssbs +sb +fptoint
    #        +bti +predres +i8mm +bf16
    APPLE_M2 = _cd.V8_6A | _cd.SHA3 | _cd.BF16


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

# Default for arm64 will compile for M1 and with additional targets for M2
default_configuration = {'baseline_cpu': predefined.apple_m1.cpu,
                         'baseline_features': predefined.apple_m1.features,
                         'targets_features': (predefined.apple_m2,)}


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

        i8 = langref.types.i8
        i32 = langref.types.i32

        # this is a flag to indicate that some variant matched so don't use
        # the default, 0 = no match, !0 = match.
        found = builder.alloca(i8, name="found")
        self._ctx.init_alloca(builder, found)
        builder.store(ir.Constant(i8, 0), found)

        cpu_list = [
            "APPLE_M1",
            "APPLE_M2",
        ]

        def call_sysctlbyname_hw_feat(builder):
            """Call sysctlbyname for hardware feature.

            Similar to: https://github.com/llvm/llvm-project/blob/8d237190ecc4ac90865d80dbb11a34c6719b406d/compiler-rt/lib/builtins/cpu_model/aarch64/fmv/apple.inc#L25-L30  # noqa: 501
            """
            module = builder.module
            size_t = c.stddef.size_t
            i8ptr = i8.as_pointer()

            def gen_sysctlbyname(module):
                # int
                # sysctlbyname(const char *, void *, size_t *, void *, size_t)
                ftype_sysctlbyname = ir.FunctionType(
                    i32,
                    [i8ptr,                # name
                     i8ptr,                # oldp
                     size_t.as_pointer(),  # oldlenp
                     i8ptr,                # newp
                     size_t]               # newlen
                )
                sysctlbyname = get_or_insert_function(
                    module, ftype_sysctlbyname, name='sysctlbyname')
                ftype = ir.FunctionType(i32, [i8ptr])
                fn = get_or_insert_function(module, ftype,
                                            name='_pixie_wrap_sysctlbyname')
                if fn.is_declaration:
                    fn.linkage = 'internal'
                    builder = ir.IRBuilder(fn.append_basic_block())
                    [arg_name] = fn.args
                    oldp = builder.alloca(i32)
                    oldlenp = builder.alloca(size_t)

                    builder.store(i32(0), oldp)

                    # sizeof i32
                    i32ptr = i32.as_pointer()
                    size = builder.ptrtoint(builder.gep(i32ptr(None),
                                                        [i32(1)]), size_t)
                    builder.store(size, oldlenp)

                    ret = builder.call(sysctlbyname, [
                        arg_name,
                        builder.bitcast(oldp, i8ptr),
                        oldlenp,
                        i8ptr(None),
                        size_t(0),
                    ])
                    failed = builder.icmp_unsigned('!=', ret, ret.type(0))
                    with builder.if_then(failed):
                        msg = self._ctx.insert_const_string(
                            module, "sysctlbyname failed: "
                        )
                        c.stdio.perror(builder, msg)
                        c.stdlib.abort(builder)
                    builder.ret(builder.load(oldp))
                return fn

            hw_feat_check_fn = gen_sysctlbyname(module)

            # Use CPU features to determine the CPU model.
            # DotProd implies M1
            # BF15 implies M2
            feature_list = {
                # The feature names are from
                # https://github.com/llvm/llvm-project/blob/cad72632eb0d612fe18c38ac4526d80a6b800f96/compiler-rt/lib/builtins/cpu_model/aarch64/fmv/apple.inc#L117-L146 # noqa: 501
                "hw.optional.arm.FEAT_DotProd": "APPLE_M1",
                "hw.optional.arm.FEAT_BF16":    "APPLE_M2",
            }
            cpu_sel_type = i8
            cpu_selected_ptr = builder.alloca(cpu_sel_type)
            builder.store(cpu_sel_type(-1), cpu_selected_ptr)

            for feat, cpu_name in feature_list.items():
                hw_name = self._ctx.insert_const_string(builder.module, feat)
                out = builder.call(hw_feat_check_fn, [hw_name])
                msg = f"[selector] sysctlbyname({feat}) -> %d\n"
                self.debug_print(builder, msg, out)
                idx = cpu_list.index(cpu_name)
                is_set = builder.icmp_unsigned('!=', out, out.type(0))
                with builder.if_then(is_set):
                    builder.store(cpu_sel_type(idx), cpu_selected_ptr)

            cpu_selected = builder.load(cpu_selected_ptr)
            return cpu_selected

        supplied_variants = set(self._data.keys()) ^ {'baseline'}

        def cpu_release_order(*args):
            (feat,) = args
            return tuple(cpu_dispatchable.__members__.keys()).index(feat)

        variant_order = sorted(list(supplied_variants), key=cpu_release_order)
        self._generate_env_var_check(builder, found, variant_order)

        cpu_sel = call_sysctlbyname_hw_feat(builder)

        bb_default = builder.append_basic_block()
        swt = builder.switch(cpu_sel, bb_default)
        with builder.goto_block(bb_default):
            self.debug_print(builder, '[selector] unknown cpu\n')
            # call sigabrt.
            self.debug_print(builder, "calling exit\n")
            c.stdlib.exit(builder, c.sysexits.EX_SOFTWARE)
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

        # Match against CPU family
        for i, name in enumerate(cpu_list):
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
