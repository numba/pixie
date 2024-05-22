# x86_64 target
from enum import auto, IntEnum, Enum
from llvmlite import ir
from pixie.selectors import Selector
from pixie.targets.common import create_cpu_enum_for_target, FeaturesEnum
from pixie.mcext import c, langref

cpus = create_cpu_enum_for_target("x86_64-unknown-unknown")


# This enum is adapted from NumPy:
# https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/numpy/core/src/common/npy_cpu_features.h#L11-L100


class features(FeaturesEnum):
    NONE = 0
    # X86
    mmx               = 1  # noqa: E221
    sse               = 2  # noqa: E221
    sse2              = 3  # noqa: E221
    sse3              = 4  # noqa: E221
    ssse3             = 5  # noqa: E221
    sse41             = 6  # noqa: E221
    popcnt            = 7  # noqa: E221
    sse42             = 8  # noqa: E221
    avx               = 9  # noqa: E221
    f16c              = 10  # noqa: E221
    xop               = 11  # noqa: E221
    fma4              = 12  # noqa: E221
    fma3              = 13  # noqa: E221
    avx2              = 14  # noqa: E221
    # avx2 & fma3, provides backward compatibility
    fma               = 15  # noqa: E221
    avx512f           = 30  # noqa: E221
    avx512cd          = 31  # noqa: E221
    avx512er          = 32  # noqa: E221
    avx512pf          = 33  # noqa: E221
    avx5124fmaps      = 34  # noqa: E221
    avx5124vnniw      = 35  # noqa: E221
    avx512vpopcntdq   = 36  # noqa: E221
    avx512bw          = 37  # noqa: E221
    avx512dq          = 38  # noqa: E221
    avx512vl          = 39  # noqa: E221
    avx512ifma        = 40  # noqa: E221
    avx512vbmi        = 41  # noqa: E221
    avx512vnni        = 42  # noqa: E221
    avx512vbmi2       = 43  # noqa: E221
    avx512bitalg      = 44  # noqa: E221
    feature_max       = auto()  # noqa: E221

    def __str__(self):
        return f'{self.name}'


class cpu_features(IntEnum):
    NONE = 0
    # X86
    MMX               = 1  # noqa E221
    SSE               = 2  # noqa E221
    SSE2              = 3  # noqa E221
    SSE3              = 4  # noqa E221
    SSSE3             = 5  # noqa E221
    SSE41             = 6  # noqa E221
    POPCNT            = 7  # noqa E221
    SSE42             = 8  # noqa E221
    AVX               = 9  # noqa E221
    F16C              = 10  # noqa E221
    XOP               = 11  # noqa E221
    FMA4              = 12  # noqa E221
    FMA3              = 13  # noqa E221
    AVX2              = 14  # noqa E221
    # AVX2 & FMA3, provides backward compatibility
    FMA               = 15  # noqa E221
    AVX512F           = 30  # noqa E221
    AVX512CD          = 31  # noqa E221
    AVX512ER          = 32  # noqa E221
    AVX512PF          = 33  # noqa E221
    AVX5124FMAPS      = 34  # noqa E221
    AVX5124VNNIW      = 35  # noqa E221
    AVX512VPOPCNTDQ   = 36  # noqa E221
    AVX512BW          = 37  # noqa E221
    AVX512DQ          = 38  # noqa E221
    AVX512VL          = 39  # noqa E221
    AVX512IFMA        = 40  # noqa E221
    AVX512VBMI        = 41  # noqa E221
    AVX512VNNI        = 42  # noqa E221
    AVX512VBMI2       = 43  # noqa E221
    AVX512BITALG      = 44  # noqa E221
    AVX512FP16        = 45  # noqa E221
    MAX               = auto()  # noqa E221


# These are effectively the features that NumPy cares about, see:
# https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/doc/source/reference/simd/gen_features.py # noqa: E501
# and
# https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/doc/source/reference/simd/generated_tables/cpu_features.inc # noqa: E501

# This defines bit vectors of "interesting" features, the "older"
# instruction sets are generally just supersets of one another. Since
# AVX512F this isn't the case, therefore the "interesting" features are
# grouped approximately by CPU codename. This enum essentially encodes:
# https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/doc/source/reference/simd/generated_tables/cpu_features.inc
class cpu_dispatchable(Enum):
    SSE               = (1 << cpu_features.SSE)  # noqa E221
    SSE2              = (1 << cpu_features.SSE2)     | SSE  # noqa E221
    SSE3              = (1 << cpu_features.SSE3)     | SSE | SSE2  # noqa E221
    SSSE3             = (1 << cpu_features.SSE3)     | SSE | SSE2 | SSE3  # noqa E221
    SSE41             = (1 << cpu_features.SSE41)    | SSE | SSE2 | SSE3 | SSSE3  # noqa E221
    POPCNT            = (1 << cpu_features.POPCNT)   | SSE | SSE2 | SSE3 | SSSE3 | SSE41  # noqa E221, E501
    SSE42             = (1 << cpu_features.SSE42)    | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT  # noqa E221, E501
    AVX               = (1 << cpu_features.AVX)      | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42  # noqa E221, E501
    XOP               = (1 << cpu_features.XOP)      | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX  # noqa E221, E501
    FMA4              = (1 << cpu_features.FMA4)     | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX  # noqa E221, E501
    F16C              = (1 << cpu_features.F16C)     | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX  # noqa E221, E501
    FMA3              = (1 << cpu_features.FMA3)     | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX | F16C  # noqa E221, E501
    AVX2              = (1 << cpu_features.AVX2)     | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX | F16C  # noqa E221, E501
    AVX512F           = (1 << cpu_features.AVX512F)  | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX | F16C | FMA3 | AVX2  # noqa E221, E501
    AVX512CD          = (1 << cpu_features.AVX512CD) | SSE | SSE2 | SSE3 | SSSE3 | SSE41 | POPCNT | SSE42 | AVX | F16C | FMA3 | AVX2 | AVX512F  # noqa E221, E501
    # X86 CPU Groups
    # Knights Landing (F,CD,ER,PF)
    AVX512_KNL        = AVX512CD | (1 << cpu_features.AVX512ER) | (1 << cpu_features.AVX512PF)  # noqa E221, E501
    # Knights Mill    (F,CD,ER,PF,4FMAPS,4VNNIW,VPOPCNTDQ), which is
    #                 AVX512_KNL + (4FMAPS,4VNNIW,VPOPCNTDQ)
    AVX512_KNM        = AVX512_KNL | (1 << cpu_features.AVX5124FMAPS) | (1 << cpu_features.AVX5124VNNIW) | (1 << cpu_features.AVX512VPOPCNTDQ)  # noqa E221, E501
    # Skylake-X       (F,CD,BW,DQ,VL)
    AVX512_SKX        = AVX512CD | (1 << cpu_features.AVX512VL) | (1 << cpu_features.AVX512BW) | (1 << cpu_features.AVX512DQ)  # noqa E221, E501
    # Cascade Lake    (F,CD,BW,DQ,VL,VNNI), which is AVX512_SKX + VNNI
    AVX512_CLX        = AVX512_SKX | (1 << cpu_features.AVX512VNNI)  # noqa E221
    # Cannon Lake     (F,CD,BW,DQ,VL,IFMA,VBMI), which is
    #                 AVX512_SKX + (IFMA, VBMI)
    AVX512_CNL        = AVX512_SKX | (1 << cpu_features.AVX512IFMA) | (1 << cpu_features.AVX512VBMI)  # noqa E221, E501
    # Ice Lake        (F,CD,BW,DQ,VL,IFMA,VBMI,VNNI,VBMI2,BITALG,VPOPCNTDQ),
    #                 which is AVX512_CNL + (VBMI2,BITALG,VPOPCNTDQ)
    AVX512_ICL        = AVX512_CNL | (1 << cpu_features.AVX512VBMI2) | (1 << cpu_features.AVX512BITALG) | (1 << cpu_features.AVX512VPOPCNTDQ)  # noqa E221, E501
    # Sapphire Rapids (Ice Lake + AVX512FP16)
    AVX512_SPR        = AVX512_ICL | (1 << cpu_features.AVX512FP16)  # noqa E221


class x86CPUSelector(Selector):

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

        # based on https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/numpy/core/src/common/npy_cpu_features.c # noqa: E501
        def gen_cpuid_probe(mod):
            # start cpuid_probe
            cpuid_probe_args = (langref.types.i32,) + (i32_ptr,) * 4
            cpuid_probe_fnty = ir.FunctionType(c.types.void, cpuid_probe_args)
            func_cpuid_probe = ir.Function(mod, cpuid_probe_fnty,
                                           name="cpuid_probe")
            func_cpuid_probe.attributes.add('noinline')  # this is for debug
            func_cpuid_probe.linkage = "internal"
            block = func_cpuid_probe.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            arg = builder.alloca(i32)
            self._ctx.init_alloca(builder, arg)
            builder.store(func_cpuid_probe.args[0], arg)
            fty = ir.FunctionType(ir.LiteralStructType([i32, i32, i32, i32]),
                                  (i32, i32))

            # TODO: Fix/investigate this...
            # This is supposed be the thing to do for PIC and should be the
            # equivalent to the following inline asm in C, however, it seemingly
            # causes segfaults.
            "xchg{l}\t{%%}ebx, %1\n\t"
            "cpuid\n\t"
            "xchg{l}\t{%%}ebx, %1\n\t"
            # asm_str = ("xchg$(l$)\t$(%$)ebx, $1\n\t"
            #            "cpuid\n\t"
            #            "xchg$(l$)\t$(%$)ebx, $1\n\t")
            # reg_info = ("={ax},=r,={cx},={dx},{ax},{cx},"
            #             "~{dirflag},~{fpsr},~{flags}")
            #
            # This is the non-PIC version, which seems to work ok.
            asm_str = "cpuid\n\t"
            reg_info = ("={ax},={bx},={cx},={dx},{ax},{cx},"
                        "~{dirflag},~{fpsr},~{flags}")
            result = builder.asm(fty, asm_str, reg_info,
                                 (builder.load(arg), ir.Constant(i32, 0)),
                                 False, name="asm_cpuid")
            for i in range(4):
                builder.store(builder.extract_value(result, i),
                              func_cpuid_probe.args[i + 1])
            builder.ret_void()
            return func_cpuid_probe
            # end cpuid_probe

        # stage call to cpuid_probe with arg 0
        self.debug_print(builder, "Staging cpuid probe")
        i32_zero = ir.Constant(i32, 0)
        func_cpuid_probe = gen_cpuid_probe(builder.module)
        r0to4 = [builder.alloca(i32) for _ in range(4)]
        [self._ctx.init_alloca(builder, x) for x in r0to4]
        builder.call(func_cpuid_probe, (i32_zero, *r0to4))

        # This might come back as zero on (assumably) v. old machines.
        pred = builder.icmp_signed("==", builder.load(r0to4[0]), i32_zero)
        with builder.if_then(pred):
            self.debug_print(builder, "Have MMX\n")
            self.debug_print(builder, "Have SSE\n")
            self.debug_print(builder, "Have SSE2\n")
            # if 64bit then this is likely...
            self.debug_print(builder, "Have SSE3\n")
            # return early selecting the baseline
            self._select(builder, 'baseline')
            self.debug_print(builder, "Early return with baseline\n")
            builder.ret_void()

        # Stage call to cpuid_probe with arg 1 and search
        i32_one = ir.Constant(i32, 1)
        [builder.store(i32_zero, x) for x in r0to4]
        builder.call(func_cpuid_probe, (i32_one, *r0to4))
        older_instructions = [("MMX   ", 23, 3),
                              ("SSE   ", 25, 3),
                              ("SSE2  ", 26, 3),
                              ("SSE3  ",  0, 2),
                              ("SSSE3 ",  9, 2),
                              ("SSE41 ", 19, 2),
                              ("POPCNT", 23, 2),
                              ("SSE42 ", 20, 2),
                              ("F16C  ", 29, 2),]
        for x in older_instructions:
            name, shift, reg = x
            shifted = builder.shl(i32_one, ir.Constant(i32, shift))
            feature = builder.and_(shifted, builder.load(r0to4[reg]))
            self.debug_print(builder, f"Have {name} %d\n",
                             builder.icmp_signed("!=", feature, i32_zero))

        # write the bit pattern into a uint64 based on the cpu_features enum.
        # features ordered from newest to oldest as msb to lsb, i.e. MMX. is
        # near lsb.

        # First generate feature named checking functions
        module = builder.module
        i8 = langref.types.i8

        def generate_feature_check(feat, shift):
            fnty = ir.FunctionType(i8, (i32,))
            func = ir.Function(module, fnty, name=f"check_feature_{feat}")
            func.attributes.add('alwaysinline')
            func.attributes.add('nounwind')
            func.linkage = "internal"
            block = func.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            shift_const = ir.Constant(i32, shift)
            shifted = builder.lshr(func.args[0], shift_const)
            truncd = builder.trunc(shifted, i8)
            result = builder.and_(truncd, ir.Constant(i8, 1))
            builder.ret(result)
            return func

        # Create CPU SSE probes
        # arg = 1
        features = [(cpu_features.MMX,    3, 23),
                    (cpu_features.SSE,    3, 25),
                    (cpu_features.SSE2,   3, 26),
                    (cpu_features.SSE3,   2, 0),
                    (cpu_features.SSSE3,  2, 9),
                    (cpu_features.SSE41,  2, 19),
                    (cpu_features.POPCNT, 2, 23),
                    (cpu_features.SSE42,  2, 20),
                    (cpu_features.F16C,   2, 29),]

        non_avx_call_map = {}
        for item in features:
            feat_id, reg, shift = item
            function = generate_feature_check(feat_id.name, shift)
            non_avx_call_map[feat_id] = (reg, function)

        # memset features, skip this?
        i64 = langref.types.i64
        features_vect = builder.alloca(i64,
                                       name="cpu_features_vect")
        self._ctx.init_alloca(builder, features_vect)
        builder.store(ir.Constant(i64, 0x0), features_vect)

        for k, v in non_avx_call_map.items():
            register, function = v
            result = builder.call(function, (builder.load(r0to4[register]),))
            idx = ir.Constant(i64, k.value)
            bit_loc = builder.shl(builder.zext(result, i64), idx)
            mask = builder.or_(builder.load(features_vect), bit_loc)
            builder.store(mask, features_vect)

        self.debug_print(builder, "Features vect %x\n",
                         builder.load(features_vect))

        supplied_variants = set(self._data.keys()) ^ {'baseline'}
        # this is a flag to indicate that some variant matched so don't use
        # the default, 0 = no match, !0 = match.
        found = builder.alloca(i8, name="found")
        self._ctx.init_alloca(builder, found)
        builder.store(ir.Constant(i8, 0), found)

        fv = builder.load(features_vect)

        def cpu_release_order(*args):
            (feat,) = args
            return tuple(cpu_dispatchable.__members__.keys()).index(feat)

        variant_order = sorted(list(supplied_variants), key=cpu_release_order)

        # This is an "escape hatch" for debug etc whereby the ISA is picked from
        # the environment variable PIXIE_USE_ISA (if it has been set).
        # TODO: use secure_getenv if available
        PIXIE_USE_ISA_env_var_str = self._ctx.insert_const_string(
            builder.module, "PIXIE_USE_ISA")
        # this returns a char * to the env var contents, or NULL if no match
        PIXIE_USE_ISA_value = c.stdlib.getenv(builder,
                                              PIXIE_USE_ISA_env_var_str)
        envvar_set_pred = builder.not_(
            self._ctx.is_null(builder, PIXIE_USE_ISA_value))
        self.debug_print(builder, "Testing for env var dispatch.\n")
        with builder.if_else(envvar_set_pred, likely=False) as (then,
                                                                otherwise):
            with then:
                self.debug_print(builder, "Using env var dispatch.\n")
                # TODO: this is like the env var selector impl, could that be
                # used?
                for specific_feature in variant_order:
                    zero = ir.Constant(c.types.int, 0)
                    max_len = ir.Constant(c.stddef.size_t, 255)
                    feature_name_str = self._ctx.insert_const_string(
                        builder.module, specific_feature)
                    strcmp_res = c.string.strncmp(builder, PIXIE_USE_ISA_value,
                                                  feature_name_str,
                                                  max_len)
                    pred = builder.icmp_signed("==", strcmp_res, zero)
                    with builder.if_then(pred):
                        msg = "Using version from env var: PIXIE_USE_ISA=%s\n"
                        self.debug_print(builder, msg, PIXIE_USE_ISA_value)
                        self._select(builder, specific_feature)
                        # mark having found a suitable match
                        builder.store(ir.Constant(i8, 1), found)
                        builder.ret_void()

                # check that a match was found and abort otherwise
                pred = builder.icmp_unsigned("==", builder.load(found),
                                             ir.Constant(i8, 0))
                with builder.if_then(pred, likely=False):
                    message = ("No matching library is available for ISA "
                               "\"%s\" supplied via environment variable "
                               "PIXIE_USE_ISA.\n"
                               "\nThis error is unrecoverable and the program "
                               "will now exit. Try checking that the supplied "
                               "ISA is valid and then rerun.\n")
                    error_message = self._ctx.insert_const_string(
                        builder.module, message)
                    c.stdio.printf(builder, error_message, PIXIE_USE_ISA_value)
                    # call sigabrt.
                    self.debug_print(builder, "calling exit")
                    c.stdlib.exit(builder, c.sysexits.EX_SOFTWARE)
                    builder.ret_void()
            with otherwise:
                self.debug_print(builder, "No env var set.\n")

        # The env var escape hatch wasn't used so do the standard dispatch
        for specific_feature in variant_order:
            disp_feat = ir.Constant(i64,
                                    cpu_dispatchable[specific_feature].value)
            mask = builder.and_(disp_feat, fv)
            pred = builder.icmp_unsigned("==", mask, disp_feat)
            with builder.if_else(pred) as (then, otherwise):
                with then:
                    msg = f"branch checking {specific_feature}: Success\n"
                    self.debug_print(builder, msg)
                    self._select(builder, specific_feature)
                    builder.store(ir.Constant(i8, 1), found)
                with otherwise:
                    msg = f"branch checking {specific_feature}: Failed\n"
                    self.debug_print(builder, msg)
                    # can just return
                    builder.ret_void()

        # if found is 0, didn't find anything acceptable, so return baseline,
        # this is mainly for debug, the "baseline" could just be set earlier.
        pred = builder.icmp_unsigned("==", builder.load(found),
                                     ir.Constant(i8, 0))
        with builder.if_then(pred, likely=False):
            self._select(builder, 'baseline')
        builder.ret_void()


CPUSelector = x86CPUSelector
