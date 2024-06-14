# x86_64 target
from enum import auto, Enum
from types import SimpleNamespace
from llvmlite import ir
from pixie.selectors import Selector
from pixie.targets.common import (create_cpu_enum_for_target, FeaturesEnum,
                                  CPUDescription)
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
    avx512fp16        = 45  # noqa: E221
    feature_max       = auto()  # noqa: E221

    def as_feature_str(self):
        if self.name == "sse41":
            return "sse4.1"
        elif self.name == "sse42":
            return "sse4.2"
        else:
            return f'{self.name}'

    def __str__(self):
        return f'{self.name}'


# NOTE: This enum isn't used yet. It needs support for multiple ISA's supplied
# in a single target description wiring through to here.
#
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
    sse               = (1 << features.sse)  # noqa E221
    sse2              = (1 << features.sse2)     | sse  # noqa E221
    sse3              = (1 << features.sse3)     | sse | sse2  # noqa E221
    ssse3             = (1 << features.sse3)     | sse | sse2 | sse3  # noqa E221
    sse41             = (1 << features.sse41)    | sse | sse2 | sse3 | ssse3  # noqa E221
    popcnt            = (1 << features.popcnt)   | sse | sse2 | sse3 | ssse3 | sse41  # noqa E221, E501
    sse42             = (1 << features.sse42)    | sse | sse2 | sse3 | ssse3 | sse41 | popcnt  # noqa E221, E501
    avx               = (1 << features.avx)      | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42  # noqa E221, E501
    xop               = (1 << features.xop)      | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx  # noqa E221, E501
    fma4              = (1 << features.fma4)     | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx  # noqa E221, E501
    f16c              = (1 << features.f16c)     | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx  # noqa E221, E501
    fma3              = (1 << features.fma3)     | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx | f16c  # noqa E221, E501
    avx2              = (1 << features.avx2)     | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx | f16c  # noqa E221, E501
    avx512f           = (1 << features.avx512f)  | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx | f16c | fma3 | avx2  # noqa E221, E501
    avx512cd          = (1 << features.avx512cd) | sse | sse2 | sse3 | ssse3 | sse41 | popcnt | sse42 | avx | f16c | fma3 | avx2 | avx512f  # noqa E221, E501
    # X86 CPU Groups
    # Knights Landing (F,CD,ER,PF)
    avx512_knl        = avx512cd | (1 << features.avx512er) | (1 << features.avx512pf)  # noqa E221, E501
    # Knights Mill    (F,CD,ER,PF,4FMAPS,4VNNIW,VPOPCNTDQ), which is
    #                 AVX512_KNL + (4FMAPS,4VNNIW,VPOPCNTDQ)
    avx512_knm        = avx512_knl | (1 << features.avx5124fmaps) | (1 << features.avx5124vnniw) | (1 << features.avx512vpopcntdq)  # noqa E221, E501
    # Skylake-X       (F,CD,BW,DQ,VL)
    avx512_skx        = avx512cd | (1 << features.avx512vl) | (1 << features.avx512bw) | (1 << features.avx512dq)  # noqa E221, E501
    # Cascade Lake    (F,CD,BW,DQ,VL,VNNI), which is AVX512_SKX + VNNI
    avx512_clx        = avx512_skx | (1 << features.avx512vnni)  # noqa E221
    # Cannon Lake     (F,CD,BW,DQ,VL,IFMA,VBMI), which is
    #                 AVX512_SKX + (IFMA, VBMI)
    avx512_cnl        = avx512_skx | (1 << features.avx512ifma) | (1 << features.avx512vbmi)  # noqa E221, E501
    # Ice Lake        (F,CD,BW,DQ,VL,IFMA,VBMI,VNNI,VBMI2,BITALG,VPOPCNTDQ),
    #                 which is AVX512_CNL + (VBMI2,BITALG,VPOPCNTDQ)
    avx512_icl        = avx512_cnl | (1 << features.avx512vbmi2) | (1 << features.avx512bitalg) | (1 << features.avx512vpopcntdq)  # noqa E221, E501
    # Sapphire Rapids (Ice Lake + AVX512FP16)
    avx512_spr        = avx512_icl | (1 << features.avx512fp16)  # noqa E221


class x86CPUSelector(Selector):

    _DEBUG = False

    def selector_impl(self, builder):
        # based on https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/numpy/core/src/common/npy_cpu_features.c # noqa: E501

        # Check the keys supplied are valid
        def check_keys():
            supplied_variants = set(self._data.keys())
            assert 'baseline' in supplied_variants, supplied_variants
            supplied_variants.remove('baseline')
            memb = features.__members__

            for k in supplied_variants:
                assert k in memb, f"{k} not in {memb.keys()}"

        check_keys()

        i1 = langref.types.i1
        i8 = langref.types.i8
        i32 = langref.types.i32
        i32_ptr = i32.as_pointer()

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

            # This is for 32-bit x86 PIC
            # "xchg{l}\t{%%}ebx, %1\n\t"
            # "cpuid\n\t"
            # "xchg{l}\t{%%}ebx, %1\n\t"
            # asm_str = ("xchg$(l$)\t$(%$)ebx, $1\n\t"
            #            "cpuid\n\t"
            #            "xchg$(l$)\t$(%$)ebx, $1\n\t")
            # reg_info = ("={ax},=r,={cx},={dx},{ax},{cx},"
            #             "~{dirflag},~{fpsr},~{flags}")
            #
            # This is for all 64 bit x86
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

        def gen_getxcr0_probe(mod):
            # This has to be done in asm, no named xgetbv on lots of platforms
            # the following is the equivalent to:
            #
            # int check(void) {
            #     unsigned int eax = 123, edx = 456;
            #     __asm__("xgetbv" : "=a"(eax), "=d"(edx) : "c" (0));
            #     return eax;
            # }
            #
            # verification of the byte sequence can be done by e.g. putting the
            # above source into `test_xgetbv.c`, then compiling with:
            #
            # $CC -O0 -g test_xgetbv.c -c
            #
            # and then using e.g. objdump to disassemble the binary to find the
            # byte sequence for the xgetbv instruction
            #
            # $ objdump -D test_xgetbv.o |grep xgetbv
            #
            # Example output:
            # test_xgetbv.o: file format elf64-x86-64
            # 19: 0f 01 d0    xgetbv
            getxcr0_args = ()
            getxcr0_fnty = ir.FunctionType(i32, getxcr0_args)
            func_getxcr0 = ir.Function(mod, getxcr0_fnty, name="getxcr0")
            func_getxcr0.attributes.add("noinline")
            func_getxcr0.linkage = "internal"
            block = func_getxcr0.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            fty = ir.FunctionType(ir.LiteralStructType([i32, i32]), (i32,))
            asm_str = ".byte 0x0f, 0x01, 0xd0"
            reg_info = "={ax},={dx},{cx},~{dirflag},~{fpsr},~{flags}"
            result = builder.asm(fty, asm_str, reg_info, (ir.Constant(i32, 0),),
                                 False, name="asm_getxcr0")
            builder.ret(builder.extract_value(result, 0))
            return func_getxcr0

        def cpu_release_order(*args):
            (feat,) = args
            # from pixie.targets.x86_64 import features
            return features[feat.lower()]

        supplied_variants = set(self._data.keys()) ^ {'baseline'}
        variant_order = sorted(list(supplied_variants), key=cpu_release_order)

        # this is a flag to indicate that some variant matched so don't use
        # the default, 0 = no match, !0 = match.
        found = builder.alloca(i8, name="found")
        self._ctx.init_alloca(builder, found)
        builder.store(ir.Constant(i8, 0), found)

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

        # first, set up a block for the case where the tree exits early, this
        # will be filled in later.
        tree_exit = builder.append_basic_block("tree_exit")

        # This is the features vector, it records the features it finds as a bit
        # index based on the enum above.
        i64 = langref.types.i64
        features_vect = builder.alloca(i64,
                                       name="cpu_features_vect")
        self._ctx.init_alloca(builder, features_vect)
        builder.store(ir.Constant(i64, 0x0), features_vect)

        # register slots for cpuid calls
        r0to4 = [builder.alloca(i32) for _ in range(4)]
        [self._ctx.init_alloca(builder, x) for x in r0to4]

        # stage call to cpuid_probe with arg 0
        self.debug_print(builder, "Staging cpuid probe\n")
        func_cpuid_probe = gen_cpuid_probe(builder.module)

        i32_zero = ir.Constant(i32, 0)
        builder.call(func_cpuid_probe, (i32_zero, *r0to4))

        # This might come back as zero on (assumably) v. old machines.
        pred = builder.icmp_signed("==", builder.load(r0to4[0]), i32_zero)
        with builder.if_then(pred):
            # TODO: This needs to write in MMX, SSE, SSE2 and SSE3 to the
            # feature_vector and then jump to tree_exit
            self.debug_print(builder, "Have MMX\n")
            self.debug_print(builder, "Have SSE\n")
            self.debug_print(builder, "Have SSE2\n")
            # if 64bit then this is likely...
            self.debug_print(builder, "Have SSE3\n")
            # return early selecting the baseline
            self._select(builder, 'baseline')
            self.debug_print(builder, "Early return with baseline\n")
            builder.ret_void()

        # write the bit pattern into a uint64 based on the cpu_features enum.
        # features ordered from newest to oldest as msb to lsb, i.e. MMX. is
        # near lsb.

        # First generate feature named checking functions
        module = builder.module

        def generate_feature_check(feat, shift):
            # This generates a function that does a rshift on the arg by the
            # constant declared in "shift" and truncates the result to a single
            # bit.
            fnty = ir.FunctionType(i1, (i32,))
            func = ir.Function(module, fnty, name=f"check_feature_{feat}")
            func.attributes.add('alwaysinline')
            func.attributes.add('nounwind')
            func.linkage = "internal"
            block = func.append_basic_block(name="entry")
            builder = ir.IRBuilder(block)
            shift_const = ir.Constant(i32, shift)
            shifted = builder.lshr(func.args[0], shift_const)
            result = builder.trunc(shifted, i1)
            self.debug_print(builder, f"Have {feat} %d\n",
                             builder.icmp_signed("!=", result, i32_zero))
            builder.ret(result)
            return func

        def probe_and_decode(cpuid_arg, lfeatures, features_vect):
            i32_cpuid_arg = ir.Constant(i32, cpuid_arg)
            [builder.store(i32_zero, x) for x in r0to4]
            builder.call(func_cpuid_probe, (i32_cpuid_arg, *r0to4))

            for item in lfeatures:
                feat_id, reg, shift = item
                function = generate_feature_check(feat_id.name, shift)
                result = builder.call(function, (builder.load(r0to4[reg]),))
                idx = ir.Constant(i64, feat_id.value)
                bit_loc = builder.shl(builder.zext(result, i64), idx)
                mask = builder.or_(builder.load(features_vect), bit_loc)
                builder.store(mask, features_vect)

        # Create CPU SSE probes
        # cpuid arg = 1
        lfeatures = [(features.mmx,    3, 23),
                     (features.sse,    3, 25),
                     (features.sse2,   3, 26),
                     (features.sse3,   2, 0),
                     (features.ssse3,  2, 9),
                     (features.sse41,  2, 19),
                     (features.popcnt, 2, 23),
                     (features.sse42,  2, 20),
                     (features.f16c,   2, 29),]

        probe_and_decode(1, lfeatures, features_vect)

        # This is a synopsis of the logic suggested in the "Intel 64 and IA-32
        # Architectures Software Developer's Manual" Order Number 325462-083US
        # March 2024. Page 362. Section 14.3 "Detection of Intel AVX
        # Instructions".
        #
        # AVX requires:
        # 1. OSXSAVE (%ecx (register[2]), bit 27) is set. This is the OS
        #    level signal that X{SET,GET}BV instructions can access XCR0
        # 2. XCR0.SSE (bit 1) and XCR0.AVX (bit 2) is set
        #    (https://en.wikipedia.org/wiki/Control_register#Additional_Control_registers_in_Intel_x86-64_series),
        #    this is in effect a match of mask 0x6. This is the OS level signal
        #    that XSAVE/XRSTOR for ymm and zmm registers used by AVX is
        #    supported. Run `xgetbv(0)` and decode the state in the return.
        # 3. CPUID.1:ECX.AVX (%ecx (register[2]), bit 28) is set. This is the
        #    processor itself declaring the AVX is supported in the hardware.

        # 1. OSXSAVE check
        function = generate_feature_check("OSXSAVE", 27)
        OSXSAVE = builder.call(function, (builder.load(r0to4[2]),))
        with builder.if_then(builder.not_(OSXSAVE)):
            self.debug_print(builder, "No OSXSAVE, exit now.\n")
            builder.branch(tree_exit)

        # 2. XCR0 check, call gen_getxcr0_probe, mask result against 0x6.
        getxcr0_func = gen_getxcr0_probe(module)
        xcr0 = builder.call(getxcr0_func, ())
        six = ir.Constant(i32, 6)
        no_avx_os = builder.icmp_unsigned("!=", builder.and_(xcr0, six), six)
        with builder.if_then(no_avx_os):
            self.debug_print(builder, "No XCR0.SEE/XCR0.AVX, exit now.\n")
            builder.branch(tree_exit)

        # 3. check AVX CPU support
        function = generate_feature_check("AVX", 28)
        HAVE_AVX = builder.call(function, (builder.load(r0to4[2]),))
        with builder.if_then(builder.not_(HAVE_AVX)):
            self.debug_print(builder, "No AVX CPU support, exit now.\n")
            builder.branch(tree_exit)
        idx = ir.Constant(i64, features.avx.value)
        bit_loc = builder.shl(builder.zext(HAVE_AVX, i64), idx)
        mask = builder.or_(builder.load(features_vect), bit_loc)
        builder.store(mask, features_vect)

        # Create AMD extension probes
        # cpuid arg = 0x80000001
        lfeatures = [(features.xop,    2, 11),
                     (features.fma4,   2, 16),]
        probe_and_decode(0x80000001, lfeatures, features_vect)

        # Create AVX2 and AVX512F extension probes
        # cpuid arg = 7
        lfeatures = [(features.avx2,   1, 5),
                     (features.avx512f,   1, 16),
                     (features.avx512cd,   1, 28),
                     (features.avx512pf,   1, 26),
                     (features.avx512er,   1, 27),
                     (features.avx512vpopcntdq,   2, 14),
                     (features.avx5124vnniw,   3, 2),
                     (features.avx5124fmaps,   3, 3),
                     (features.avx512dq,   1, 17),
                     (features.avx512bw,   1, 30),
                     (features.avx512vl,   1, 31),
                     (features.avx512vnni,   2, 11),
                     (features.avx512ifma,   1, 21),
                     (features.avx512vbmi,   2, 1),
                     (features.avx512vbmi2,   2, 6),
                     (features.avx512bitalg,   2, 12),
                     (features.avx512fp16,   3, 23),]
        probe_and_decode(7, lfeatures, features_vect)

        self.debug_print(builder, "Features vect 0x%llx\n",
                         builder.load(features_vect))
        # Jump to the tree exit.
        builder.branch(tree_exit)

        # This completes the probes, the features_vect is now populated with
        # what this CPU supports.

        # Now to fill in tree_exit block, this block processes the
        # feature_vector and selects the embedded library that has the best
        # matching set of features for the CPU.
        builder.position_at_end(tree_exit)

        fv = builder.load(features_vect)

        # The env var escape hatch wasn't used so do the standard dispatch.
        for specific_feature in variant_order:
            disp_feat = ir.Constant(i64,
                                    features[specific_feature].value)
            mask = builder.shl(ir.Constant(i64, 1), disp_feat)
            self.debug_print(builder, "disp feat = %llx, mask = %llx\n",
                             disp_feat, mask)
            # does the feature vect match the mask?
            pred = builder.icmp_unsigned("==", mask, builder.and_(mask, fv))
            with builder.if_else(pred) as (then, otherwise):
                with then:
                    msg = f"branch checking {specific_feature}: Success\n"
                    self.debug_print(builder, msg)
                    self._select(builder, specific_feature)
                    builder.store(ir.Constant(i8, 1), found)
                with otherwise:
                    msg = f"branch checking {specific_feature}: Failed\n"
                    self.debug_print(builder, msg)
                    # if `found` is non-zero it's ok to just return here,
                    # something previously matched
                    pred = builder.icmp_unsigned("==", builder.load(found),
                                                 ir.Constant(i8, 0))
                    with builder.if_then(pred, likely=True):
                        builder.ret_void()

        # if found is 0, didn't find anything acceptable, so return baseline,
        # this is mainly for debug, the "baseline" could just be set earlier.
        pred = builder.icmp_unsigned("==", builder.load(found),
                                     ir.Constant(i8, 0))
        with builder.if_then(pred, likely=False):
            self._select(builder, 'baseline')
        builder.ret_void()


CPUSelector = x86CPUSelector

# These are the psABI variants. See something like:
# https://github.com/archspec/archspec-json/blob/80ce086dd8a981955bb2048561e27b4159b97440/cpu/microarchitectures.json#L47-L364
# for the feature groupings.
_x86_64 = CPUDescription(cpus.generic, (features.sse,
                                        features.sse2,))

_x86_64_v2 = CPUDescription(cpus.generic, (_x86_64.features + (features.ssse3,
                                                               features.sse41,
                                                               features.sse42,
                                                               features.popcnt)
                                           ))

_x86_64_v3 = CPUDescription(cpus.generic, _x86_64_v2.features + (
    features.avx,
    features.avx2,
    features.f16c,
    features.fma,
    ))

_x86_64_v4 = CPUDescription(cpus.generic, _x86_64_v3.features + (
    features.avx512f,
    features.avx512bw,
    features.avx512cd,
    features.avx512dq,
    features.avx512vl,
    ))


# a set of predefined targets
predefined = SimpleNamespace(x86_64=_x86_64,
                             x86_64_v1=_x86_64,
                             x86_64_v2=_x86_64_v2,
                             x86_64_v3=_x86_64_v3,
                             x86_64_v4=_x86_64_v4)

# extend to true psABI "spelling" (it uses dashes `-`).
predefined.__dict__["x86-64"] = _x86_64
predefined.__dict__["x86-64-v2"] = _x86_64_v2
predefined.__dict__["x86-64-v3"] = _x86_64_v3
predefined.__dict__["x86-64-v4"] = _x86_64_v4

del _x86_64, _x86_64_v2, _x86_64_v3, _x86_64_v4


# Default for x86_64 will compile for all psABI variants as targets, with the
# baseline set as the psABI x86_64 variant.
default_configuration = {'baseline_cpu': predefined.x86_64.cpu,
                         'baseline_features': predefined.x86_64.features,
                         'targets_features': (predefined.x86_64_v2,
                                              predefined.x86_64_v3,
                                              predefined.x86_64_v4)}
