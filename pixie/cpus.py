from enum import IntEnum, auto
from functools import cached_property
from llvmlite import binding as llvm

# This enum is adapted from NumPy:
# https://github.com/numpy/numpy/blob/08e2a6ede4ebb074747b50128b19c5903a47e8ad/numpy/core/src/common/npy_cpu_features.h#L11-L100

class x86(IntEnum):
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


class Features():

    def __init__(self, features):
        if not isinstance(features, tuple):
            self.features = (features,)
        else:
            self.features = features

    def _host_cpu_features(self):
        return llvm.get_host_cpu_features()

    @cached_property
    def as_feature_flags(self):
        # get host features
        known_features = self._host_cpu_features()
        # set all to False
        for k in known_features.keys():
            known_features[k] = False
        # set these features to True
        for x in self.features:
            known_features[str(x)] = True
        return known_features.flatten()

    @cached_property
    def as_selected_feature_flags(self):
        # get host features
        known_features = self._host_cpu_features()
        # set all to False
        for k in known_features.keys():
            known_features[k] = False
        # set these features to True
        for x in self.features:
            known_features[str(x)] = True
        ret = ','.join(f'+{k}' for k, v in sorted(known_features.items()) if v)
        return ret

    def __str__(self):
        return self.as_selected_feature_flags

    def __repr__(self):
        return str(self)
