# aarch64 target
from pixie.targets.common import create_cpu_enum_for_target, FeaturesEnum
from pixie.selectors import Selector

cpus = create_cpu_enum_for_target("aarch64-unknown-unknown")


class features(FeaturesEnum):
    NONE = 0
    # TODO: implement
    pass


class aarch64CPUSelector(Selector):
    pass


def selector_impl(self, builder):
    # TODO: implementation
    pass


CPUSelector = aarch64CPUSelector
