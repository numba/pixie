from pixie.targets.common import (CPUDescription, TargetDescription,
                                  decode_llvm_triple, llvm_triple)
from pixie.tests.support import PixieTestCase
from pixie.targets.x86_64 import cpus, features
import unittest


class TestTargetDescription(PixieTestCase):
    # this tests that the numerous "spellings" for a target description coming
    # in from the frontend are handled correctly.

    def check(self, td, *, baseline_target, additional_targets):
        assert td.baseline_target == baseline_target
        for x, y in zip(td.additional_targets, additional_targets, strict=True):
            assert x == y

    def test_strings(self):
        triple = "x86_64-unknown-linux-gnu"
        nocona_sse3 = CPUDescription(cpus.nocona, (features.sse3,))
        descr = TargetDescription(triple, "nocona", "sse3", "")
        self.check(descr, baseline_target=nocona_sse3, additional_targets=())

        descr = TargetDescription(triple, "nocona", "sse3", "avx")
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),))

        descr = TargetDescription(triple, "nocona", ("sse3", "sse42"), "avx")
        self.check(descr,
                   baseline_target=CPUDescription(cpus.nocona,
                                                  (features.sse3,
                                                   features.sse42)),
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),))

        descr = TargetDescription(triple, "nocona", "sse3", ("avx", "avx2"))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),
                                       CPUDescription(cpus.nocona,
                                                      (features.avx2,)),))

        descr = TargetDescription(triple, "nocona", "sse3",
                                  (("cascadelake", "avx"),
                                   ("tigerlake", "avx2")))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.cascadelake,
                                                      (features.avx,)),
                                       CPUDescription(cpus.tigerlake,
                                                      (features.avx2,)),))

        descr = TargetDescription(triple, "nocona", "sse3",
                                  (("cascadelake", ("avx", "avx2")),
                                   ("tigerlake", ("sse42", "avx2")),))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.cascadelake,
                                                      (features.avx,
                                                       features.avx2,)),
                                       CPUDescription(cpus.tigerlake,
                                                      (features.sse42,
                                                       features.avx2,)),))

    def test_enums(self):

        triple = "x86_64-unknown-linux-gnu"
        nocona_sse3 = CPUDescription(cpus.nocona, (features.sse3,))
        descr = TargetDescription(triple, cpus.nocona, features.sse3, ())
        self.check(descr, baseline_target=nocona_sse3, additional_targets=())

        descr = TargetDescription(triple, cpus.nocona, features.sse3,
                                  features.avx)
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),))

        descr = TargetDescription(triple, cpus.nocona,
                                  (features.sse3, features.sse42),
                                  features.avx)
        self.check(descr, baseline_target=CPUDescription(cpus.nocona,
                                                         (features.sse3,
                                                          features.sse42)),
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),))

        descr = TargetDescription(triple, cpus.nocona, features.sse3,
                                  (features.avx, features.avx2))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.nocona,
                                                      (features.avx,)),
                                       CPUDescription(cpus.nocona,
                                                      (features.avx2,)),))

        descr = TargetDescription(triple, cpus.nocona, features.sse3,
                                  ((cpus.cascadelake, features.avx),
                                   (cpus.tigerlake, features.avx2)))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.cascadelake,
                                                      (features.avx,)),
                                       CPUDescription(cpus.tigerlake,
                                                      (features.avx2,)),))

        descr = TargetDescription(triple, cpus.nocona, features.sse3,
                                  ((cpus.cascadelake,
                                    (features.avx, features.avx2,)),
                                   (cpus.tigerlake,
                                    (features.sse42, features.avx2,))))
        self.check(descr, baseline_target=nocona_sse3,
                   additional_targets=(CPUDescription(cpus.cascadelake,
                                                      (features.avx,
                                                       features.avx2,)),
                                       CPUDescription(cpus.tigerlake,
                                                      (features.sse42,
                                                       features.avx2,)),))


class TestTripleDecode(PixieTestCase):

    def test_3_triple_decode(self):
        triple = decode_llvm_triple("x86_64-unknown-linux")
        assert triple == llvm_triple("x86_64", "", "unknown", "linux", "")

    def test_4_triple_decode(self):
        triple = decode_llvm_triple("x86_64-unknown-linux-gnu")
        assert triple == llvm_triple("x86_64", "", "unknown", "linux", "gnu")

    def test_5_triple_decode(self):
        triple = decode_llvm_triple("arm-v5-unknown-linux-eabi")
        assert triple == llvm_triple("arm", "v5", "unknown", "linux", "eabi")


if __name__ == '__main__':
    unittest.main()
