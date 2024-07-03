from pixie import PIXIECompiler, TranslationUnit
from pixie.tests.support import PixieTestCase
import unittest


llvm_foo_double_double = """
    define void @"_Z3fooPdS_"(double* %".1", double* %".2", double* %".out")
    {
    entry:
        %.3 = load double, double * %.1
        %.4 = load double, double * %.2
        %"res" = fadd double %".3", %".4"
        store double %"res", double* %".out"
        ret void
    }
    """


class TestCompilerInterface(PixieTestCase):

    def setUp(self):
        self.cfg = self.default_test_config()
        self.tus = (TranslationUnit("foo", llvm_foo_double_double),)

    def test_invalid_library_name_type(self):
        expected = (".*kwarg library_name should be a string, got.*object")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name=object())

    def test_invalid_library_name_value(self):
        expected = ("kwarg library_name cannot be an empty string")
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="")

    def test_invalid_translation_units_type(self):
        expected = ("kwarg translation_units should be a tuple or list, got "
                    ".*str")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units="invalid")

    def test_invalid_translation_units_none_given(self):
        expected = ("kwarg translation_units is an empty tuple, no translation "
                    "units were given, there is nothing to compile!")
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",)

    def test_invalid_export_configuration_type(self):
        expected = ("kwarg export_configuration must be of type None or a "
                    "pixie.ExportConfiguration instance, got .*str")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          export_configuration="invalid")

    def test_no_baseline_cpu(self):
        with self.assertRaisesRegex(ValueError, "The baseline_cpu kwarg must*"):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,)

    def test_invalid_baseline_cpu(self):
        expected = "Target.*has no CPU named 'not a cpu'.*baseline_cpu"
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu="not a cpu")

    def test_invalid_baseline_features(self):
        expected = ("Feature 'invalid feature' is not a known feature for "
                    "target .*baseline_features")
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=("invalid feature"))

    def test_invalid_targets_features(self):
        expected = ("Feature 'invalid feature' is not a known feature for "
                    "target .*targets_features")
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          targets_features=("invalid feature"))

    def test_invalid_uuid_type(self):
        expected = ("kwarg uuid must be a string representation of a uuid4 or "
                    "None")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          uuid=('invalid type',))

    def test_invalid_uuid_value(self):
        expected = ("badly formed.*")
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          uuid="this is not a uuid",)

    def test_invalid_opt_value(self):
        expected = "kwarg opt must be an integer value in.*got 4"
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          opt=4,)

    def test_invalid_opt_flags_type(self):
        expected = "kwarg opt_flags must be a dictionary"
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          opt=2,
                          opt_flags="invalid")

    def test_invalid_opt_flags_invalid_key_value(self):
        expected = "kwarg opt_flags contains an invalid key: invalid"
        with self.assertRaisesRegex(ValueError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          opt=2,
                          opt_flags={"invalid": False})

    def test_invalid_opt_flags_invalid_value_type(self):
        expected = ("kwarg opt_flags key 'slp_vectorize' has an invalid "
                    r"value type 'str' \(expected bool\).")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          opt=2,
                          opt_flags={"slp_vectorize": "invalid"})

    def test_invalid_output_dir_value(self):
        expected = ("kwarg output_dir should be a string, bytes or "
                    "os.PathLike, got list")
        with self.assertRaisesRegex(TypeError, expected):
            PIXIECompiler(library_name="lib",
                          translation_units=self.tus,
                          baseline_cpu=self.cfg.baseline_target.cpu,
                          baseline_features=self.cfg.baseline_target.features,
                          output_dir=[])


class TestTranslationUnitInterface(PixieTestCase):

    def test_bad_name(self):
        expected = "name must be a string, got list"
        with self.assertRaisesRegex(TypeError, expected):
            TranslationUnit([], llvm_foo_double_double)

    def test_bad_source(self):
        expected = "Expected string or bytes for source, got list."
        with self.assertRaisesRegex(TypeError, expected):
            TranslationUnit("foo", [])


if __name__ == '__main__':
    unittest.main()
