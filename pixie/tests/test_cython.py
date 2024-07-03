from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase
import unittest
import os


class TestCython(PixieTestCase):
    """Tests that Cython works as an input source for PIXIE.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        cython_src = """def addone_function(int x):\n\treturn x + 1"""
        cython_file = os.path.join(cls.tmpdir.name, "addone.pyx")
        with open(cython_file, "wt") as f:
            f.write(cython_src)

        tus = []
        tus.append(TranslationUnit.from_cython_source(cython_file))

        export_config = ExportConfiguration()

        target_descr = cls.default_test_config()
        bfeat = target_descr.baseline_target.features
        libfoo = PIXIECompiler(library_name='addone',
                               translation_units=tus,
                               export_configuration=export_config,
                               baseline_cpu=target_descr.baseline_target.cpu,
                               baseline_features=bfeat,
                               targets_features=target_descr.additional_targets,
                               python_cext=True,
                               output_dir=cls.tmpdir.name)

        libfoo.compile()

    def test_cython(self):
        with self.load_pixie_module('addone') as addone_library:
            # check that the module has loaded and that the __PIXIE__ overlay
            # has been added.
            assert "__PIXIE__" in dir(addone_library)
            x = 123
            assert addone_library.addone_function(x) == x + 1


if __name__ == '__main__':
    unittest.main()
