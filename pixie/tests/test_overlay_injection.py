from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase, needs_clang
import tempfile
import unittest


@needs_clang
class TestOverlayInjection(PixieTestCase):
    """Tests injection of the __PIXIE__ dict overlay in the case of a single-
    phase and a multi-phase module initialisation process.
    """

    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

        src1 = """
        #include <Python.h>

        static PyMethodDef single_phase_methods[] = {{NULL, NULL, 0, NULL}};

        static struct PyModuleDef single_phase_moddef = {
            PyModuleDef_HEAD_INIT,
            "single_phase",
            NULL,
            -1,
            single_phase_methods
        };

        PyMODINIT_FUNC PyInit_single_phase(void)
        {
            PyObject * mod = PyModule_Create(&single_phase_moddef);
            if (mod == NULL) {
                return NULL;
            }
            return mod;
        }
        """

        src2 = """
        #include <Python.h>

        static PyMethodDef multi_phase_methods[] = {{NULL, NULL, 0, NULL}};

        static struct PyModuleDef_Slot multi_phase_slots[] = {
            {0, NULL},
        };

        static struct PyModuleDef multi_phase_moddef = {
            PyModuleDef_HEAD_INIT,
            "multi_phase",
            NULL,
            0,
            multi_phase_methods,
            multi_phase_slots,
            NULL,
            NULL,
            NULL,
        };

        PyMODINIT_FUNC PyInit_multi_phase(void)
        {
            PyObject * mod = PyModuleDef_Init(&multi_phase_moddef);
            if (mod == NULL) {
                return NULL;
            }
            return mod;
        }

        """

        def compile_local(src, library_name):
            tus = []
            with tempfile.NamedTemporaryFile('wt') as ntf:
                ntf.write(src)
                ntf.flush()
                tus.append(TranslationUnit.from_c_source(ntf.name))

            export_config = ExportConfiguration()

            target_descr = cls.default_test_config()
            bcpu = target_descr.baseline_target.cpu
            bfeat = target_descr.baseline_target.features

            compiler = PIXIECompiler(library_name=library_name,
                                     translation_units=tus,
                                     export_configuration=export_config,
                                     baseline_cpu=bcpu,
                                     baseline_features=bfeat,
                                     python_cext=True,
                                     output_dir=cls.tmpdir.name)
            compiler.compile()

        compile_local(src1, "single_phase")
        compile_local(src2, "multi_phase")

    def test_call_single_phase(self):
        with self.load_pixie_module('single_phase') as foo_library:
            self.assertIn("__PIXIE__", dir(foo_library))

    def test_call_multi_phase(self):
        with self.load_pixie_module('multi_phase') as foo_library:
            self.assertIn("__PIXIE__", dir(foo_library))


if __name__ == '__main__':
    unittest.main()
