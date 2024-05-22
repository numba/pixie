from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase
import unittest
import os
import subprocess
import tempfile
import sysconfig


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

        def c_from_cython_source(fname):
            cmd = ('cython', '-3', fname)
            subprocess.run(cmd)
            outfile = os.path.join(cls.tmpdir.name,
                                   ''.join([fname.split(".")[0], '.c']))
            with open(outfile, 'rt') as f:
                data = f.read()
            return data, outfile

        # TODO: This is shared with the C tests, put it in support.py
        def tu_from_c_source(fname):
            prefix = 'pixie-c-build-'
            with tempfile.TemporaryDirectory(prefix=prefix) as build_dir:
                outfile = os.path.join(build_dir, 'tmp.bc')

                cmd = ('clang', '-x', 'c++',
                       '-I', sysconfig.get_path("include"),
                       '-fPIC', '-mcmodel=small',
                       '-emit-llvm', fname, '-o', outfile, '-c')
                # TODO: need to check exit status
                subprocess.run(cmd)
                with open(outfile, 'rb') as f:
                    data = f.read()
            return TranslationUnit(fname, data)

        cython_c_src, cython_c_file = c_from_cython_source(cython_file)

        tus = []
        tus.append(tu_from_c_source(cython_c_file))

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
                               output_dir='.')

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
