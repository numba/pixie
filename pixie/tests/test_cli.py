import os
import subprocess
import textwrap

from pixie.cli import pixie_cythonize
from pixie.tests.support import PixieTestCase

class TestCLI(PixieTestCase):

    def test_pixie_cc(self):
        cfile_name = "test.c"
        cfile_source = textwrap.dedent("""
            #include<math.h>

            void f(double* x, double* result){
                *result = cos(*x) + 1;
            }
        """)
        pixie_cc_test_dir = os.path.join(self.tmpdir.name, "pixie_cc_test")
        os.mkdir(pixie_cc_test_dir)
        cfile_path = os.path.join(pixie_cc_test_dir, cfile_name)
        with open(cfile_path, "wt") as f:
            f.write(cfile_source)
        testlib_name = "testctlib"
        command = ["pixie-cc", cfile_name, "-o", testlib_name]
        subprocess.run(command, check=True, cwd=pixie_cc_test_dir)
        files = sorted(os.listdir(pixie_cc_test_dir))
        self.assertEqual(2, len(files))
        self.assertEqual(files[0], cfile_name)
        self.assertTrue(files[1].startswith(testlib_name))

    def test_pixie_cythonize(self):
        cyfile_name = "test.pyx"
        cfile_name = "test.c"
        cyfile_source = textwrap.dedent("""
            def addone_function(int x):
                return x + 1
        """)
        pixie_cythonize_test_dir = os.path.join(self.tmpdir.name, "pixie_cythonize_test")
        os.mkdir(pixie_cythonize_test_dir)
        cyfile_path = os.path.join(pixie_cythonize_test_dir, cyfile_name)
        with open(cyfile_path, "wt") as f:
            f.write(cyfile_source)
        testlib_name = "testcylib"
        command = ["pixie-cythonize", cyfile_name, "-o", testlib_name]
        subprocess.run(command, check=True, cwd=pixie_cythonize_test_dir)
        files = sorted(os.listdir(pixie_cythonize_test_dir))
        self.assertEqual(3, len(files))
        self.assertEqual(files[0], cfile_name)
        self.assertEqual(files[1], cyfile_name)
        self.assertTrue(files[2].startswith(testlib_name))
