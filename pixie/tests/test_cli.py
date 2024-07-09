import os
import subprocess
import textwrap
from tempfile import TemporaryDirectory

from pixie.tests.support import PixieTestCase

example_config = """
{
    "translation_unit": [
    {
      "name": "llvm_foo_double_double",
      "source": "int _Z3fooPdS_(double* a, double* b, double* c) {
                    *c = *a + *b;
                }"
    },
    {
      "name": "llvm_foo_float_float",
      "path": "llvm_foo_float_float.c",
      "extra_flags": []
    },
  ],
  "export_config": {
      "symbols": [
        {
          "python_name": "foo",
          "symbol_name": "_Z3fooPdS_",
          "signature": "void(double*, double*, double*)"
        }
      ]
    }
}"""


class TestCLI(PixieTestCase):

    def test_pixie_cc_basic(self):
        cfile_name = "test.c"
        cfile_source = textwrap.dedent(
            """
            int f(x) {
                return x + 1;
            }
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            cfile_path = os.path.join(tmpdir, cfile_name)
            with open(cfile_path, "wt") as f:
                f.write(cfile_source)
            testlib_name = "testclib"
            command = ["pixie-cc", cfile_name, "-g", "-O0", "-o", testlib_name]
            subprocess.run(command, check=True, cwd=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(2, len(files))
            self.assertEqual(files[0], cfile_name)
            self.assertTrue(files[1].startswith(testlib_name))

    def test_pixie_cc_config(self):
        cfile_name = "test.c"
        cfile_source = textwrap.dedent(
            """
            int f(x) {
                return x + 1;
            }
            """)
        json_file_name = "config.json"
        json_file_source = example_config
        tu_cfile_name = "llvm_foo_float_float.c"
        tu_cfile_source = textwrap.dedent(
            """
            int foo(float* a, float* b, float* c) { *c = *a + *b; }
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            cfile_path = os.path.join(tmpdir, cfile_name)
            with open(cfile_path, "wt") as f:
                f.write(cfile_source)
            json_file_path = os.path.join(tmpdir, json_file_name)
            with open(json_file_path, "wt") as f:
                f.write(json_file_source)
            with open(os.path.join(tmpdir, tu_cfile_name), "wt") as f:
                f.write(tu_cfile_source)
            testlib_name = "testclib"
            command = ["pixie-cc", cfile_name, "-g", "-O0", "-o",
                       testlib_name, "-C", json_file_name]
            subprocess.run(command, check=True, cwd=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(4, len(files))
            self.assertEqual(files[0], json_file_name)
            self.assertEqual(files[1], tu_cfile_name)
            self.assertEqual(files[2], cfile_name)
            self.assertTrue(files[3].startswith(testlib_name))

    def test_pixie_cc_two_files(self):
        cfile1_name = "test1.c"
        cfile1_source = textwrap.dedent(
            """
            int f(x) {
                return x + 1;
            }
            """)

        cfile2_name = "test2.c"
        cfile2_source = textwrap.dedent(
            """
            int g(x) {
                return x * 2;
            }
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            for name, src in ((cfile1_name, cfile1_source),
                              (cfile2_name, cfile2_source)):
                cfile_path = os.path.join(tmpdir, name)
                with open(cfile_path, "wt") as f:
                    f.write(src)
            testlib_name = "testc2lib"
            command = ["pixie-cc", cfile1_name, cfile2_name, "-o", testlib_name]
            subprocess.run(command, check=True, cwd=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(3, len(files))
            self.assertEqual(files[0], cfile1_name)
            self.assertEqual(files[1], cfile2_name)
            self.assertTrue(files[2].startswith(testlib_name))

    def test_pixie_cc_two_files_no_output_name(self):
        cfile1_name = "test1.c"
        cfile1_source = textwrap.dedent(
            """
            int f(x) {
                return x + 1;
            }
            """)

        cfile2_name = "test2.c"
        cfile2_source = textwrap.dedent(
            """
            int g(x) {
                return x * 2;
            }
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            for name, src in ((cfile1_name, cfile1_source),
                              (cfile2_name, cfile2_source)):
                cfile_path = os.path.join(tmpdir, name)
                with open(cfile_path, "wt") as f:
                    f.write(src)
            command = ["pixie-cc", cfile1_name, cfile2_name]
            with self.assertRaises(subprocess.CalledProcessError) as e:
                subprocess.run(command, check=True, cwd=tmpdir,
                               capture_output=True)

                captured_stderr = e.exception.stderr.decode()
                expected = ("pixie-cc: error: Option -o (output library) is "
                            "missing and cannot be inferred as there are "
                            "multiple input files.")
                self.assertIn(expected, captured_stderr)

    def test_pixie_cythonize(self):
        cyfile_name = "test.pyx"
        cyfile_source = textwrap.dedent(
            """
            def addone_function(int x):
                return x + 1
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            cyfile_path = os.path.join(tmpdir, cyfile_name)
            with open(cyfile_path, "wt") as f:
                f.write(cyfile_source)
            testlib_name = "testcylib"
            command = ["pixie-cythonize", cyfile_name, "-g", "-O2",
                       "-o", testlib_name]
            subprocess.run(command, check=True, cwd=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(2, len(files))
            self.assertEqual(files[0], cyfile_name)
            self.assertTrue(files[1].startswith(testlib_name))

    def test_pixie_cythonize_two_files(self):
        cyfile1_name = "test1.pyx"
        cyfile1_source = textwrap.dedent(
            """
            def addone_function(int x):
                return x + 1
            """)
        cyfile2_name = "test2.pyx"
        cyfile2_source = textwrap.dedent(
            """
            def timestwo_function(int x):
                return x * 2
            """)
        with TemporaryDirectory(prefix=self.tmpdir.name) as tmpdir:
            for name, src in ((cyfile1_name, cyfile1_source),
                              (cyfile2_name, cyfile2_source)):
                cfile_path = os.path.join(tmpdir, name)
                with open(cfile_path, "wt") as f:
                    f.write(src)
            testlib_name = "testcy2lib"
            command = ["pixie-cythonize", cyfile1_name, cyfile2_name, "-o",
                       testlib_name]
            subprocess.run(command, check=True, cwd=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(3, len(files))
            self.assertEqual(files[0], cyfile1_name)
            self.assertEqual(files[1], cyfile2_name)
            self.assertTrue(files[2].startswith(testlib_name))
