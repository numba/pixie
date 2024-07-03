import os
import unittest
import uuid
import tempfile
import numpy as np
import ctypes
import yaml

from pixie import PIXIECompiler, TranslationUnit, ExportConfiguration
from pixie.tests.support import PixieTestCase

# Yaml objects for the LLVM remarks output.
# Based on https://github.com/llvm/llvm-project/blob/617a15a9eac96088ae5e9134248d8236e34b91b1/llvm/tools/opt-viewer/optrecord.py#L61-L286  # noqa: E501


class Remark(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader


class Analysis(Remark):
    yaml_tag = "!Analysis"


class Passed(Remark):
    yaml_tag = "!Passed"


class Missed(Remark):
    yaml_tag = "!Missed"


class TestOptFlags(PixieTestCase):
    @classmethod
    def setUpClass(cls):
        PixieTestCase.setUpClass()

    def setUp(self):
        remarks_file = os.path.join(self.tmpdir.name, str(uuid.uuid4().hex))
        self._bkup_remarks_file = os.environ.get("PIXIE_LLVM_REMARKS_FILE")
        os.environ["PIXIE_LLVM_REMARKS_FILE"] = remarks_file
        self.remarks_file = remarks_file

        self._bkup_remarks_filter = os.environ.get("PIXIE_LLVM_REMARKS_FILTER")
        os.environ["PIXIE_LLVM_REMARKS_FILTER"] = ".*vectorize.*"

    def tearDown(self):
        if self._bkup_remarks_file is not None:
            os.environ["PIXIE_LLVM_REMARKS_FILE"] = self._bkup_remarks_file
        else:
            os.environ.pop("PIXIE_LLVM_REMARKS_FILE", None)

        if self._bkup_remarks_filter is not None:
            os.environ["PIXIE_LLVM_REMARKS_FILTER"] = self._bkup_remarks_filter
        else:
            os.environ.pop("PIXIE_LLVM_REMARKS_FILTER", None)

    def test_loop_vectorize(self):
        vect_src = """
        #include<math.h>

        void f(double* x, double* y, int *n){
            for (int i = 0; i < *n; i++) {
                x[i] = 2. * y[i] - 3. * (x[i] - y[i]) + 4. * y[i];
            }
        }
        """
        tus = []
        with tempfile.NamedTemporaryFile('wt') as ntf:
            ntf.write(vect_src)
            ntf.flush()
            tus.append(TranslationUnit.from_c_source(ntf.name))

        sig_str = 'void(double*, double*, i64*)'
        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='f',
                                 symbol_name='f',
                                 signature=sig_str,)

        target_descr = self.default_test_config()

        bfeat = target_descr.baseline_target.features

        compiler = PIXIECompiler(library_name='vect_loop',
                                 translation_units=tus,
                                 export_configuration=export_config,
                                 baseline_cpu=target_descr.baseline_target.cpu,
                                 baseline_features=bfeat,
                                 python_cext=True,
                                 opt=3,
                                 opt_flags={"loop_vectorize": True},
                                 output_dir=self.tmpdir.name)
        compiler.compile()

        with self.load_pixie_module('vect_loop') as vect_library:
            symbols = vect_library.__PIXIE__['symbols']
            n = 4
            x = 3 * np.ones(n)
            y = 2 * np.ones(n)
            f = symbols['f'][sig_str]['cfunc']
            arg_n = ctypes.c_long(n)
            f(ctypes.cast(x.ctypes, ctypes.POINTER(ctypes.c_double)),
              ctypes.cast(y.ctypes, ctypes.POINTER(ctypes.c_double)),
              ctypes.byref(arg_n))
            np.testing.assert_allclose(x, 9. * np.ones(n))

            with open(self.remarks_file, 'rt') as f:
                raw_remarks = f.read()

            remarks = [x for x in yaml.load_all(raw_remarks, yaml.SafeLoader)]
            for remark in remarks:
                if remark.Function == "f":
                    if remark.Pass == "loop-vectorize":
                        break
                else:
                    raise ValueError("Expected a loop-vectorized passed remark")

    def test_slp_vectorize(self):

        # Sample translated from:
        # https://www.llvm.org/docs/Vectorizers.html#the-slp-vectorizer

        vect_src = r'''
        #include<math.h>
        #include<stdio.h>
        #include<stdint.h>

        void f(int64_t *a1, int64_t *a2, int64_t *b1, int64_t *b2, int64_t *A) {
            A[0] = *a1 * (*a1 + *b1);
            A[1] = *a2 * (*a2 + *b2);
            A[2] = *a1 * (*a1 + *b1);
            A[3] = *a2 * (*a2 + *b2);
            // SLP only seems to run if there's some more work
            const int n = 10;
            char buffer[n];
            snprintf(buffer, n, "%ld", A[0]);
        }
        '''
        tus = []
        with tempfile.NamedTemporaryFile('wt') as ntf:
            ntf.write(vect_src)
            ntf.flush()
            tus.append(TranslationUnit.from_c_source(ntf.name))

        sig_str = 'void(i64*, i64*, i64*, i64*, i64*)'
        export_config = ExportConfiguration()
        export_config.add_symbol(python_name='f',
                                 symbol_name='f',
                                 signature=sig_str,)

        target_descr = self.default_test_config()

        bfeat = target_descr.baseline_target.features

        compiler = PIXIECompiler(library_name='vect_slp',
                                 translation_units=tus,
                                 export_configuration=export_config,
                                 baseline_cpu=target_descr.baseline_target.cpu,
                                 baseline_features=bfeat,
                                 python_cext=True,
                                 opt=3,
                                 opt_flags={"slp_vectorize": True},
                                 output_dir=self.tmpdir.name)
        compiler.compile()

        with self.load_pixie_module('vect_slp') as vect_library:
            symbols = vect_library.__PIXIE__['symbols']
            f = symbols['f'][sig_str]['cfunc']
            A = np.zeros(4, np.int64)
            a1 = ctypes.c_long(2)
            a2 = ctypes.c_long(3)
            b1 = ctypes.c_long(5)
            b2 = ctypes.c_long(7)
            f(ctypes.byref(a1), ctypes.byref(a2), ctypes.byref(b1),
              ctypes.byref(b2),
              ctypes.cast(A.ctypes, ctypes.POINTER(ctypes.c_long)))

            np.testing.assert_allclose(A, np.array([14, 30, 14, 30]))

            with open(self.remarks_file, 'rt') as f:
                raw_remarks = f.read()

            remarks = [x for x in yaml.load_all(raw_remarks, yaml.SafeLoader)]
            for remark in remarks:
                if remark.Function == "f":
                    if remark.Pass == "slp-vectorizer":
                        break
                else:
                    raise ValueError("Expected a slp-vectorizer remark")


if __name__ == '__main__':
    unittest.main()
