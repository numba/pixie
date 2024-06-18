"""Python C-API bindings"""
from llvmlite import ir
from pixie import llvm_types as lt


class RawPyAPI(object):

    @classmethod
    def PyImport_ImportModule(self, llvm_module):
        def PyImport_ImportModule_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *
                                        (lt._char_star,))

            name = "PyImport_ImportModule"
            return signature, name
        PyImport_ImportModule_fn = ir.Function(
            llvm_module, *PyImport_ImportModule_sig_type())
        PyImport_ImportModule_fn.linkage = 'external'
        return PyImport_ImportModule_fn

    @classmethod
    def PyObject_GetAttrString(self, llvm_module):
        def PyObject_GetAttrString_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, char *
                                        (lt._pyobject_head_p,
                                         lt._char_star,))
            name = "PyObject_GetAttrString"
            return signature, name

        PyObject_GetAttrString_fn = ir.Function(
            llvm_module, *PyObject_GetAttrString_sig_type())
        PyObject_GetAttrString_fn.linkage = 'external'
        return PyObject_GetAttrString_fn

    @classmethod
    def PyBytes_FromStringAndSize(self, llvm_module):
        def PyBytes_FromStringAndSize_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # char *, size_t
                                        (lt._char_star, lt._llvm_py_ssize_t))
            name = "PyBytes_FromStringAndSize"
            return signature, name
        PyBytes_FromStringAndSize_fn = ir.Function(
            llvm_module, *PyBytes_FromStringAndSize_sig_type())
        PyBytes_FromStringAndSize_fn.linkage = 'external'
        return PyBytes_FromStringAndSize_fn

    @classmethod
    def PyObject_CallFunctionObjArgs(self, llvm_module):
        def PyObject_CallFunctionObjArgs_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # pyobj*, ...
                                        (lt._pyobject_head_p,), var_arg=True)
            name = "PyObject_CallFunctionObjArgs"
            return signature, name

        PyObject_CallFunctionObjArgs_fn = ir.Function(
            llvm_module, *PyObject_CallFunctionObjArgs_sig_type())
        PyObject_CallFunctionObjArgs_fn.linkage = 'external'
        return PyObject_CallFunctionObjArgs_fn

    @classmethod
    def Py_BuildValue(self, llvm_module):
        def Py_BuildValue_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # const char *, ...
                                        (lt._char_star,),  var_arg=True)
            name = "Py_BuildValue"
            return signature, name

        Py_BuildValue_fn = ir.Function(llvm_module, *Py_BuildValue_sig_type())
        Py_BuildValue_fn.linkage = 'external'
        return Py_BuildValue_fn

    @classmethod
    def PyDict_SetItemString(self, llvm_module):
        def PyDict_SetItemString_sig_type():
            signature = ir.FunctionType(lt._int32,
                                        # (PyObject *p, const char *key,
                                        # PyObject *val)
                                        (lt._pyobject_head_p,
                                         lt._char_star,
                                         lt._pyobject_head_p), )
            name = "PyDict_SetItemString"
            return signature, name

        PyDict_SetItemString_fn = ir.Function(llvm_module,
                                              *PyDict_SetItemString_sig_type())
        PyDict_SetItemString_fn.linkage = 'external'
        return PyDict_SetItemString_fn

    @classmethod
    def PyDict_GetItemString(self, llvm_module):
        def PyDict_GetItemString_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # (PyObject *p, const char *key,)
                                        (lt._pyobject_head_p,
                                         lt._char_star,), )
            name = "PyDict_GetItemString"
            return signature, name

        PyDict_GetItemString_fn = ir.Function(llvm_module,
                                              *PyDict_GetItemString_sig_type())
        PyDict_GetItemString_fn.linkage = 'external'
        return PyDict_GetItemString_fn

    @classmethod
    def PyRun_String(self, llvm_module):
        def PyRun_String_sig_type():
            signature = ir.FunctionType(lt._pyobject_head_p,
                                        # const char *, int, pyobj *, pyobj*
                                        (lt._char_star,
                                         lt._int32,
                                         lt._pyobject_head_p,
                                         lt._pyobject_head_p), )
            name = "PyRun_String"
            return signature, name

        PyRun_String_fn = ir.Function(llvm_module, *PyRun_String_sig_type())
        PyRun_String_fn.linkage = 'external'
        return PyRun_String_fn

    @classmethod
    def PyUnicode_AsUTF8AndSize(self, llvm_module):
        def PyUnicode_AsUTF8AndSize_sig_type():
            signature = ir.FunctionType(lt._char_star,
                                        # pyobj *, py_ssize_t*
                                        (lt._pyobject_head_p,
                                         lt._llvm_py_ssize_t_star,),)
            name = "PyUnicode_AsUTF8AndSize"
            return signature, name
        args = PyUnicode_AsUTF8AndSize_sig_type()
        PyUnicode_AsUTF8AndSize_fn = ir.Function(llvm_module, *args)
        PyUnicode_AsUTF8AndSize_fn.linkage = 'external'
        return PyUnicode_AsUTF8AndSize_fn

    @classmethod
    def PyType_IsSubtype(self, llvm_module):
        def PyType_IsSubtype_sig_type():
            signature = ir.FunctionType(lt._int64,
                                        # PyTypeObject *, PyTypeObject *
                                        (lt._pytypeobject_head_p,
                                         lt._pytypeobject_head_p,),)
            name = "PyType_IsSubtype"
            return signature, name
        args = PyType_IsSubtype_sig_type()
        PyType_IsSubtype_fn = ir.Function(llvm_module, *args)
        PyType_IsSubtype_fn.linkage = 'external'
        return PyType_IsSubtype_fn
