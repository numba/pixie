import sys
from distutils import sysconfig
from distutils.command import build
from distutils.spawn import spawn

from setuptools import find_packages, setup
import versioneer

_version_module = None
try:
    from packaging import version as _version_module
except ImportError:
    try:
        from setuptools._vendor.packaging import version as _version_module
    except ImportError:
        pass


min_python_version = "3.8"
max_python_version = "3.12"  # exclusive
min_llvmlite_version = "0.39"
max_llvmlite_version = "0.42"

if sys.platform.startswith('linux'):
    # Make wheels without libpython
    sysconfig.get_config_vars()['Py_ENABLE_SHARED'] = 0


def _guard_py_ver():
    if _version_module is None:
        return

    parse = _version_module.parse

    min_py = parse(min_python_version)
    max_py = parse(max_python_version)
    cur_py = parse('.'.join(map(str, sys.version_info[:3])))

    if not min_py <= cur_py < max_py:
        msg = ('Cannot install on Python version {}; only versions >={},<{} '
               'are supported.')
        raise RuntimeError(msg.format(cur_py, min_py, max_py))


_guard_py_ver()


class build_doc(build.build):
    description = "build documentation"

    def run(self):
        spawn(['make', '-C', 'docs', 'html'])


versioneer.VCS = 'git'
versioneer.versionfile_source = 'pixie/_version.py'
versioneer.versionfile_build = 'pixie/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'pixie-'

cmdclass = versioneer.get_cmdclass()
cmdclass['build_doc'] = build_doc

packages = find_packages(include=["pixie", "pixie.*"])

install_requires = [
    'llvmlite >={},<{}'.format(min_llvmlite_version, max_llvmlite_version),
    'setuptools',
]

metadata = dict(
    name='pixie',
    description=("Create C extensions containing portable information to"
                 "permit recompilation/inclusion."),
    version=versioneer.get_version(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Compilers",
    ],
    url="",
    packages=packages,
    setup_requires=[],
    install_requires=install_requires,
    python_requires=">={}".format(min_python_version),
    license="BSD",
    cmdclass=cmdclass,
)

with open('README.rst') as f:
    metadata['long_description'] = f.read()

setup(**metadata)
