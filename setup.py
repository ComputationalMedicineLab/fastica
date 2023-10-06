import os
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy as np

Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True

# This is unfortunately indirect - but seems also most reliable.
include_dirs = [np.get_include()]
include_dirs.extend(np.__config__.blas_info['include_dirs'])
library_dirs = np.__config__.blas_info['library_dirs']
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
extra_link_args = ['-fopenmp']

# TODO: comment on what these do and why they're used
extra_compile_args = [
    '-fopenmp',
    '-O3',
    '-Wno-unused',
    '-march=native',
    '-mfpmath=sse',
    '-mavx',
#    '-mdaz-ftz', #Why does GCC not seem to recognize this?
]

# Use the ILP64 interface libraries directly to avoid conflicts with Numpy /
# Scipy et. al. - when they load MKL for blas/lapack they load `mkl_rt`, the
# single dynamic library (SDL). The SDL can be configured to use either 32-bit
# or 64-bit MKL_INTs, but if we set it to 64-bit then
#   (1) The setting is only respected if made before the first MKL call, so our
#       code must always be imported _prior_ to numpy, scipy, etc or we'll be
#       locked into 32-bit MKL_INT (NB: it needs to be imported first
#       _anyways_, no matter which lib we link against, or we lose 64-bit
#       capability. But linking against mkl_intel_ilp64 at least leaves scipy
#       alone.)
#   (2) Even if we successfully set to 64-bit, scipy.linalg is hard-coded to
#       expect 32-bit: eigh and other functions will flat-out break entirely.
# Therefore, we *must* always link against `mkl_intel_ilp64` (and hence also
# `mkl_intel_thread`, `mkl_core`, `iomp5`, and `pthread`, _in that order_),
# rather than against `mkl_rt`. The macro and other args are also needed.
libraries = ['mkl_intel_ilp64', 'mkl_intel_thread', 'mkl_core',
             'iomp5', 'pthread', 'm', 'dl']
define_macros.append(("MKL_ILP64", 1))
extra_link_args.append('-Wl,--no-as-needed')
extra_compile_args.append('-m64')

# Reference for Extension arguments:
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference
module = Extension("fastica", ['fastica.pyx'],
                   language='c++',
                   include_dirs=include_dirs,
                   define_macros=define_macros,
                   libraries=libraries,
                   library_dirs=library_dirs,
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

# Cythonize argument reference:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
ext_modules = cythonize([module],
                        nthreads=32,
                        annotate=True,
                        force=False,
                        compiler_directives={
                            'binding': True,
                            'boundscheck': False,
                            'cdivision': True,
                            'embedsignature': True,
                            'emit_code_comments': True,
                            'initializedcheck': False,
                            'language_level': 3,
                            'wraparound': False,
                        # Enable these for debugging
                            #'profile': True,
                            #'linetrace': True,
                        })

setup(name="fastica",
      version='0.1',
      ext_modules=ext_modules,
      package_data={'fastica': ['mkl_wrapper.pxd']},
      entry_points={
          'console_scripts': ['fastica=fastica:cli'],
      },
      include_package_data=True,
      zip_safe=False)
