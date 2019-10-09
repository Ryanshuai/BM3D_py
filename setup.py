from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("helloworld.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("precompute_bm.pyx"),
    include_dirs=[numpy.get_include()]
)
