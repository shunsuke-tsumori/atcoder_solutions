from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("b.pyx", compiler_directives={'boundscheck': False, 'wraparound': False})
)