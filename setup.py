from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# OSX only
# from os import environ
# environ["CC"] = "g++-6"
# environ["CXX"] = "g++-6"

extensions = [
    Extension("ndsar",
              sources=["ndsar.pyx"],
              language="c++",
              include_dirs=[np.get_include(), "./include"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-lgomp']
              ),
    ]

setup(
        name = "NDSAR filters for multi-dimensional SAR images.",
    ext_modules = cythonize(extensions)
    )
