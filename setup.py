from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob

import numpy

sourcefiles = ["pyinverseflow.pyx"]
# sourcefiles.extend(glob("inverse_flow/*.cpp"))
extensions = [
    Extension("pyinverseflow", sourcefiles, include_dirs=[numpy.get_include()])
]
setup(
    name="pyinverseflow",
    version="0.1",
    description="""Python wrapper for inverse optical flow code.
See https://ctim.ulpgc.es/research_works/computing_inverse_optical_flow/.""",
    author="Benjamin Wilhelm",
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
