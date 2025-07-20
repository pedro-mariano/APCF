from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('APCF_sims', ['APCF_sims.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))
