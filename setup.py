# setup.py
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='hrv_utils',
    packages=['hrv_utils'],
    description='Fast implementation of DMA and general HRV methods.',
    author='Joao Kruse',
    version='1.0',
    ext_modules=cythonize(
        "hrv_utils/dma.pyx",
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
    include_dirs=[numpy.get_include()],
)
