# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Helper to read requirements.txt so you don't have to list them twice
def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='hrv_utils',
    packages=['hrv_utils'],
    description='Fast implementation of DMA and general HRV methods.',
    author='Joao Kruse',
    version='1.0',
    
    # 1. Runtime dependencies go here (replaces the subprocess call)
    install_requires=parse_requirements("requirements.txt"),
    
    # 2. Cython configuration
    ext_modules=cythonize(
        "hrv_utils/dma.pyx",
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
    
    # 3. Include numpy headers
    include_dirs=[numpy.get_include()],
)
