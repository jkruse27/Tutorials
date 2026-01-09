# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


def parse_requirements(filename):
    with open(filename) as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
            ]


extensions = [
    Extension(
        name="hrv_utils.dma",
        sources=["hrv_utils/dma.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="hrv_utils.rri_utils",
        sources=["hrv_utils/rri_utils.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
    Extension(
        name="hrv_utils.nongaussian",
        sources=["hrv_utils/nongaussian.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    name='hrv_utils',
    packages=['hrv_utils'],
    description='Fast implementation of DMA and general HRV methods.',
    author='Joao Kruse',
    version='1.0',

    install_requires=parse_requirements("requirements.txt"),

    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,
    ),
)
