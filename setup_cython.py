"""
Setup per compilare il modulo Cython nw_core.pyx.

Uso:
    python setup_cython.py build_ext --inplace

Questo crea il file nw_core.cpython-*.so che può essere importato
direttamente da Python come un normale modulo.
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="nw_core",
    ext_modules=cythonize(
        "nw_core.pyx",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
