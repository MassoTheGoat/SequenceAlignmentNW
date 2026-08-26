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
