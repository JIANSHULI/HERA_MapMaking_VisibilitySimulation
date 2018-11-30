from distutils.core import setup
from Cython.Build import cythonize
from HERA_MapMaking_VisibilitySimulation import DATA_PATH

setup(
    ext_modules=cythonize(DATA_PATH + "/../scripts/HERA-VisibilitySimulation-MapMaking.pyx"),
)