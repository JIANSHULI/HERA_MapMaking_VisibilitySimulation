# from __future__ import absolute_import
# from distutils.core import setup, Extension
import os, glob, numpy
import os.path as op
from src import version
import json
# from setuptools import Extension, Command
# from setuptools import setup
from distutils.core import setup, Extension

########## pyuvdata version ##########
data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(op.join('src', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

########## hera_dp_vs_mp version ###########
__version__ = '0.0.2'

def indir(dir, files): return [dir+f for f in files]
def globdir(dir, files):
    rv = []
    for f in files: rv += glob.glob(dir+f)
    return rv

setup(name = 'HERA_MapMaking_VisibilitySimulation',
    # version = __version__,
    version = version.version,
    description = __doc__,
    long_description = __doc__,
    license = 'GPL',
    author = 'Eric Yang, Jeff Zheng, Jianshu Li',
    author_email = '',
    url = 'https://github.com/JIANSHULI/HERA_MapMaking_VisibilitySimulation.git',
    package_dir = {'HERA_MapMaking_VisibilitySimulation':'src'},
    packages = ['HERA_MapMaking_VisibilitySimulation'],
    include_package_data = True,
    ext_modules = [
        Extension('HERA_MapMaking_VisibilitySimulation._boost_math',
            globdir('src/_boost_math/',
                ['*.cpp', '*.c', '*.cc']),
            include_dirs = ['src/_boost_math/include', '/usr/include', numpy.get_include()],
            extra_compile_args=['-Wno-write-strings', '-O3']
        ),
        Extension('HERA_MapMaking_VisibilitySimulation._Bulm',
            globdir('src/_Bulm/',
                ['*.cpp', '*.c', '*.cc']),
            include_dirs = ['src/_Bulm/include', '/usr/include', numpy.get_include()],
            extra_compile_args=['-Wno-write-strings', '-O3']
        ),
        Extension('HERA_MapMaking_VisibilitySimulation._omnical',
            ['src/_omnical/omnical_wrap.cpp', 'src/_omnical/omnical_redcal.cc'],
            # globdir('src/_omnical/',
            #    ['*.cpp', '*.c', '*.cc']),
            include_dirs=['src/_omnical/include', '/usr/include', numpy.get_include()],
            extra_compile_args=['-Wno-write-strings', '-O3']
        )
    ],
    scripts = glob.glob('scripts/*'),
)

