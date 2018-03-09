from distutils.core import setup, Extension
import os, glob, numpy



__version__ = '0.0.1'

def indir(dir, files): return [dir+f for f in files]
def globdir(dir, files):
    rv = []
    for f in files: rv += glob.glob(dir+f)
    return rv

setup(name = 'wignerpy',
    version = __version__,
    description = __doc__,
    long_description = __doc__,
    license = 'GPL',
    author = 'Eric Yang, Jeff Zheng, Jianshu Li',
    author_email = '',
    url = '',
    package_dir = {'HERA_MapMaking_VisibilitySimulation':'src'},
    packages = ['HERA_MapMaking_VisibilitySimulation'],
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
            #globdir('src/_omnical/',
            #    ['*.cpp', '*.c', '*.cc']),
            include_dirs=['src/_omnical/include', numpy.get_include()],
            extra_compile_args=['-Wno-write-strings', '-O3']
        )
    ],
    scripts = glob.glob('scripts/*'),
)
