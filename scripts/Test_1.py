import numpy as np
import healpy as hp
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import astropy as ap
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import sys
import os

# print('sys.argv[0]')
# print('sys.argv: {0}'.format(sys.argv))

for str in ['lijianshu', 'victor']:
    os.system('ipython Test_2.py {0}'.format(str))

    