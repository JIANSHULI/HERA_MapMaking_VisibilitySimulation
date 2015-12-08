__author__ = 'omniscope'

import healpy as hp
import healpy.sphtfunc as hps
import numpy as np
import matplotlib.pyplot as plt

nside = 256
npix = hp.nside2npix(nside)
m = np.zeros(npix).astype('float')
za, az = hp.pix2ang(nside, np.arange(npix))

m[abs(az) < 20 * np.pi / 180.] = 1.
m[abs(za) < 20 * np.pi / 180.] = 1.
msmooth = hps.smoothing(m, fwhm=hp.nside2resol(16), nest)

hp.mollview(msmooth)
plt.show()