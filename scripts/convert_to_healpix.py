#!/usr/bin/env python

# Read beam output from CST and convert to HEALPix.

import numpy as np
import healpy as hp
from astropy.io import fits
from scipy import interpolate
import optparse
import sys
import re

o = optparse.OptionParser()
o.add_option('-n', '--nside', dest='nside', default=64,
             help='nside parameter of output healpix maps')
o.add_option('-o', '--outfile', dest='outfile',
             default='healpix_beam.fits', help='Output filename')
opts, args = o.parse_args(sys.argv[1:])
nside = opts.nside
outfile = opts.outfile
filenames = args

freqs = np.array([float(re.findall(r'\d+', f.split()[-2])[0]) for f in filenames])
order = np.argsort(freqs)
freqs = freqs[order]
filenames = np.array(filenames)[order]

hmap = np.zeros((hp.nside2npix(nside), len(freqs)))
thetai, phii = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

for fi, f in enumerate(filenames):
    data = np.loadtxt(f, skiprows=2)
    lat = np.unique(data[:, 0]) * np.pi / 180.0
    nlat = len(lat)
    lon = np.unique(data[:, 1]) * np.pi / 180.0
    nlon = len(lon)
    gain = data[:, 2].reshape(nlon, nlat).transpose()
    lut = interpolate.RectBivariateSpline(lat, lon, gain)
    for i in np.arange(hp.nside2npix(nside)):
        hmap[i, fi] = lut(thetai[i], phii[i])

new_hdul = fits.HDUList()
new_hdul.append(fits.ImageHDU(data=hmap, name='BEAM_E'))
new_hdul.append(fits.ImageHDU(data=freqs, name='FREQS'))
new_hdul.writeto(outfile)

