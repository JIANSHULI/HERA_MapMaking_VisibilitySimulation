from astropy.io import fits
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

filename = 'healpix_beam.fits'
beam_E = fits.getdata(filename, extname='BEAM_E')
nside = hp.npix2nside(beam_E.shape[0])
freqs = fits.getdata(filename, extname='FREQS')

print('Frequencies: ')
print(freqs)

ind = np.where(freqs == 150)[0][0]
hp.mollview(10.0 * np.log10(beam_E[:,ind]), title='HERA Beam-East, 150 MHz',
            unit='dBi')
plt.show()
