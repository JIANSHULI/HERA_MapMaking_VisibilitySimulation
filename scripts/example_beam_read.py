from astropy.io import fits
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from pyuvdata import UVBeam

filename = 'healpix_beam.fits'
beam_E = fits.getdata(filename, extname='BEAM_E')
nside = hp.npix2nside(beam_E.shape[0])
freqs = fits.getdata(filename, extname='FREQS')

print('Frequencies: ')
print(freqs)

#ind = np.where(freqs == 150)[0][0]
#hp.mollview(10.0 * np.log10(beam_E[:,ind]), title='HERA Beam-East, 150 MHz',
#            unit='dBi')
#plt.show()


d = fits.open('healpix_beam.fits')
beams = d[0].data
freqs = d[1].data

# select only 100-200 MHz data
freq_select = np.where((freqs >= 100) & (freqs <=200))[0]
beams = beams[:, freq_select]
freqs = freqs[freq_select]
Nfreqs = len(freqs)

# take East pol and rotate to get North pol
beam_theta, beam_phi = hp.pix2ang(64, np.arange(64**2 * 12))
R = hp.Rotator(rot=[0,0,-np.pi/2], deg=False)
beam_theta2, beam_phi2 = R(beam_theta, beam_phi)
beam_rot = np.array(map(lambda x: hp.get_interp_val(x, beam_theta2, beam_phi2), beams.T))
beam_data = np.array([beams.T, beam_rot])
beam_data.resize(1, 1, 2, Nfreqs, 49152)

# normalize each frequency to max of 1
for i in range(beam_data.shape[-2]):
      beam_data[:, :, :, i, :] /= beam_data[:, :, :, i, :].max()


ind = np.where(freqs == 150)[0][0]
hp.mollview(10.0 * np.log10(beam_E[:,ind]), title='HERA Beam-East, 150 MHz',
            unit='dBi')
plt.show(block=False)
hp.mollview(10.0 * np.log10(beam_data[0,0,1,ind]), title='HERA Beam-North, 150 MHz',
            unit='dBi')
plt.show(block=False)