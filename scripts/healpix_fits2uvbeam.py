import numpy as np
from pyuvdata import UVBeam
import astropy.io.fits as fits
import healpy as hp

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


# make uvbeam object
uvb = UVBeam()
uvb.set_simple()
uvb.antenna_type = 'simple'
uvb.telescope_name = 'hera'
uvb.feed_name = 'dipole'
uvb.feed_version = '0.0'
uvb.model_name = 'beam_sims'
uvb.model_version = '0.1'
uvb.history = ''
uvb.pixel_coordinate_system='healpix'
uvb.nside = 64
uvb.ordering = 'ring'
uvb.pixel_array = np.arange(64**2 * 12)
uvb.Npixels = len(uvb.pixel_array)
uvb.freq_array = freqs * 1e6
uvb.spw_array = np.array([0])
uvb.Nspws = 1
uvb.data_normalization = 'peak'
uvb.Nfreqs = len(freqs)
uvb.freq_array = freqs.reshape(1, Nfreqs) * 1e6
uvb.bandpass_array = np.ones((1, uvb.Nfreqs))
uvb.beam_type = 'power'
uvb.set_power()
uvb.polarization_array = np.array([-5, -6])
uvb.Npols = 2
uvb.Naxes_vec = 1
beams.resize
uvb.data_array = beam_data

uvb.write_beamfits("NF_HERA_Beams.beamfits", clobber=True)

