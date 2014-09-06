import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem, sys
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import scipy.interpolate as si
import omnical.calibration_omni as omni

tlist = np.arange(16, 24, .1)
infofile = '/home/omniscope/omnical/doc/redundantinfo_X5_q3x.bin'
info = omni.read_redundantinfo(infofile)


bnside = 8
pol = 'xx'
freq = 160.

vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0,45.2977*np.pi/180])#self.zenithequ

beam_healpix = np.fromfile('/home/omniscope/simulate_visibilities/data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s.bin'%(bnside, freq, pol), dtype='float32')
#vs.import_beam(beam_healpix)

timer = time.time()

nside = 8
ubls = [ubl for ubl in 3.*info['ubl'] if la.norm(ubl) < (nside * 299.792458 / freq)]
print "%i UBLs to include"%len(ubls)

if len(tlist)*len(ubls) < 12*nside**2:
    raise Exception('Not enough degree of freedom! %i*%i<%i.'%(len(tlist), len(ubls), 12*nside**2))
A = np.empty((len(tlist)*len(ubls), 12*nside**2), dtype='complex64')
for i in range(12*nside**2):
    dec, ra = hpf.pix2ang(nside, i)#gives theta phi
    dec = np.pi/2 - dec
    print "%.1f%%"%(100.*float(i)/(12.*nside**2)),
    sys.stdout.flush()
    A[:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_healpix_hor = beam_healpix, tlist = tlist) for d in ubls]).flatten()
print float(time.time()-timer)/60.
A.tofile('/home/omniscope/simulate_visibilities/data/Amatrix_' + '_nside%i_%iby%i_'%(nside, len(A), len(A[0])) + infofile.split('/')[-1])

