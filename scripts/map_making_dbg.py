#400 minutes      for 80 steps at nside=16, 75 ubl


import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem, sys, os
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import omnical.calibration_omni as omni


tag = "q3_abscalibrated"
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = 440
nf = 1
nUBL = 75
nside = 16
bnside = 8
lat_degree = 45.2977
force_recompute = False

C = 299.792458
kB = 1.3806488* 1.e-23

#deal with beam: create a dictionary for 'x' and 'y' each with a callable function of the form y(freq) in MHz
local_beam = {}
for p in ['x', 'y']:
    freqs = range(150,170,10)
    beam_array = np.zeros((len(freqs), 12*bnside**2))
    for f in range(len(freqs)):
        beam_array[f] = np.fromfile('/home/omniscope/simulate_visibilities/data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s%s.bin'%(bnside, freqs[f], p, p), dtype='float32')
    local_beam[p] = si.interp1d(freqs, beam_array, axis=0)

A = {}
data = {}
Ni = {}
for p in ['x']:
    pol = p+p

    #tf file
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]
    #print freq, tlist

    #ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include"%len(ubls)

    #beam
    beam_healpix = local_beam[p](freq)
    #hpv.mollview(beam_healpix, title='beam %s'%p)
    #plt.show()

    vs = sv.Visibility_Simulator()
    vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
    beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))




    #get Ni (1/variance) and data
    data_filename = datadir + tag + '_%s%s_%i_%i.dat'%(p, p, nt, nUBL)
    data[p] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL)).conjugate()

data = data['x']
#data = np.concatenate((np.real(data), np.imag(data))).astype('float32')

u=46
sim_data = {}
#load A matrix
for nside in [16,32,64]:
    A_filename = datadir + tag + '_%s%s_%i_%i.A'%('x', 'x', len(tlist)*len(ubls), 12*nside**2)
    print "Reading A matrix from %s"%A_filename
    sys.stdout.flush()
    A = np.fromfile(A_filename, dtype='complex64')[u*len(tlist)*12*nside**2:(u+1)*len(tlist)*12*nside**2].reshape((len(tlist), 12*nside**2))
    print "done."
    sys.stdout.flush()

    #simulate
    nside_standard = nside
    pca1 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm1.fits' + str(nside_standard))
    pca2 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm2.fits' + str(nside_standard))
    pca3 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm3.fits' + str(nside_standard))
    gsm_standard = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
    equatorial_GSM_standard = np.zeros(12*nside_standard**2,'float')
    #rotate sky map
    print "Rotating GSM_standard...",
    sys.stdout.flush()
    for i in range(12*nside_standard**2):
        ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside_standard,i))
        equatorial_GSM_standard[i] = hpf.get_interp_val(gsm_standard, ang[0], ang[1])
    print "done."
    sys.stdout.flush()
    sim_data[nside_standard] = A.dot(equatorial_GSM_standard)/(1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside**2)))


plt.plot(np.imag(data[:,u]),'b')
plt.plot(np.imag(sim_data[16]),'r')
plt.plot(np.imag(sim_data[32]),'g')
plt.plot(np.imag(sim_data[64]),'c')
plt.show()
