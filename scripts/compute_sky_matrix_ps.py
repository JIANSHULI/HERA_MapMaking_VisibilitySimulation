#400 minutes      for 80 steps at nside=16, 75 ubl


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

p = 'x'
infofile = '/home/omniscope/omnical/doc/redundantinfo_X5_q3%s.bin'%(p)#'/home/omniscope/omnical/doc/redundantinfo_PSA128_26ba_6bu_08-15-2014.bin'#
pol = p+p
info = omni.read_redundantinfo(infofile)
nside = 32
nside_target = 8

inclusion_thresh = 1 #betweennnn 0 and 1. ubl/lambda must be this times nside_target less
all_ubl = False
bnside = 8

freq = 160.

vs = sv.Visibility_Simulator()
if 'X5' in infofile and 'PSA' in infofile:
    raise Exception('Cant tell if the iinfo is for PAPER or MITEoR')
elif 'X5' in infofile:
    lat_degree = 45.2977
    infoubl = info['ubl'].dot([[1.5,0,0],[0,1.5,0],[0,0,0]])
    beam_healpix = np.fromfile('/home/omniscope/simulate_visibilities/data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s.bin'%(bnside, freq, pol), dtype='float32')
    point_sources = [[np.pi*((23+(23.+26./60.)/60.)/12.), np.pi*(58.+48./60.)/180.], [np.pi*((19.+(59.+28.3566/60.)/60.)/12.), np.pi*(40.+(44.+2.096/60.)/60.)/180.]]
    nps = len(point_sources)
elif 'PSA' in infofile:
    raise Exception('PSA point source not yet programmed.')
    lat_degree = -(30.+ 43./60. + 17.5/3600.)
    infoubl = info['ubl'].dot([[4,0,0],[0,15,0],[0,0,0]])
    calfile = 'psa6240_v003'
    ######get array  information and accurate ubl vectors
    aa = ap.cal.get_aa(calfile, np.array([freq/1000.]))

    ######load beam
    nsideB = 8
    Bl = nsideB*3 - 1

    beam_healpix = np.zeros(12*nsideB**2, dtype='float32')
    if pol[0] != pol[-1]:
        raise Exception('ERROR: polarization string %s not supported in the code.'%pol)
    for i in range(12*nsideB**2):
        beam_healpix[i] = aa[0].bm_response(sv.rotatez_matrix(-np.pi/2).dot(hpf.pix2vec(nsideB, i)), pol[0])[0]#in paper's bm_response convention, (x,y) = (0,1) points north.
else:
    raise Exception('Cant tell if the iinfo is for PAPER or MITEoR')

vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))
timer = time.time()

if all_ubl:
    ubls = infoubl
else:
    ubls = [ubl for ubl in infoubl if la.norm(ubl) < inclusion_thresh * (nside_target * 299.792458 / freq)]
print "%i UBLs to include"%len(ubls)

if len(tlist)*len(ubls) < 12*nside**2:
    for i in range(5):
        print 'Not enough degree of freedom! %i*%i<%i.'%(len(tlist), len(ubls), 12*nside**2)


A = np.empty((len(tlist)*len(ubls), nps), dtype='complex64')
for i in range(nps):
    ra, dec = point_sources[i]
    theta = np.pi/2 - dec
    phi = ra
    theta_heal, phi_heal = hpf.pix2ang(nside, hpf.ang2pix(nside, theta, phi))

    A[:, i] = np.array([vs.calculate_pointsource_visibility(phi_heal, np.pi/2.-theta_heal, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()
print float(time.time()-timer)/60.
if all_ubl:
    A.tofile('/home/omniscope/data/GSM_data/AmatrixPS_allubl_nside%i_%iby%i_'%(nside, len(A), len(A[0])) + infofile.split('/')[-1])
else:
    A.tofile('/home/omniscope/data/GSM_data/AmatrixPS_%iubl_nside%i_%iby%i_'%(len(ubls), nside, len(A), len(A[0])) + infofile.split('/')[-1])

