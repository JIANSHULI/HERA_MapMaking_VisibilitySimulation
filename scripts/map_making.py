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
for p in ['x', 'y']:
    #tf file
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]
    pol = p+p

    #ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include"%len(ubls)

    #beam
    beam_healpix = local_beam[p](freq)


    vs = sv.Visibility_Simulator()
    vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
    beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))



    #compute A matrix
    A_filename = datadir + tag + '_%s%s_%i_%i.A'%(p, p, len(tlist)*len(ubls), 12*nside**2)

    if os.path.isfile(A_filename) and not force_recompute:
        print "Reading A matrix from %s"%A_filename
        A[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(tlist)*len(ubls), 12*nside**2))
    else:
        print "Computing A matrix for %s pol..."%p
        timer = time.time()
        A[p] = np.empty((len(tlist)*len(ubls), 12*nside**2), dtype='complex64')
        for i in range(12*nside**2):
            dec, ra = hpf.pix2ang(nside, i)#gives theta phi
            dec = np.pi/2 - dec
            print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(12.*nside**2), (12.*nside**2-i)/(i+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()

        print "%f minutes used"%(float(time.time()-timer)/60.)
        A[p].tofile(A_filename)

    #get Ni (1/variance) and data
    var_filename = datadir + tag + '_%s%s_%i_%i.var'%(p, p, nt, nUBL)
    Ni[p] = 1./np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL)).transpose().flatten()
    data_filename = datadir + tag + '_%s%s_%i_%i.dat'%(p, p, nt, nUBL)
    data[p] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL)).transpose().flatten()

data = np.concatenate((data['x'],data['y']))
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
#plt.plot(Ni['x'][::nt])
#plt.show()
Ni = np.concatenate((Ni['x'],Ni['y']))
Ni = np.concatenate((Ni/2, Ni/2))
A = np.concatenate((A['x'],A['y']))
A = np.concatenate((np.real(A), np.imag(A))).astype('float32')

#compute AtNi
AtNi = A.transpose().conjugate() * Ni

#compute AtNiAi
rcondA = 1.e-6
AtNiAi_filename = datadir + tag + '_%i_%i.AtNiAi%i'%(12*nside**2, 12*nside**2, np.log10(rcondA))
if os.path.isfile(AtNiAi_filename) and not force_recompute:
    print "Reading AtNiAi matrix from %s"%AtNiAi_filename
    AtNiAi = np.fromfile(AtNiAi_filename, dtype='float32').reshape((12*nside**2, 12*nside**2))
else:
    print "Computing AtNiAi matrix..."
    timer = time.time()
    AtNiAi = la.pinv(AtNi.dot(A), rcond=rcondA)
    print "%f minutes used"%(float(time.time()-timer)/60.)
    AtNiAi.tofile(AtNiAi_filename)

#compute raw x
x = AtNiAi.dot(AtNi.dot(data))
hpv.mollview(x, min=0,max=5000,title='raw solution')
plt.show()
