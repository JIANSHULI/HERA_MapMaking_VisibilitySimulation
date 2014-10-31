import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si

def pixelize(sky, nside_distribution, nside_standard, nside_start, thresh, final_index, thetas, phis):
    #thetas = []
    #phis = []
    for inest in range(12*nside_start**2):
        pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
        #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
        #thetas += newt.tolist()
        #phis += newp.tolist()
    #return np.array(thetas), np.array(phis)

def pixelize_helper(sky, nside_distribution, nside_standard, nside, inest, thresh, final_index, thetas, phis):
    #print "visiting ", nside, inest
    starti, endi = inest*nside_standard**2/nside**2, (inest+1)*nside_standard**2/nside**2
    ##local mean###if nside == nside_standard or np.std(sky[starti:endi])/np.mean(sky[starti:endi]) < thresh:
    if nside == nside_standard or np.std(sky[starti:endi]) < thresh:
        nside_distribution[starti:endi] = nside
        final_index[starti:endi] = len(thetas)#range(len(thetas), len(thetas) + endi -starti)
        #return hp.pix2ang(nside, [inest], nest=True)
        newt, newp = hp.pix2ang(nside, [inest], nest=True)
        thetas += newt.tolist()
        phis += newp.tolist()

    else:
        #thetas = []
        #phis = []
        for jnest in range(inest * 4, (inest + 1) * 4):
            pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh, final_index, thetas, phis)
            #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh)
            #thetas += newt.tolist()
            #phis += newp.tolist()
        #return np.array(thetas), np.array(phis)

nside_start = 4
nside_beamweight = 16
nside_standard = 256
bnside = 8
plotcoord = 'C'
thresh = 0.10
lat_degree = 45.2977
C = 299.792458
kB = 1.3806488* 1.e-23

force_recompute = False
####################################################
################data file and load beam##############
####################################################
tag = "q3_abscalibrated"
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = 440
nf = 1
nUBL = 75

#deal with beam: create a dictionary for 'x' and 'y' each with a callable function of the form y(freq) in MHz
local_beam = {}
for p in ['x', 'y']:
    freqs = range(150,170,10)
    beam_array = np.zeros((len(freqs), 12*bnside**2))
    for f in range(len(freqs)):
        beam_array[f] = np.fromfile('/home/omniscope/simulate_visibilities/data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s%s.bin'%(bnside, freqs[f], p, p), dtype='float32')
    local_beam[p] = si.interp1d(freqs, beam_array, axis=0)

A = {}
for p in ['x', 'y']:
    pol = p+p
    #tf file
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])

    #tf mask file, 0 means flagged bad data
    try:
        tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
        tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
        tmask = np.array(tfmlist[:,0].astype('bool'))
        #print tmask
    except:
        print "No mask file found"
        tmask = np.zeros_like(tlist).astype(bool)
    #print freq, tlist

    #ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include"%len(ubls)

    A_filename = datadir + tag + '_%s%s_%i_%i.A'%(p, p, len(tlist)*len(ubls), 12*nside_beamweight**2)

    if os.path.isfile(A_filename) and not force_recompute:
        print "Reading A matrix from %s"%A_filename
        sys.stdout.flush()
        A[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(ubls), len(tlist), 12*nside_beamweight**2))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), 12*nside_beamweight**2))
    else:
        #beam
        beam_healpix = local_beam[p](freq)
        #hpv.mollview(beam_healpix, title='beam %s'%p)
        #plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
        beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))
        print "Computing A matrix for %s pol..."%p
        sys.stdout.flush()
        timer = time.time()
        A[p] = np.empty((len(tlist)*len(ubls), 12*nside_beamweight**2), dtype='complex64')
        for i in range(12*nside_beamweight**2):
            dec, ra = hpf.pix2ang(nside_beamweight, i)#gives theta phi
            dec = np.pi/2 - dec
            print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(12.*nside_beamweight**2), (12.*nside_beamweight**2-i)/(i+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()

        print "%f minutes used"%(float(time.time()-timer)/60.)
        sys.stdout.flush()
        A[p].tofile(A_filename)
        A[p] = A[p].reshape((len(ubls), len(tlist), 12*nside_beamweight**2))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), 12*nside_beamweight**2))

####################################################
###beam weights using an equal pixel A matrix######
#################################################
print "Computing beam weight...",
sys.stdout.flush()
beam_weight = np.array([(la.norm(A['x'][:,col])**2 + la.norm(A['y'][:,col])**2)**.5 for col in range(len(A['x'][0]))])[hpf.nest2ring(nside_beamweight, range(12*nside_beamweight**2))]
beam_weight = beam_weight/np.mean(beam_weight)
beam_weight = np.array([beam_weight for i in range(nside_standard**2/nside_beamweight**2)]).transpose().flatten()
print "done."
sys.stdout.flush()

################################################
#####################GSM###########################
#############################################
pca1 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm3.fits' + str(nside_standard))
gsm_standard = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)#hardcode
equatorial_GSM_standard = np.zeros(12*nside_standard**2,'float')
#rotate sky map
print "Rotating GSM_standard...",
sys.stdout.flush()

equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000,stdtime=2013.8))
ang0, ang1 =hp.rotator.rotateDirection(equ2013_to_gal_matrix, hpf.pix2ang(nside_standard, range(12*nside_standard**2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
print "done."
sys.stdout.flush()



########################################################################
########################processing#################################
########################################################################

nside_distribution = np.zeros(12*nside_standard**2)
final_index = np.zeros(12*nside_standard**2)
thetas, phis = [], []
abs_thresh = np.mean(equatorial_GSM_standard * beam_weight) * thresh
pixelize(equatorial_GSM_standard * beam_weight, nside_distribution, nside_standard, nside_start, abs_thresh, final_index, thetas, phis)
npix = len(thetas)
fake_solution = equatorial_GSM_standard[hpf.ang2pix(nside_standard, thetas, phis, nest=True)]
##################################################################
####################################sanity check########################
###############################################################
#npix = 0
#for i in nside_distribution:
    #npix += i**2/nside_standard**2
#print npix, len(thetas)

stds = np.std((equatorial_GSM_standard*beam_weight).reshape(12*nside_standard**2/4,4), axis = 1)

##################################################################
####################################plotting########################
###############################################################
hpv.mollview(beam_weight, min=0,max=4, coord=plotcoord, title='beam', nest=True)
hpv.mollview(np.log10(equatorial_GSM_standard), min=0,max=4, coord=plotcoord, title='GSM', nest=True)
hpv.mollview(np.log10(fake_solution[np.array(final_index).tolist()]), min=0,max=4, coord=plotcoord, title='GSM gridded', nest=True)
hpv.mollview(np.log10(stds/abs_thresh), min=np.log10(thresh)-3, max = 3, coord=plotcoord, title='std', nest=True)
hpv.mollview(np.log2(nside_distribution), min=np.log2(nside_start),max=np.log2(nside_standard), coord=plotcoord, title='count %i %.3f'%(len(thetas), float(len(thetas))/(12*nside_standard**2)), nest=True)
plt.show()


##################################################################
####################compute A matrix########################
###############################################################

A = {}
for p in ['x', 'y']:
    pol = p+p
    #tf file
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])

    #tf mask file, 0 means flagged bad data
    try:
        tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
        tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
        tmask = np.array(tfmlist[:,0].astype('bool'))
        #print tmask
    except:
        print "No mask file found"
        tmask = np.zeros_like(tlist).astype(bool)
    #print freq, tlist

    #ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include"%len(ubls)

    A_filename = datadir + tag + '_%s%s_%i_%i.Ad'%(p, p, len(tlist)*len(ubls), npix)

    if os.path.isfile(A_filename) and not force_recompute:
        print "Reading A matrix from %s"%A_filename
        sys.stdout.flush()
        A[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(ubls), len(tlist), npix))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), npix))
    else:
        #beam
        beam_healpix = local_beam[p](freq)
        #hpv.mollview(beam_healpix, title='beam %s'%p)
        #plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
        beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))
        print "Computing A matrix for %s pol..."%p
        sys.stdout.flush()
        timer = time.time()
        A[p] = np.empty((len(tlist)*len(ubls), npix), dtype='complex64')
        for i in range(npix):
            ra = phis[i]
            dec = np.pi/2 - thetas[i]
            print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(npix), (npix-i)/(i+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()

        print "%f minutes used"%(float(time.time()-timer)/60.)
        sys.stdout.flush()
        A[p].tofile(A_filename)
        A[p] = A[p].reshape((len(ubls), len(tlist), npix))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), npix))

