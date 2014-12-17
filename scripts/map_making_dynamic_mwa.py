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

def pixelize(sky, nside_distribution, nside_standard, nside_start, thresh, final_index, thetas, phis, sizes):
    #thetas = []
    #phis = []
    for inest in range(12*nside_start**2):
        pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis, sizes)
        #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
        #thetas += newt.tolist()
        #phis += newp.tolist()
    #return np.array(thetas), np.array(phis)

def pixelize_helper(sky, nside_distribution, nside_standard, nside, inest, thresh, final_index, thetas, phis, sizes):
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
        #sizes += (np.ones_like(newt) * nside_standard**2 / nside**2).tolist()
        sizes += (np.ones_like(newt) / nside**2).tolist()

    else:
        #thetas = []
        #phis = []
        for jnest in range(inest * 4, (inest + 1) * 4):
            pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh, final_index, thetas, phis, sizes)
            #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh)
            #thetas += newt.tolist()
            #phis += newp.tolist()
        #return np.array(thetas), np.array(phis)

nside_start = 8
nside_beamweight = 16
nside_standard = 256
bnside = 128
plotcoord = 'C'
thresh = 0.10
S_scale = 2
S_thresh = 1000#Kelvin
S_type = 'gsm%irm%i'%(S_scale,S_thresh)

lat_degree = -26.703319
C = 299.792458
kB = 1.3806488* 1.e-23
script_dir = os.path.dirname(os.path.realpath(__file__))
plot_pixelization = True
force_recompute = False
force_recompute_AtNiAi = False
force_recompute_S = False
force_recompute_SEi = False

####################################################
################data file and load beam##############
####################################################
tag = "mwa_test"
datatag = '_seccasa.rad'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/'
nt = 60
nf = 1
nUBL = 1222
pols = ['x']


A = {}
data = {}
Ni = {}
for p in pols:
    pol = p+p

    #tf file, t in lst hours
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]

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
        #deal with beam: create a dictionary for 'x' and 'y' each with a callable function of the form y(freq) in MHz
        local_beam = {}
        for p in pols:
            freqs = range(150,170,10)

        beam_healpix = np.fromfile(datadir + 'mwa_curtin_beam_%s_nside%i_freq167.275_zenith_float32.dat'%(p, bnside), dtype='float32')
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
beam_weight = ((la.norm(A['x'], axis = 0)**2 + la.norm(A['y'], axis = 0)**2)**.5)[hpf.nest2ring(nside_beamweight, range(12*nside_beamweight**2))]
beam_weight = beam_weight/np.mean(beam_weight)
beam_weight = np.array([beam_weight for i in range(nside_standard**2/nside_beamweight**2)]).transpose().flatten()
print "done."
sys.stdout.flush()

################################################
#####################GSM###########################
#############################################
pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
components = np.loadtxt(script_dir + '/../data/components.dat')
scale_loglog = si.interp1d(np.log(components[:,0]), np.log(components[:,1]))
w1 = si.interp1d(components[:,0], components[:,2])
w2 = si.interp1d(components[:,0], components[:,3])
w3 = si.interp1d(components[:,0], components[:,4])
gsm_standard = np.exp(scale_loglog(np.log(freq))) * (w1(freq)*pca1 + w2(freq)*pca2 + w3(freq)*pca3)

#rotate sky map and converts to nest
equatorial_GSM_standard = np.zeros(12*nside_standard**2,'float')
print "Rotating GSM_standard and converts to nest...",
sys.stdout.flush()
equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000,stdtime=2013.8))
ang0, ang1 =hp.rotator.rotateDirection(equ2013_to_gal_matrix, hpf.pix2ang(nside_standard, range(12*nside_standard**2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
print "done."
sys.stdout.flush()



########################################################################
########################processing dynamic pixelization######################
########################################################################

nside_distribution = np.zeros(12*nside_standard**2)
final_index = np.zeros(12*nside_standard**2)
thetas, phis, sizes = [], [], []
abs_thresh = np.mean(equatorial_GSM_standard * beam_weight) * thresh
pixelize(equatorial_GSM_standard * beam_weight, nside_distribution, nside_standard, nside_start, abs_thresh, final_index, thetas, phis, sizes)
npix = len(thetas)
fake_solution = hpf.get_interp_val(equatorial_GSM_standard, thetas, phis, nest=True)

if plot_pixelization:
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
####################compute dynamic A matrix########################
###############################################################

A = {}
for p in pols:
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

    A_filename = datadir + tag + '_%s%s_%i_%i.Ad%i_%.3f'%(p, p, len(tlist)*len(ubls), npix, nside_standard, thresh)

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
            print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(npix), float(npix-i)/(i+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()

        print "%f minutes used"%(float(time.time()-timer)/60.)
        sys.stdout.flush()
        A[p].tofile(A_filename)
        A[p] = A[p].reshape((len(ubls), len(tlist), npix))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), npix))
    #get Ni (1/variance) and data
    var_filename = datadir + tag + '_%s%s_%i_%i.var'%(p, p, nt, nUBL)
    Ni[p] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose().flatten() * (1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside_standard**2)))**2)
    data_filename = datadir + tag + '_%s%s_%i_%i'%(p, p, nt, nUBL) + datatag
    data[p] = (np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose().flatten()*1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside_standard**2))).conjugate()#there's a conjugate convention difference
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()
data = np.concatenate((data['x'],data['y']))
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
Ni = np.concatenate((Ni['x'],Ni['y']))
Ni = np.concatenate((Ni/2, Ni/2))

print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
A = np.concatenate((A['x'], A['y']))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
A = np.concatenate((np.real(A), np.imag(A)))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

#simulate visibilities

sim_data = A.dot(fake_solution * sizes * nside_standard**2) + np.random.randn(len(data))/Ni**.5
#plt.plot(sim_data[:5000], 'g--')
#plt.plot(data[:5000], 'b--')
#plt.show()
#quit()

#compute AtNi and AtNi.y
AtNi = A.transpose() * Ni
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

AtNi_data = AtNi.dot(data)
AtNi_sim_data = AtNi.dot(sim_data)


#compute AtNiA eigensystems
eigvl_filename = datadir + tag + '_%i.5AtNiAel'%(npix)
eigvc_filename = datadir + tag + '_%i_%i.5AtNiAev'%(npix, npix)
if os.path.isfile(eigvl_filename) and os.path.isfile(eigvc_filename):
    print "Reading eigen system of AtNiA from %s and %s"%(eigvl_filename, eigvc_filename)
    del(AtNi)
    #del(A)
    eigvl = np.fromfile(eigvl_filename, dtype='float32')
    eigvc = np.fromfile(eigvc_filename, dtype='float32').reshape((npix, npix))
else:
    print "Computing AtNiA eigensystem...",
    sys.stdout.flush()
    timer = time.time()
    eigvl, eigvc = sla.eigh(AtNi.dot(A))
    print "%f minutes used"%(float(time.time()-timer)/60.)
    sys.stdout.flush()
    del(AtNi)
    #del(A)
    if la.norm(eigvl) == 0:
        print "ERROR: Eigensistem calculation failed...matrix %i by %i is probably too large."%(npix, npix)
    print "%f minutes used"%(float(time.time()-timer)/60.)
    eigvl.tofile(eigvl_filename)
    eigvc.tofile(eigvc_filename)

plt.plot(eigvl)

#compute AtNiAi
rcondA = 1.e-5
AtNiAi_filename = datadir + tag + '_%i_%i.5AtNiAi%i'%(npix, npix, np.log10(rcondA))
if os.path.isfile(AtNiAi_filename) and not force_recompute_AtNiAi:
    print "Reading AtNiAi matrix from %s"%AtNiAi_filename
    AtNiAi = np.fromfile(AtNiAi_filename, dtype='float32').reshape((npix, npix))

else:
    print "Computing AtNiAi matrix...",
    sys.stdout.flush()
    timer = time.time()
    max_eigv = max(eigvl)
    print "Min eig %.2e"%min(eigvl), "Max eig %.2e"%max_eigv, "Add eig %.2e"%(max_eigv * rcondA)
    if min(eigvl) < 0 and np.abs(min(eigvl)) > max_eigv * rcondA:
        print "!WARNING!: negative eigenvalue %.2e is smaller than the added identity %.2e! min rcond %.2e needed."%(min(eigvl), max_eigv * rcondA, np.abs(min(eigvl))/max_eigv)
    AtNiAi = (eigvc / (max_eigv * rcondA + eigvl)).dot(eigvc.transpose())
    print "%f minutes used"%(float(time.time()-timer)/60.)
    sys.stdout.flush()
    AtNiAi.tofile(AtNiAi_filename)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#solve for x
x = AtNiAi.dot(AtNi_data)
sim_x = AtNiAi.dot(AtNi_sim_data)

chisq = np.sum(Ni * np.abs((A.dot(x) - data))**2) / float(len(data) - npix)
chisq_sim = np.sum(Ni * np.abs((A.dot(sim_x) - sim_data))**2) / float(len(sim_data) - npix)

hpv.mollview(np.log10(fake_solution[np.array(final_index).tolist()]), min= 0, max =4, coord=plotcoord, title='GSM gridded', nest=True)
hpv.mollview(np.log10((x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw solution, chi^2=%.2f'%chisq, nest=True)
hpv.mollview(np.log10((sim_x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw simulated solution, chi^2=%.2f'%chisq_sim, nest=True)
plt.show()
