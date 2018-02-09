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
import omnical.calibration_omni as omni

def pinv_sym(M, rcond = 1.e-15, verbose = True):
    eigvl,eigvc = la.eigh(M)
    max_eigv = max(eigvl)
    if verbose:
        print "Min eig %.2e"%min(eigvl), "Max eig %.2e"%max_eigv, "Add eig %.2e"%(max_eigv * rcond)
        if min(eigvl) < 0 and np.abs(min(eigvl)) > max_eigv * rcond:
            print "!WARNING!: negative eigenvalue %.2e is smaller than the added identity %.2e! min rcond %.2e needed."%(min(eigvl), max_eigv * rcond, np.abs(min(eigvl))/max_eigv)
    eigvli = 1 / (max_eigv * rcond + eigvl)
    #for i in range(len(eigvli)):
        #if eigvl[i] < max_eigv * rcond:
            #eigvli[i] = 0
        #else:
            #eigvli[i] = 1/eigvl[i]
    return (eigvc*eigvli).dot(eigvc.transpose())

tag = "q3_abscalibrated"
datatag = '_seccasa.rad'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = 440
nf = 1
nUBL = 75
nside = 64
S_scale = 2
S_thresh = 1000#Kelvin
S_type = 'gsm%irm%i'%(S_scale,S_thresh)
plotcoord = 'C'
bnside = 8
lat_degree = 45.2977
force_recompute = False
force_recompute_AtNiAi = False
force_recompute_S = False
force_recompute_SEi = False

C = 299.792458
kB = 1.3806488* 1.e-23

#read in data vectors
data = {}
Ni = {}
for p in ['x', 'y']:
    pol = p+p
    #tf file
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]
    print freq

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

    #get Ni (1/variance) and data
    var_filename = datadir + tag + '_%s%s_%i_%i.var'%(p, p, nt, nUBL)
    Ni[p] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose().flatten() * (1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside**2)))**2)
    data_filename = datadir + tag + '_%s%s_%i_%i'%(p, p, nt, nUBL) + datatag
    data[p] = (np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose().flatten()*1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside**2))).conjugate()#there's a conjugate convention difference

data = np.concatenate((data['x'],data['y']))
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')

Ni = np.concatenate((Ni['x'],Ni['y']))
Ni = np.concatenate((Ni/2, Ni/2))

pixm_filename = datadir + tag + '_%i.pixm'%(12*nside**2)
pix_mask = np.fromfile(pixm_filename, dtype = 'bool')
npix = np.sum(pix_mask)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

#load A matrix
A_filename = datadir + tag + '_%i_%i.Axyri'%(len(data), npix)
print "Reading A matrix from %s"%A_filename
sys.stdout.flush()
A = np.fromfile(A_filename, dtype='float32').reshape((len(data), npix))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)

#simulate data using GSM
nside_standard = nside
pca1 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm3.fits' + str(nside_standard))
gsm_standard = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
equatorial_GSM_standard = np.zeros(12*nside_standard**2,'float')
#rotate sky map
print "Rotating GSM_standard...",
sys.stdout.flush()
#print hp.rotator.Rotator(coord='cg').mat
for i in range(12*nside_standard**2):
    ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside_standard,i))
    equatorial_GSM_standard[i] = hpf.get_interp_val(gsm_standard, ang[0], ang[1])
print "done."
sys.stdout.flush()

print "Simulating GSM data...",
sys.stdout.flush()
sim_data = np.array([A[i].dot(equatorial_GSM_standard[pix_mask]) for i in range(len(A))]) + np.random.randn(len(data))/Ni**.5
print "done."
sys.stdout.flush()

#compute At.y
print "Computing At.y...",
sys.stdout.flush()
qaz = Ni * data
Atdata = np.array([A[:, i].dot(qaz) for i in range(len(A[0]))])
qaz = Ni * sim_data
Atsim_data = np.array([A[:, i].dot(qaz) for i in range(len(A[0]))])
print "done."
sys.stdout.flush()
del(A)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#load AtNiAi
rcondA = 1.e-5
AtNiAi_filename = datadir + tag + '_%i_%i.4AtNiAi%i'%(npix, npix, np.log10(rcondA))
print "Reading AtNiAi matrix from %s"%AtNiAi_filename
AtNiAi = np.fromfile(AtNiAi_filename, dtype='float32').reshape((npix, npix))

#compute raw x
x = np.zeros(12*nside**2, dtype='float32')
x[pix_mask] = AtNiAi.dot(Atdata)
#simulate solution
sim_sol = np.zeros(12*nside**2, dtype='float32')
sim_sol[pix_mask] = AtNiAi.dot(Atsim_data)
del(AtNiAi)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()





########generalized eigenvalue problem
####genSEv_filename = datadir + tag + '_%i_%i.genSEv_%s_%i'%(len(S), len(S), S_type, np.log10(rcondA))
####genSEvec_filename = datadir + tag + '_%i_%i.genSEvec_%s_%i'%(len(S), len(S), S_type, np.log10(rcondA))
####print "Computing generalized eigenvalue problem...",
####sys.stdout.flush()
####timer = time.time()
####genSEv, genSEvec = sla.eigh(S, b=AtNiAi)
####print "%f minutes used"%(float(time.time()-timer)/60.)
####genSEv.tofile(genSEv_filename)
####genSEvec.tofile(genSEvec_filename)

####genSEvecplot = np.zeros_like(equatorial_GSM_standard)
####for eigs in [-1,-2,1,0]:
    ####genSEvecplot[pix_mask] = genSEvec[:,eigs]
    ####hpv.mollview(genSEvecplot, coord=plotcoord, title=genSEv[eigs])

####plt.show()


SEi_filename = datadir + tag + '_%i_%i.4SEi_%s_%i'%(npix, npix, S_type, np.log10(rcondA))
if os.path.isfile(SEi_filename) and not force_recompute_SEi:
    print "Reading Wiener filter component...",
    sys.stdout.flush()
    SEi = np.fromfile(SEi_filename, dtype='float32').reshape((npix, npix))
else:
    SE_filename = datadir + tag + '_%i_%i.2S_%s_E%i'%(npix, npix, S_type, np.log10(rcondA))
    print "Reading S+E...",
    sys.stdout.flush()
    SE = np.fromfile(SE_filename, dtype='float32').reshape((npix, npix))

    print "Computing Wiener filter component...",
    sys.stdout.flush()
    timer = time.time()
    SEi = sla.inv(SE)#.astype('float32')
    del(SE)
    print "%f minutes used. Saving to disk..."%(float(time.time()-timer)/60.)
    sys.stdout.flush()
    SEi.tofile(SEi_filename)

print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

print "Applying Wiener filter step 1...",
sys.stdout.flush()
w_solution = np.zeros_like(x)
w_solution[pix_mask] = SEi.dot(x[pix_mask])
w_GSM = np.zeros_like(equatorial_GSM_standard)
w_GSM[pix_mask] = SEi.dot(equatorial_GSM_standard[pix_mask])
w_sim_sol = np.zeros_like(sim_sol)
w_sim_sol[pix_mask] = SEi.dot(sim_sol[pix_mask])
del(SEi)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()


print "Applying Wiener filter step 2...",
sys.stdout.flush()
###load S
S_filename = datadir + tag + '_%i_%i.2S_%s'%(npix, npix, S_type)
print "Reading S matrix %s..."%S_type,
sys.stdout.flush()
S = np.fromfile(S_filename, dtype = 'float32').reshape((npix, npix))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()


w_solution[pix_mask] = S.dot(w_solution[pix_mask])
w_GSM[pix_mask] = S.dot(w_GSM[pix_mask])
w_sim_sol[pix_mask] = S.dot(w_sim_sol[pix_mask])
del(S)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#load A matrix again
A_filename = datadir + tag + '_%i_%i.Axyri'%(len(data), npix)
print "Reading A matrix from %s"%A_filename
sys.stdout.flush()
A = np.fromfile(A_filename, dtype='float32').reshape((len(data), npix))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

chisq = np.sum(Ni * np.abs(A.dot(x[pix_mask]) - data)**2)/float(len(data) - np.sum(pix_mask))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()
chisq_sim = np.sum(Ni * np.abs(A.dot(sim_sol[pix_mask]) - sim_data)**2)/float(len(sim_data) - np.sum(pix_mask))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

print "chi^2:", chisq, chisq_sim

if False:
    hpv.mollview(equatorial_GSM_standard, min=0,max=5000, coord='CG', title='GSM')
    hpv.mollview(w_GSM, min=0,max=5000, coord='CG', title='wiener GSM')
    hpv.mollview(x, min=0,max=5000, coord='CG', title='raw solution, chi^2=%.2f'%chisq)
    hpv.mollview(w_solution, min=0,max=5000, coord='CG', title='wiener solution')
    hpv.mollview(sim_sol, min=0,max=5000, coord='CG', title='simulated noisy solution, chi^2=%.2f'%chisq_sim)
    hpv.mollview(w_sim_sol, min=0,max=5000, coord='CG', title='simulated wiener noisy solution')
else:
    #hpv.mollview(np.log10(gsm_standard), min=0,max=4, coord='G', title='GSM')
    hpv.mollview(np.log10(equatorial_GSM_standard), min=0,max=4, coord=plotcoord, title='GSM')
    hpv.mollview(np.log10(w_GSM), min=0,max=4, coord=plotcoord, title='wiener GSM')
    hpv.mollview(np.log10(x), min=0,max=4, coord=plotcoord, title='raw solution, chi^2=%.2f'%chisq)
    hpv.mollview(np.log10(w_solution), min=0,max=4, coord=plotcoord, title='wiener solution')
    hpv.mollview(np.log10(np.abs(w_solution)), min=0,max=4, coord=plotcoord, title='abs wiener solution')
    hpv.mollview(np.log10(sim_sol), min=0,max=4, coord=plotcoord, title='simulated noisy solution, chi^2=%.2f'%chisq_sim)
    hpv.mollview(np.log10(w_sim_sol), min=0,max=4, coord=plotcoord, title='simulated wiener noisy solution')
plt.show()
