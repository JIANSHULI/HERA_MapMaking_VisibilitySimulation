import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import omnical.calibration_omni as omni

def pinv_sym(M, rcond = 1.e-15):
    eigvl,eigvc = la.eigh(M)
    eigvli = np.empty_like(eigvl)
    max_eigv = max(eigvl)
    for i in range(len(eigvli)):
        if eigvl[i] < max_eigv * rcond:
            eigvli[i] = 0
        else:
            eigvli[i] = 1/eigvl[i]
    return (eigvc*eigvli).dot(eigvc.transpose())

class InverseCholeskyMatrix:
    def __init__(self, matrix):
        if type(matrix).__module__ != np.__name__ or len(matrix.shape) != 2:
            raise TypeError("matrix must be a 2D numpy array");
        try:
            self.L = la.cholesky(matrix)#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
            self.Lt = self.L.conjugate().transpose()
            print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
        except:
            raise TypeError("cholesky failed. matrix is not positive definite.")

    def dotv(self, vector):
        return sla.solve_triangular(self.Lt, sla.solve_triangular(self.L, vector, lower=True), lower=False)

    def dotM(self, matrix):
        return np.array([self.dotv(v) for v in matrix.transpose()]).transpose()

tag = "q3_abscalibrated"
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = 440
nf = 1
nUBL = 75
nside = 8
S_type = 'uniform'
bnside = 8
lat_degree = 45.2977
force_recompute = False
force_recompute_AtNiAi = False
force_recompute_S = False
force_recompute_SEi = False

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
for p in ['x', 'y']:
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
    data[p] = (np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL)).transpose().flatten()*1.e-26*(C/freq)**2/kB/(4*np.pi/(12*nside**2))).conjugate()#there's a conjugate convention difference

data = np.concatenate((data['x'],data['y']))
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
#plt.plot(Ni['x'][::nt])
#plt.show()
Ni = np.concatenate((Ni['x'],Ni['y']))
Ni = np.concatenate((Ni/2, Ni/2))

pix_mask = np.array([la.norm(col) != 0 for col in A['x'].transpose()])
A = np.concatenate((A['x'][:, pix_mask],A['y'][:, pix_mask]))
A = np.concatenate((np.real(A), np.imag(A))).astype('float32')
npix = A.shape[1]
#compute AtNi
AtNi = A.transpose() * Ni


#compute AtNiAi
rcondA = 1.e-6
print "Computing AtNiAi matrix..."
timer = time.time()
##AtNiAi = la.pinv(AtNi.dot(A), rcond=rcondA)
#eigA, eigvc = la.eigh(AtNi.dot(A).astype('float64'))
#plt.plot(np.abs(eigA))
#print eigA[:10]

#AtNiAi = la.pinv(AtNi.dot(A) + 1e-15 * np.identity(len(A[0])), rcond = 1e-15)
#print "%f minutes used"%(float(time.time()-timer)/60.)
#eigA, eigvc = la.eigh(AtNi.dot(A) + 1e-15 * np.identity(len(A[0])))
#plt.plot(np.abs(eigA))
#print eigA[:10]
#plt.show()


AtNiAi = pinv_sym(AtNi.dot(A) + 1e-15 * np.identity(len(A[0])))
print "%f minutes used"%(float(time.time()-timer)/60.)
#eigA, eigvc = la.eigh(AtNiAi)
#plt.plot(eigA)
#print eigA[:10]

#la.cholesky(AtNiAi)

#plt.show()
#quit()


##compute raw x
x = np.zeros(12*nside**2, dtype='float32')
x[pix_mask] = AtNiAi.dot(AtNi.dot(data))


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
#sim_data = A.dot(equatorial_GSM_standard[pix_mask])
#sim_sol = np.zeros(12*nside**2)
#sim_sol[pix_mask] = AtNiAi.dot(AtNi.dot(sim_data))

###compute S


print "Computing S matrix %s..."%S_type,
sys.stdout.flush()
angular_scale = 2/(freq/300*np.max([la.norm(ubl) for ubl in ubls]))
S = np.identity(12 * nside**2)
S = np.maximum(np.array([hp.sphtfunc.smoothing(pix_vec, sigma = angular_scale, verbose = False) for pix_vec in S]), 0)[pix_mask][:, pix_mask]

ps_mask = (equatorial_GSM_standard[pix_mask] > 1000) #mask for points with high flux
S[ps_mask] = 0 #can't do these two operations at once
S[:, ps_mask] = 0 #can't do these two operations at once
S = S + np.diag(ps_mask.astype(int))
S = S * np.median(equatorial_GSM_standard)**2
rcondSE = 1e-2
#S2 = (S.dot(S.transpose()))**.5
#print la.norm(S-S2)/la.norm(S)
eigS, eigvc = la.eigh(S + 1e-2 * np.identity(len(S[0])))
#plt.plot(np.abs(eigS))
#print eigS[:10]
#plt.show()
#quit()

#print "Computing Wiener filter component...",
#sys.stdout.flush()
#timer = time.time()
#SEi = pinv_sym(S + AtNiAi, rcond = rcondSE).astype('float32')
#print "%f minutes used"%(float(time.time()-timer)/60.)

#print "Computing Wiener filter component...",
#sys.stdout.flush()
#S = S.astype('float64')
#timer = time.time()
#eigS, eigvc = la.eigh(S)
#eigA, eigvc = la.eigh(AtNiAi)
#eigSA, eigvc = la.eigh(S + AtNiAi)
#eigSAI, eigvc = la.eigh(S + AtNiAi + 1e9 * rcondSE * np.identity(len(S)))
#plt.plot(eigS, 'b')
#plt.plot(eigA, 'g')
#plt.plot(eigSA, 'r')
#plt.plot(eigSAI, 'c')
#plt.show()
#quit()
timer = time.time()
SEi = la.pinv(AtNiAi + S)
s0 = S.dot(SEi.dot(equatorial_GSM_standard[pix_mask]))
print "%f minutes used"%(float(time.time()-timer)/60.)

timer = time.time()
#cholesky = la.cholesky(AtNiAi + S + rcondSE * np.median(equatorial_GSM_standard)**2 * np.identity(len(S)))
SEi = InverseCholeskyMatrix(AtNiAi + S + rcondSE * np.median(equatorial_GSM_standard)**2 * np.identity(len(S)))
s1 = S.dot(SEi.dotv(equatorial_GSM_standard[pix_mask]))
print "%f minutes used"%(float(time.time()-timer)/60.)

timer = time.time()
SEi = la.pinv(AtNiAi + S + rcondSE * np.median(equatorial_GSM_standard)**2 * np.identity(len(S)))
s2 = S.dot(SEi.dot(equatorial_GSM_standard[pix_mask]))
print "%f minutes used"%(float(time.time()-timer)/60.)

plt.plot(s0)
plt.plot(s1)
plt.plot(s2)
plt.show()
