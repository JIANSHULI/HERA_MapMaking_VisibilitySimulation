import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import healpy.visufunc as hpv
import healpy as hp
import healpy.pixelfunc as hpf
import sys

#nside = 8
#nUBL = 34
nside = 16
nUBL = 75

pca1 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm1.fits' + str(nside))
pca2 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm2.fits' + str(nside))
pca3 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm3.fits' + str(nside))
gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
equatorial_GSM = np.zeros(12*nside**2,'float')
#rotate sky map
print "Rotating GSM...",
sys.stdout.flush()
for i in range(12*nside**2):
    ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
    equatorial_GSM[i] = hpf.get_interp_val(gsm, ang[0], ang[1])
print "done."
sys.stdout.flush()
#################################plot actual errors and such
#for nt in [40, 80]:
    #A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nt*nUBL, 12*nside**2))
    ##hpv.mollview(equatorial_GSM, return_projected_map=True, min=0,max=5000,fig=1,title='Original map')
    #vis = A.dot(equatorial_GSM)
    #vis = vis + .01 * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)
    #solution = la.pinv(A.conjugate().transpose().dot(A)).dot(A.conjugate().transpose().dot(vis))
    ##hpv.mollview(solution, return_projected_map=True, min=0,max=5000,fig=2,title='Our solution')
    #hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-3, max=0,title='log10(relative error)')
#plt.show()

###################################investigate eigen values
nt = 80
A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nt*nUBL, 12*nside**2))
print "Computing eigenvalues for A...",
sys.stdout.flush()
wA,_ = la.eigh(A.conjugate().transpose().dot(A))
print "done."
sys.stdout.flush()

thresh = 0.0005
maxnorm = np.max([la.norm(v) for v in A.transpose()])
print "max norm", maxnorm
B = np.array([v for v in A.transpose() if la.norm(v) > maxnorm*thresh]).transpose()
valid_pix = [i for i in range(len(A[0])) if la.norm(A[:,i]) > maxnorm*thresh]
print "number of pixels over thresh %.1f"%(maxnorm*thresh), len(B[0])

print "Computing eigenvalues for B...",
sys.stdout.flush()
wB,_ = la.eigh(B.conjugate().transpose().dot(B))
print "condition", np.abs(wB[0])/np.abs(wB[-1])
sys.stdout.flush()

vis = A.dot(equatorial_GSM)
noise = .01 * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)
vis = vis + noise
solution = np.zeros(12*nside**2, dtype='complex64')
solution_noise = np.zeros(12*nside**2)
BtBinv = la.pinv(B.conjugate().transpose().dot(B), rcond=1e-10)
solution[valid_pix] = BtBinv.dot(B.conjugate().transpose().dot(vis))
solution_noise[valid_pix] = BtBinv.dot(B.conjugate().transpose().dot(noise))
hpv.mollview(np.log10(equatorial_GSM), return_projected_map=True, min=2.5,max=4,title='Original map')
hpv.mollview(np.log10(np.real(solution)), return_projected_map=True, min=2.5,max=4,title='Solution_real')
#hpv.mollview(np.log10(np.imag(solution)), return_projected_map=True, min=2.5,max=4,title='Solution_imag')
hpv.mollview(np.log10(np.abs(solution_noise)), return_projected_map=True, min=0.5,max=2,title='Solution noise')
hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-3, max=0,title='log10(relative error) for thresh %.3f'%thresh)
plt.show()

#plt.plot(np.abs(wA)/np.abs(wA)[0])
#plt.plot(np.abs(wB)/np.abs(wB)[0])
plt.imshow(np.log10(np.abs(BtBinv)/np.max(np.abs(BtBinv))))
plt.show()
