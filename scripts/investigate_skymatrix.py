import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import healpy.visufunc as hpv
import healpy as hp
import healpy.pixelfunc as hpf

nside = 8
A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_1360by768_redundantinfo_X5_q3x.bin'%nside,dtype='complex64').reshape((1360,768))

pca1 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm1.fits' + str(nside))
pca2 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm2.fits' + str(nside))
pca3 = hp.fitsfunc.read_map('/home/omniscope/simulate_visibilities/data/gsm3.fits' + str(nside))
gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
equatorial_GSM = np.zeros(12*nside**2,'float')
#rotate sky map
for i in range(12*nside**2):
    ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
    equatorial_GSM[i] = hpf.get_interp_val(gsm, ang[0], ang[1])
hpv.mollview(equatorial_GSM, return_projected_map=True, min=0,max=5000,fig=1,title='Original map')
#plt.show()
vis = A.dot(equatorial_GSM)
solution = la.pinv(A.conjugate().transpose().dot(A)).dot(A.conjugate().transpose().dot(vis))
hpv.mollview(solution, return_projected_map=True, min=0,max=5000,fig=2,title='Our solution')
#plt.show()
hpv.mollview(solution/equatorial_GSM, return_projected_map=True, min=0.75, max=1.25,fig=3,title='Ratio of solution/original')
plt.show()
#wA,_ = la.eig(A.conjugate().transpose().dot(A))

#for thresh in [.01]:#np.arange(.01, .2, .04):

    #maxnorm = np.max([la.norm(v) for v in A.transpose()])
    #print "max norm", maxnorm
    #B = np.array([v for v in A.transpose() if la.norm(v) > maxnorm*thresh]).transpose()
    #print "number of pixels over thresh %.1f"%(maxnorm*thresh), len(B[0])
    #wB,_ = la.eig(B.conjugate().transpose().dot(B))
    #print "condition", np.abs(wB[0])/np.abs(wB[-1])

#plt.plot(np.abs(wA))
#plt.plot(np.abs(wB))
#plt.show()
