import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import healpy.visufunc as hpv
import healpy as hp
import healpy.pixelfunc as hpf
import sys
import omnical.calibration_omni as omni
nt = 80
nside = 16
nUBL = 75#34
#nside = 16
#nUBL = 75
nps = 2#number of point sources to include
freq = 160.
thresh = 1
ubls = np.array([bl for bl in omni.read_redundantinfo('/home/omniscope/omnical/doc/redundantinfo_X5_q3x.bin')['ubl']*[1.5, 1.5, 0] if la.norm(bl) < thresh * (nside * 299.792458 / freq)])
if len(ubls) != nUBL:
    raise Exception('%i != %i!'%(len(ubls), nUBL))
#thresh2 = .7

nside_standard = 32#64
nt_standard = 80#160
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

if nside > 4:
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


for ps in [True, False]:#range(-5,5):
    thresh2 = 1
    subublindex = np.array([u for u in range(nUBL) if la.norm(ubls[u]) < thresh2 * (nside * 299.792458 / freq)])
    print "%i out of %i to include"%(len(subublindex), nUBL)
    ###################################plot actual errors and such, see if the gridding is too coarse

    #A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nUBL, nt, 12*nside**2))[subublindex].reshape((len(subublindex) * nt, 12*nside**2))/nside**2

    #ublindex = [u for u in range(len(ubls_standard)) if la.norm(ubls_standard[u]) < (nside * 299.792458 / freq)]
    A_standard = np.fromfile('/home/omniscope/data/GSM_data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nt_standard*nUBL, 12*nside_standard**2),dtype='complex64').reshape((nUBL, nt_standard, 12*nside_standard**2))[subublindex, ::(nt_standard/nt)].reshape((len(subublindex)*nt, 12*nside_standard**2))
    A_standard = np.concatenate((A_standard, np.fromfile('/home/omniscope/data/GSM_data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3y.bin'%(nUBL, nside_standard, nt_standard*nUBL, 12*nside_standard**2),dtype='complex64').reshape((nUBL, nt_standard, 12*nside_standard**2))[subublindex, ::(nt_standard/nt)].reshape((len(subublindex)*nt, 12*nside_standard**2))))

    average_size = nside_standard/nside
    B = np.zeros((12*nside**2,12*nside_standard**2),dtype='float32')
    for ringpix in range(12*nside_standard**2):
        B[hpf.nest2ring(nside, hpf.ring2nest(nside_standard, ringpix)/average_size**2), ringpix] = average_size**-2
    B.tofile('/home/omniscope/data/GSM_data/Bmatrix_nside%i_to_nside%i.bin'%(nside_standard,nside))
    if nside <= 4:
        equatorial_GSM = B.dot(equatorial_GSM_standard)

    ##plt.imshow(B.transpose().dot(B))
    ##plt.colorbar()
    ##plt.show()
    ##print "Computing BtBi...",
    ##sys.stdout.flush()
    ##rcondB = 1e-3
    ##BtBi = la.pinv(B.transpose().dot(B) , rcond=rcondB)#+ 10.**sigmap*np.identity(len(B[0])), rcond=rcondB)
    ##print "Done."
    ##sys.stdout.flush()
    ##BtBi.astype('float32').tofile('/home/omniscope/data/GSM_data/BtBi_nside%i_to_nside%i.bin'%(nside_standard,nside))
    #plt.imshow(np.abs(B.transpose().dot(B)))
    #plt.show()

    A = A_standard.dot(B.transpose())
    if ps:
        psAx = np.fromfile('/home/omniscope/data/GSM_data/AmatrixPS_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nt_standard*nUBL, nps),dtype='complex64').reshape((nUBL, nt_standard, nps))[subublindex, ::(nt_standard/nt)].reshape((len(subublindex)*nt, nps))
        psAy = np.fromfile('/home/omniscope/data/GSM_data/AmatrixPS_%iubl_nside%i_%iby%i_redundantinfo_X5_q3y.bin'%(nUBL, nside_standard, nt_standard*nUBL, nps),dtype='complex64').reshape((nUBL, nt_standard, nps))[subublindex, ::(nt_standard/nt)].reshape((len(subublindex)*nt, nps))
        A = np.concatenate((A, np.concatenate((psAx, psAy))),axis = 1)
    print "Computing AtAi...",
    sys.stdout.flush()
    rcondA = 1e-6
    AtAi = la.pinv(A.conjugate().transpose().dot(A), rcond=rcondA)
    print "Done."
    sys.stdout.flush()
    AtAi.astype('complex64').tofile('/home/omniscope/data/GSM_data/ABtABi_%iubl_nside%i_to_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nside, nt_standard*nUBL, 12*nside_standard**2))

    #plt.imshow(np.abs(AtAi.dot(A.transpose().conjugate().dot(A))))
    #plt.show()
    vis = A_standard.dot(equatorial_GSM_standard)

    vis = vis + .01 * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)
    print vis.shape, A.shape
    sys.stdout.flush()
    solution = AtAi.dot(A.conjugate().transpose().dot(vis))[:12*nside**2]/average_size**2
    #hpv.mollview(equatorial_GSM_standard, min=0,max=5000,fig=1,title='Original map')
    #hpv.mollview(equatorial_GSM, min=0,max=5000,fig=2,title='mean map')
    #hpv.mollview(solution, min=0,max=5000,fig=3,title='Our solution')
    #hpv.mollview(solution/B.dot(equatorial_GSM_standard), min=0, max=3, fig=4,title='sol/mean')
    hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-2, max=0,title='log10(relative error)')
plt.show()

#################################plot actual errors and such, see if adding PAPER PSA128 helps
#nPAPERUBL = 13
#A = np.concatenate((np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nt*nUBL, 12*nside**2)), np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_%iby%i_redundantinfo_PSA128_26ba_6bu_08-15-2014.bin'%(nside, nt*nPAPERUBL, 12*nside**2),dtype='complex64').reshape((nt*nPAPERUBL, 12*nside**2))))

#hpv.mollview(equatorial_GSM, return_projected_map=True, min=0,max=5000,fig=1,title='Original map')
#vis = A.dot(equatorial_GSM)
#vis = vis + .01 * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)
#solution = la.pinv(A.conjugate().transpose().dot(A), rcond = 1e-10).dot(A.conjugate().transpose().dot(vis))
#hpv.mollview(solution, return_projected_map=True, min=0,max=5000,fig=2,title='Our solution')
#hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-3, max=0,title='log10(relative error)')
#plt.show()

###################################investigate eigen values

#A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix__nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nt*nUBL, 12*nside**2))
#print "Computing eigenvalues for A...",
#sys.stdout.flush()
#wA,_ = la.eigh(A.conjugate().transpose().dot(A))
#print "done."
#sys.stdout.flush()

#thresh = 0.0005
#maxnorm = np.max([la.norm(v) for v in A.transpose()])
#print "max norm", maxnorm
#B = np.array([v for v in A.transpose() if la.norm(v) > maxnorm*thresh]).transpose()
#valid_pix = [i for i in range(len(A[0])) if la.norm(A[:,i]) > maxnorm*thresh]
#print "number of pixels over thresh %.1f"%(maxnorm*thresh), len(B[0])

#print "Computing eigenvalues for B...",
#sys.stdout.flush()
#wB,_ = la.eigh(B.conjugate().transpose().dot(B))
#print "condition", np.abs(wB[0])/np.abs(wB[-1])
#sys.stdout.flush()

#vis = A.dot(equatorial_GSM)
#noise = .01 * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)
#vis = vis + noise
#solution = np.zeros(12*nside**2, dtype='complex64')
#solution_noise = np.zeros(12*nside**2)
#BtBinv = la.pinv(B.conjugate().transpose().dot(B), rcond=1e-15)
#solution[valid_pix] = BtBinv.dot(B.conjugate().transpose().dot(vis))
#solution_noise[valid_pix] = BtBinv.dot(B.conjugate().transpose().dot(noise))
#hpv.mollview(np.log10(equatorial_GSM), return_projected_map=True, min=2.5,max=4,title='Original map')
#hpv.mollview(np.log10(np.real(solution)), return_projected_map=True, min=2.5,max=4,title='Solution_real')
##hpv.mollview(np.log10(np.imag(solution)), return_projected_map=True, min=2.5,max=4,title='Solution_imag')
#hpv.mollview(np.log10(np.abs(solution_noise)), return_projected_map=True, min=0.5,max=2,title='Solution noise')
#hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-3, max=0,title='log10(relative error) for thresh %.3f'%thresh)
#plt.show()

##plt.plot(np.abs(wA)/np.abs(wA)[0])
##plt.plot(np.abs(wB)/np.abs(wB)[0])

##for i in range(len(BtBinv)):
    ##BtBinv[i,:] = BtBinv[i,:] / BtBinv[i,i]**.5
    ##BtBinv[:,i] = BtBinv[:,i] / BtBinv[i,i]
#tmp = BtBinv
#for i in range(len(BtBinv)):
    #for j in range(len(BtBinv)):
        #BtBinv[i,j] = tmp[i,j] / tmp[i,i]**.5 / tmp[j,j]**.5

#plt.subplot('131')
#plot = plt.imshow(np.log10(np.abs(BtBinv)))
#plot.set_clim(-2,0)

#plt.subplot('132')
#plot = plt.imshow(np.log10(np.real(BtBinv)))
#plot.set_clim(-2,0)

#plt.subplot('133')
#plot = plt.imshow(np.log10(np.imag(BtBinv)))
#plt.colorbar()
#plot.set_clim(-2,0)
#plt.show()
#plt.imshow(np.imag(np.log10(BtBinv)))
#plt.colorbar()
#plt.show()
