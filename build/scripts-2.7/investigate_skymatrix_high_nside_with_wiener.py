import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import healpy.visufunc as hpv
import healpy as hp
import healpy.pixelfunc as hpf
import sys, os
import omnical.calibration_omni as omni

tavg = 2
force_recompute = False
nside = 16
nt = 80
nside_standard = 16#64
nt_standard = 80#160
nUBL = 75#34
#nside = 16
#nUBL = 75
nps = 2#number of point sources to include
freq = 160.
thresh = 1
ubls = np.array([bl for bl in omni.read_redundantinfo('/home/omniscope/omnical/doc/redundantinfo_X5_q3x.bin')['ubl']*[1.5, 1.5, 0] if la.norm(bl) < thresh * (nside * 299.792458 / freq)])
if len(ubls) != nUBL:
    raise Exception('%i != %i!'%(len(ubls), nUBL))

#compute S
angular_scale = 1/(freq/300*np.max([la.norm(ubl) for ubl in ubls]))
print "Computing S matrix...",
sys.stdout.flush()
S = np.identity(12 * nside**2)
S = np.maximum(np.array([hp.sphtfunc.smoothing(pix_vec, sigma = angular_scale, verbose = False) for pix_vec in S]), 0)


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

if nside != nside_standard:
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
else:
    equatorial_GSM = equatorial_GSM_standard


for rcondA in [1e-6]:#range(-5,5):
    for noise in [.01]:
        #rcondA = 1e-6
        #noise = 0.0000001#.01
        ps = True
        thresh2 = 1
        subublindex = np.array([u for u in range(nUBL) if la.norm(ubls[u]) < thresh2 * (nside * 299.792458 / freq)])
        print "%i out of %i to include"%(len(subublindex), nUBL)
        ###################################plot actual errors and such, see if the gridding is too coarse

        #A = np.fromfile('/home/omniscope/simulate_visibilities/data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside, nt*nUBL, 12*nside**2),dtype='complex64').reshape((nUBL, nt, 12*nside**2))[subublindex,::tavg].reshape((len(subublindex) * nt/tavg, 12*nside**2))/nside**2

        #ublindex = [u for u in range(len(ubls_standard)) if la.norm(ubls_standard[u]) < (nside * 299.792458 / freq)]
        A_standard = np.fromfile('/home/omniscope/data/GSM_data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nt_standard*nUBL, 12*nside_standard**2),dtype='complex64').reshape((nUBL, nt_standard, 12*nside_standard**2))[subublindex, ::(tavg*nt_standard/nt)].reshape((len(subublindex)*nt/tavg, 12*nside_standard**2))
        A_standard = np.concatenate((A_standard, np.fromfile('/home/omniscope/data/GSM_data/Amatrix_%iubl_nside%i_%iby%i_redundantinfo_X5_q3y.bin'%(nUBL, nside_standard, nt_standard*nUBL, 12*nside_standard**2),dtype='complex64').reshape((nUBL, nt_standard, 12*nside_standard**2))[subublindex, ::(tavg*nt_standard/nt)].reshape((len(subublindex)*nt/tavg, 12*nside_standard**2))))

        average_size = nside_standard/nside
        Bfile = '/home/omniscope/data/GSM_data/Bmatrix_nside%i_to_nside%i.bin'%(nside_standard,nside)
        if os.path.isfile(Bfile):
            B = np.fromfile(Bfile, dtype='float32').reshape((12*nside**2,12*nside_standard**2))
        else:
            B = np.zeros((12*nside**2,12*nside_standard**2), dtype='float32')
            for ringpix in range(12*nside_standard**2):
                B[hpf.nest2ring(nside, hpf.ring2nest(nside_standard, ringpix)/average_size**2), ringpix] = average_size**-2
            B.tofile('/home/omniscope/data/GSM_data/Bmatrix_nside%i_to_nside%i.bin'%(nside_standard,nside))
        if nside <= 4 and nside != nside_standard:
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
        if nside != nside_standard:
            print "Averaging A_standard...",
            sys.stdout.flush()
            A = A_standard.dot(B.transpose())
        else:
            A = A_standard

        print "Appending point sources...",
        sys.stdout.flush()
        if ps:
            point_sources = [[np.pi*((23+(23.+26./60.)/60.)/12.), np.pi*(58.+48./60.)/180.], [np.pi*((19.+(59.+28.3566/60.)/60.)/12.), np.pi*(40.+(44.+2.096/60.)/60.)/180.]]
            psAx = np.fromfile('/home/omniscope/data/GSM_data/AmatrixPS_%iubl_nside%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nt_standard*nUBL, nps),dtype='complex64').reshape((nUBL, nt_standard, nps))[subublindex, ::(tavg*nt_standard/nt)].reshape((len(subublindex)*nt/tavg, nps))
            psAy = np.fromfile('/home/omniscope/data/GSM_data/AmatrixPS_%iubl_nside%i_%iby%i_redundantinfo_X5_q3y.bin'%(nUBL, nside_standard, nt_standard*nUBL, nps),dtype='complex64').reshape((nUBL, nt_standard, nps))[subublindex, ::(tavg*nt_standard/nt)].reshape((len(subublindex)*nt/tavg, nps))
            A = np.concatenate((A, np.concatenate((psAx, psAy))),axis = 1)


        AtAi_file = '/home/omniscope/data/GSM_data/ABtABi_%iubl_nside%i_to_nside%i_rcond%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nside, np.log10(rcondA), len(A[0]), len(A[0]))
        if os.path.isfile(AtAi_file):
            print "Reading AtAi...",
            sys.stdout.flush()
            AtAi = np.fromfile(AtAi_file, dtype = 'complex64').reshape((len(A[0]), len(A[0])))
        else:
            print "Computing AtA eigen values...",
            sys.stdout.flush()

            ev, _ = la.eigh(A.conjugate().transpose().dot(A))
            maxev = np.max(np.abs(ev))
            print "Computing AtAi...",
            sys.stdout.flush()
            AtAi = la.pinv(A.conjugate().transpose().dot(A) + maxev * rcondA * np.identity(len(A[0])))#, rcond=rcondA)
            print "Done."
            sys.stdout.flush()
            AtAi.astype('complex64').tofile(AtAi_file)


        print "Simulating visibilities...",
        sys.stdout.flush()
        vis = A_standard.dot(equatorial_GSM_standard)

        print "Adding noise to visibilities...",
        sys.stdout.flush()

        vis = vis + noise * (2**-.5) * np.mean(np.abs(vis)) * (np.random.randn(len(vis)) + np.random.randn(len(vis)) * 1.j)

        ###computing solution
        print "Computing solution...",
        sys.stdout.flush()
        raw_solution = AtAi.dot(A.conjugate().transpose().dot(vis))/average_size**2
        solution = np.real(raw_solution[:12*nside**2])

        #####wiener filter

        N = AtAi[:12*nside**2, :12*nside**2] * noise**2
        wiener_file = '/home/omniscope/data/GSM_data/SNi_Gaussian_realN_%iubl_nside%i_to_nside%i_rcond%i_noise%i_%iby%i_redundantinfo_X5_q3x.bin'%(nUBL, nside_standard, nside, np.log10(rcondA), np.log10(noise), len(N), len(N))
        if os.path.isfile(wiener_file) and not force_recompute:
            print "Reading Wiener filter component...",
            sys.stdout.flush()
            try:
                SNi = np.fromfile(wiener_file, dtype = 'float64').reshape((len(N), len(N)))
            except ValueError:
                SNi = np.fromfile(wiener_file, dtype = 'complex128').reshape((len(N), len(N)))
        else:
            print "Computing Wiener filter component...",
            sys.stdout.flush()
            SNi = la.pinv(S + np.real(N))#should i do abs(N) ?
            print SNi.dtype
            SNi.tofile(wiener_file)
        print "Computing Wiener filter...",
        sys.stdout.flush()
        wiener = S.dot(SNi)

        print "Applying Wiener filter...",
        sys.stdout.flush()
        w_solution = wiener.dot(solution)
        #plt.imshow(np.log10(wiener))
        #plt.colorbar()
        #plt.show()

        ###adding back point source:
        if ps:
            for i in range(len(point_sources)):
                ra, dec = point_sources[i]
                theta = np.pi/2 - dec
                phi = ra
                solution[hpf.ang2pix(nside, theta, phi)] += raw_solution[12*nside**2 + i]
                w_solution[hpf.ang2pix(nside, theta, phi)] += raw_solution[12*nside**2 + i]


        #plt.plot(np.abs(ev))
        #plt.show()
        #hpv.mollview(equatorial_GSM_standard, min=0,max=5000,fig=1,title='Original map')
        #if nside != nside_standard:
            #hpv.mollview(equatorial_GSM, min=0,max=5000,fig=2,title='mean map')
        hpv.mollview(solution, min=0,max=5000,title='Raw solution, rcond = %i, noise = %i'%(np.log10(rcondA), np.log10(noise)))
        hpv.mollview(np.log10(np.abs(solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-2, max=0,title='log10(relative error), rcond = %i, noise = %i'%(np.log10(rcondA), np.log10(noise)))
        hpv.mollview(w_solution, min=0,max=5000,title='Wiener solution, rcond = %i, noise = %i'%(np.log10(rcondA), np.log10(noise)))
        hpv.mollview(np.log10(np.abs(w_solution-equatorial_GSM)/equatorial_GSM), return_projected_map=True, min=-2, max=0,title='log10(relative error wiener), rcond = %i, noise = %i'%(np.log10(rcondA), np.log10(noise)))
        #hpv.mollview(solution/B.dot(equatorial_GSM_standard), min=0, max=3, fig=4,title='sol/mean')

        print " "
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
