import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem, sys
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import omnical.calibration_omni as omni

######other params
freqs = np.arange(100.,200., 100./203.) #MHz
nf = len(freqs)
nt = 8640 #every ~10 seconds
year = 2013.0
calfile = 'psa6730_v000'#'psa6240_v003'
pol = 'yy'
datadir = '/home/omniscope/simulate_visibilities/data/'
opdir = '/home/omniscope/data/psa64gsm/'
initial_lst = 0.#in radian

####constants
MIN_GSM_NSIDE = 64


######cal file loaded
aa = ap.cal.get_aa(calfile,freqs/1000.)

#####get ephem observer
sa = ephem.Observer()
sa.lon = aa.lon
sa.lat = aa.lat


#####get ubl information########
#calibrator = omni.RedundantCalibrator_PAPER(aa)
#ubls = calibrator.compute_redundantinfo()
ubls = np.array([[0,0,0]])
unit_ubls = np.round(ubls)
######construct visibility_simulatorplt.imshow()a
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([initial_lst, aa.lat])

######load beam
nsideB = 32
Bl = nsideB*3 - 1

beam_healpix = np.zeros((len(freqs),12*nsideB**2), dtype='float32')

healpixvecs = np.array(hpf.pix2vec(nsideB, range(12*nsideB**2)))
for i, paper_angle in enumerate((healpixvecs[:, healpixvecs[2]>=0]).transpose().dot(sv.rotatez_matrix(-np.pi/2).transpose())):#in paper's bm_response convention, (x,y) = (0,1) points north.
    beam_healpix[:, i] = (aa[0].bm_response(paper_angle, pol[0]) * aa[0].bm_response(paper_angle, pol[1])).flatten()

#hpv.mollview(beam_healpix[0], title='PAPER_beam')
#plt.show()
#exit()

######load GSM weights
gsm_weights = np.loadtxt(datadir + '/components.dat')
gsm_weight0_f = si.interp1d(np.log(gsm_weights[:,0]), np.log(gsm_weights[:,1]), kind = 'linear', axis = 0)
gsm_weights_f = si.interp1d(gsm_weights[:,0], gsm_weights[:,2:], kind = 'linear', axis = 0)
#for i in range(4):
    #plt.plot(np.log(gsm_weights[:,0]), [gsm_weights_f(np.log(freq))[i]/gsm_weights_f(np.log(45))[i] for freq in gsm_weights[:,0]])
#plt.show()

GSMs = {}#will store equatorial coord maps for 3 components for the key word of nside

nside = MIN_GSM_NSIDE
print "Loading GSMs:",
while nside < 2 * 5 * np.max(la.norm(ubls, axis = 1))/(299.792458/np.max(freqs)) or nside == MIN_GSM_NSIDE:
    print nside,
    pca1 = hp.fitsfunc.read_map(datadir + 'gsm1.fits' + str(nside), verbose=False)
    pca2 = hp.fitsfunc.read_map(datadir + 'gsm2.fits' + str(nside), verbose=False)
    pca3 = hp.fitsfunc.read_map(datadir + 'gsm3.fits' + str(nside), verbose=False)

    equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000,stdtime=year))
    ang0, ang1 =hp.rotator.rotateDirection(equ2013_to_gal_matrix, hpf.pix2ang(nside, range(12*nside**2)))

    GSMs[nside] = np.zeros((3, 12*nside**2))
    GSMs[nside][0] = hpf.get_interp_val(pca1, ang0, ang1)
    GSMs[nside][1] = hpf.get_interp_val(pca2, ang0, ang1)
    GSMs[nside][2] = hpf.get_interp_val(pca3, ang0, ang1)
    nside = nside * 2
print "Done."
######start calculation
for ubl,unit_ubl in zip(ubls,unit_ubls):
    result = np.zeros((nt, nf), dtype='complex64')
    for f in range(len(freqs)):
        print "Starting UBL: %s at frequency %.3f MHz, %.2f wavelengths."%(ubl, freqs[f], la.norm(ubl)/(299.792458/freqs[f])),
        timer = time.time()

        vs.import_beam(beam_healpix[f])

        ######decide nside for GSM
        nside = MIN_GSM_NSIDE
        while nside < 5 * la.norm(ubl)/(299.792458/freqs[f]):#factor of 5 as safety margin
            nside = nside * 2
        nside = min(nside, 512)
        L = nside*3 - 1
        print "Using nside = %i for GSM."%nside,
        sys.stdout.flush()
        if nside not in GSMs:
            raise Exception("GSM not precomputed for nside=%i."%nside)
            #print "Loading..."
            #sys.stdout.flush()
            #pcas = [hp.fitsfunc.read_map(datadir + '/gsm%i.fits'%(i+1) + str(nside), verbose=False) for i in range(3)]
            #print "done.",
            #sys.stdout.flush()
            #####rotate sky map and get alm
            #print "Rotating GSM",
            #sys.stdout.flush()
            #GSMs[nside] = np.zeros((3,12*nside**2),'float')
            #for i in range(12*nside**2):
                #ang = g2e_rotator(hpf.pix2ang(nside,i))
                #for j in range(3):
                    #GSMs[nside][j,i] = hpf.get_interp_val(pcas[j], ang[0], ang[1])
            #print "Done."
        gsm_weights = np.append([np.exp(gsm_weight0_f(np.log(freqs[f])))], gsm_weights_f(freqs[f]))
        print "GSM weights:", gsm_weights
        sys.stdout.flush()

        gsm = gsm_weights[0]*(gsm_weights[1]*GSMs[nside][0] + gsm_weights[2]*GSMs[nside][1] + gsm_weights[3]*GSMs[nside][2])
        alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(gsm), 3 * nside - 1)
        result[:,f] = vs.calculate_visibility(sv.expand_real_alm(alm), d=ubl, freq=freqs[f], nt=nt, L = 3*nside-1, verbose = True)

        print "Time taken %.4f min."%(float((time.time()-timer)/60.))
    result.tofile(opdir+'/Visibilties_for_%i_south_%i_east_0_up_%s_pol_%i_step_%i_freq_%.1f.bin'%(unit_ubl[0],unit_ubl[1],pol,nt,nf,year))

