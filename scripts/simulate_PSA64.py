import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem, sys
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import scipy.interpolate as si
######ubls to simulate
ny = 3
ubls = np.zeros((ny*7, 3))
u = 0
for y in range(1,1+ny):
    for x in range(-3,4):
        ubls[u, 0] = x
        ubls[u, 1] = y
        u = u + 1
######other params
freqs = np.arange(120.,130,10.) #MHz
nt = 8640 #every ~10 seconds
calfile = 'psa6240_v003'
pol = 'x'
datadir = '/home/omniscope/simulate_visibilities/data/'
opdir = '/home/omniscope/data/psa64gsm/'
######get array  information and accurate ubl vectors
aa = ap.cal.get_aa(calfile,freqs/1000.)
antennaLocation = np.zeros((len(aa),3))
for i in range(len(aa.ant_layout)):
    for j in range(len(aa.ant_layout[0])):
        antennaLocation[aa.ant_layout[i][j]] = np.array([i, j, 0])
preciseAntennaLocation = (np.array([ant.pos for ant in aa]) * .299792458).dot(sv.rotatey_matrix(-(np.pi/2 - aa.lat)).transpose())

preciseMatrix = la.pinv(antennaLocation.transpose().dot(antennaLocation)).dot(antennaLocation.transpose().dot(preciseAntennaLocation))#unit location dot this matrix gives precise location
unit_ubls = ubls
ubls = ubls.dot(preciseMatrix)
#print ubls

######construct visibility_simulator
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, aa.lat])

######load beam
nsideB = 2
Bl = nsideB*3 - 1

beam_healpix = np.zeros((len(freqs),12*nsideB**2), dtype='float32')
for f in range(len(freqs)):
    for i in range(12*nsideB**2):
        beam_healpix[f, i] = aa[0].bm_response(sv.rotatez_matrix(-np.pi/2).dot(hpf.pix2vec(nsideB, i)), pol)[f]#in paper's bm_response convention, (x,y) = (0,1) points north.

######load GSM weights
    gsm_weights = np.loadtxt(datadir + '/components.dat')
    gsm_weights_f = si.interp1d(np.log(gsm_weights[:,0]), gsm_weights[:,1:], kind = 'linear', axis = 0)
    #for i in range(4):
        #plt.plot(np.log(gsm_weights[:,0]), [gsm_weights_f(np.log(freq))[i]/gsm_weights_f(np.log(45))[i] for freq in gsm_weights[:,0]])
    #plt.show()

GSMs = {}#will store equatorial coord maps for 3 components for the key word of nside
g2e_rotator = hp.rotator.Rotator(coord='cg')
######start calculation
for ubl,unit_ubl in zip(ubls,unit_ubls):
    for f in range(len(freqs)):
        print "Starting UBL: %s at frequency %.3f MHz, %.2f wavelengths."%(ubl, freqs[f], la.norm(ubl)/(299.792458/freqs[f])),
        timer = time.time()

        vs.import_beam(beam_healpix[f])

        ######decide nside for GSM
        nside = 2
        while nside < 5*la.norm(ubl)/(299.792458/freqs[f]):#factor of 5 as safety margin
            nside = nside * 2
        nside = min(nside, 512)
        L = nside*3 - 1
        print "Using nside = %i for GSM."%nside,
        sys.stdout.flush()
        if nside not in GSMs:
            print "Loading..."
            sys.stdout.flush()
            pcas = [hp.fitsfunc.read_map(datadir + '/gsm%i.fits'%(i+1) + str(nside), verbose=False) for i in range(3)]
            print "done.",
            sys.stdout.flush()
            ####rotate sky map and get alm
            print "Rotating GSM",
            sys.stdout.flush()
            GSMs[nside] = np.zeros((3,12*nside**2),'float')
            for i in range(12*nside**2):
                ang = g2e_rotator(hpf.pix2ang(nside,i))
                for j in range(3):
                    GSMs[nside][j,i] = hpf.get_interp_val(pcas[j], ang[0], ang[1])
            print "Done."
        gsm_weights = gsm_weights_f(np.log(freqs[f]))
        print "GSM weights:", gsm_weights
        sys.stdout.flush()

        gsm = gsm_weights[0]*(gsm_weights[1]*GSMs[nside][0] + gsm_weights[2]*GSMs[nside][1] + gsm_weights[3]*GSMs[nside][2])
        alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(gsm), 3 * nside - 1)
        result = vs.calculate_visibility(sv.expand_real_alm(alm), d=ubl, freq=freqs[f], nt=nt, L = 3*nside-1, verbose = True)
        result.astype('complex64').tofile(opdir+'/Visibilties_for_%i_south_%i_east_0_up_%s%s_pol_%.1f_MHz_%i_step.bin'%(unit_ubl[0],unit_ubl[1],pol,pol,freqs[f],nt))
        print "Time taken %.4f"%(float((time.time()-timer)/60.))


