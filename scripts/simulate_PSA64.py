import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
######ubls to simulate
ubls = np.zeros((5*4, 3))
u = 0
for y in range(1,6):
    for x in range(4):
        ubls[u, 0] = x
        ubls[u, 1] = y
        u = u + 1
######other params
freqs = np.arange(120.,190,10.) #MHz
calfile = 'psa6240_v003'
pol = 'x'


######get array  information and accurate ubl vectors
aa = ap.cal.get_aa(calfile,freqs/1000.)
antennaLocation = np.zeros((len(aa),3))
for i in range(len(aa.ant_layout)):
    for j in range(len(aa.ant_layout[0])):
        antennaLocation[aa.ant_layout[i][j]] = np.array([i, j, 0])
preciseAntennaLocation = (np.array([ant.pos for ant in aa]) * .299792458).dot(sv.rotatey_matrix(-(np.pi/2 - aa.lat)).transpose())

preciseMatrix = la.pinv(antennaLocation.transpose().dot(antennaLocation)).dot(antennaLocation.transpose().dot(preciseAntennaLocation))#unit location dot this matrix gives precise location
ubls = ubls.dot(preciseMatrix)
#print ubls

######construct visibility_simulator
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, aa.lat])



#for f in [0,1]:
    #beam = []
    #for theta in np.arange(0., np.pi/2., np.pi/50.):
        #beam.append(aa[0].bm_response((-np.sin(theta),0,np.cos(theta)), pol)[f])
    #plt.plot(beam,'r--')
    #beam = []
    #for theta in np.arange(0., np.pi/2., np.pi/50.):
        #beam.append(aa[0].bm_response((np.sin(theta)/1.414,np.sin(theta)/1.414,np.cos(theta)), pol)[f])
    #plt.plot(beam,'g--')
    #beam = []
    #for theta in np.arange(0., np.pi/2., np.pi/50.):
        #beam.append(aa[0].bm_response((0,np.sin(theta),np.cos(theta)), pol)[f])
    #plt.plot(beam,'b--')
#plt.show()
#quit()
######load beam
for nsideB in [4,8,16,32]:
    nside = 32
    Bl = nsideB*3 - 1
    L = nside*3 - 1

    beam_healpix = np.zeros((len(freqs),12*nsideB**2), dtype='float32')
    for f in range(len(freqs)):
        for i in range(12*nsideB**2):
            beam_healpix[f, i] = aa[0].bm_response(hpf.pix2vec(nsideB, i), pol)[f]
    print beam_healpix.shape
    beam_healpix.tofile('doc/PAPER_bm_%i_%i_%s_nside%i.bin'%(freqs[0], freqs[-1], pol, nsideB))
quit()
vs.Blm = {}
for l in range(Bl + 1):
    for m in range (-l, l+1):
        vs.Blm[(l,m)] = 1.


timer = time.time()
correctres = vs.calculate_Bulm(L, 125, np.ones(3), Bl, verbose = False)
print "Time taken %.4f"%(float((time.time()-timer)/60.))


