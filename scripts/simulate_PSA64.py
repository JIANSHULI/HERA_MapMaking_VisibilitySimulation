import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import time, ephem
import aipy as ap

######ubls to simulate
ubls = np.zeros((5*4, 3))
u = 0
for y in range(1,6):
        for x in range(4):
                ubls[u, 0] = x
                ubls[u, 1] = y
                u = u + 1
######other params
freqs = np.array([150.]) #MHz
calfile = 'psa6240_v003'



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

######load beam
Bl = 95
L = 128*3 - 1



vs.Blm = {}
for l in range(Bl + 1):
    for m in range (-l, l+1):
        vs.Blm[(l,m)] = 1.


timer = time.time()
correctres = vs.calculate_Bulm(L, 125, np.ones(3), Bl, verbose = False)
print "Time taken %.4f"%(float((time.time()-timer)/60.))


