import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.linalg as la




#########################################
#create a list of (theta,phi) for healpix32
#####################################
#nside=32
#length=12*nside**2
#positions = np.zeros([length,2],'float')
#for i in range(length):
	#positions[i] = np.array(hp.pixelfunc.pix2ang(nside,i))
	

#fopen = open('healpix32_theta_phi.txt','w')
#np.savetxt(fopen,[positions.flatten()])




###############################################
#import the map produced by mathematica, use healpy to turn it into alm
################################################
with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/beamhealpixmap.txt') as f:
	data = np.array([np.array([float(line)]) for line in f])

data = data.flatten()
beam_alm = hp.sphtfunc.map2alm(data,iter=3)

previous = hp.sphtfunc.map2alm(data,iter=3)
for i in range(4,15):
	diff = la.norm(hp.sphtfunc.map2alm(data,iter=i)-previous)
	previous = hp.sphtfunc.map2alm(data,iter=i)
	print [i,diff]


Blm={}
for l in range(21):
	for mm in range(-l,l+1):
		Blm[(l,mm)] = beam_alm[hp.sphtfunc.Alm.getidx(32,l,abs(mm))]



hp.visufunc.mollview(data)
plt.show()















