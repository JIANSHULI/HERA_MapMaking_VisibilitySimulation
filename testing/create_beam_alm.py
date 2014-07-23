import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import scipy.linalg as la




#########################################
#create a list of (theta,phi) for healpix32
#####################################
nside=32
length=12*nside**2
positions = np.zeros([length,2],'float')
for i in range(length):
	positions[i] = np.array(hp.pixelfunc.pix2ang(nside,i))
	

fopen = open('healpix64_theta_phi.txt','w')
np.savetxt(fopen,[positions.flatten()])




###############################################
#import the map produced by mathematica, use healpy to turn it into alm
################################################
with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/beamhealpixmap.txt') as f:
	data = np.array([np.array([float(line)]) for line in f])

#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/beamhealpixmap_piover4.txt') as f:
	#data1 = np.array([np.array([float(line)]) for line in f])


#with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/beamhealpixmap_test.txt') as f:
	#data2 = np.array([np.array([float(line)]) for line in f])

#data1 = data1.flatten()
#data2 = data2.flatten()




data = data.flatten()
beam_alm = hp.sphtfunc.map2alm(data,10,iter=3)    #,lmax=6

#previous = hp.sphtfunc.map2alm(data,iter=3)
#for i in range(4,15):
	#diff = la.norm(hp.sphtfunc.map2alm(data,iter=i)-previous)
	#previous = hp.sphtfunc.map2alm(data,iter=i)
	#print [i,diff]


Blm={}
for l in range(6):
	for mm in range(-l,l+1):
		if mm >= 0:
			Blm[(l,mm)] = (1.0j)**mm*beam_alm[hp.sphtfunc.Alm.getidx(10,l,abs(mm))]
		if mm < 0:
			Blm[(l,mm)] = np.conj((1.0j)**mm*beam_alm[hp.sphtfunc.Alm.getidx(10,l,abs(mm))])


#hp.visufunc.mollview(data)
##hp.visufunc.mollview(data1)
#hp.visufunc.mollview(data2)
#plt.show()



for i in range(5):
	for j in range(-i,i+1):
		print (i,j)
		print Blm[i,j]











