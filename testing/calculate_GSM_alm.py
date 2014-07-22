import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
import scipy.special as ssp
import math as m
import cmath as cm
import healpy as hp
import healpy.pixelfunc as hpf
import matplotlib.pyplot as plt
import time



#path='/home/eric/Documents/gsm_data/component_maps_408locked.dat'
#with open(path) as f:
	#gsmhealpix = np.array([np.array([float(x) for x in line.split()]) for line in f])

#pca1 = np.array([pix[0] for pix in gsmhealpix])
#pca2 = np.array([pix[1] for pix in gsmhealpix])
#pca3 = np.array([pix[2] for pix in gsmhealpix])
#hp.mollview(pca1, title='first PCA')
#hp.mollview(pca2, title='first PCA')
#hp.mollview(pca3, title='first PCA')
#plt.show()


pca1 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm1.fits64')
pca2 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm2.fits64')
pca3 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm3.fits64')
gsm = 422.952*(0.307706*pca1 -0.281772*pca2+0.0123976*pca3)



nside=64
equatorial_GSM = np.zeros(12*nside**2,'float')
#rotate sky map
for i in range(12*nside**2):
	ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i)) 
	pixindex, weight = hpf.get_neighbours(nside,ang[0],ang[1])
	for pix in range(len(pixindex)):
		equatorial_GSM[i] += np.log(weight[pix]*gsm[pixindex[pix]])


almlist = hp.sphtfunc.map2alm(equatorial_GSM)
alm={}
for l in range(96):
	for mm in range(-l,l+1):
		alm[(l,mm)] = almlist[hp.sphtfunc.Alm.getidx(nside*3-1,l,abs(mm))]

hp.visufunc.mollview(equatorial_GSM)
plt.show()


