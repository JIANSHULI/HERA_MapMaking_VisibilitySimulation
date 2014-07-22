import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
import scipy.special as ssp
import math as m
import cmath as cm
from wignerpy._wignerpy import wigner3j, wigner3jvec
from random import random
import healpy as hp
import matplotlib.pyplot as plt



with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/Visibilties_for_-6_m_south_-3_m_east_0_m_up_xx_pol_125.195_MHz.dat') as f:
	correct_v = np.array([np.array([float(x) for x in line.split()]) for line in f])

with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/sphericalharmonics_L10.txt') as f:
	spherical_10 = np.array([np.array([float(x) for x in line.split()]) for line in f])

with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/sphericalharmonics_L95.txt') as f:
	spherical_95 = np.array([np.array([float(x) for x in line.split()]) for line in f])
	
with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/sphericalharmonics_L10_test.txt') as f:
	spherical_10test = np.array([np.array([float(x) for x in line.split()]) for line in f])
	
with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/sphericalharmonics_L10_piover2.txt') as f:
	spherical_10r = np.array([np.array([float(x) for x in line.split()]) for line in f])


p1 = correct_v
p2 = spherical_10

shift=0
plt.plot([(i[0]+shift)%24 for i in p1],[i[1] for i in p1],'bo')
plt.plot([i[0] for i in p2],[i[1] for i in p2],'ko')
plt.plot([(i[0]+shift)%24 for i in p1],[i[2] for i in p1],'ro')
plt.plot([i[0] for i in p2],[i[2] for i in p2],'co')
plt.show()


