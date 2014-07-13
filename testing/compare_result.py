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



with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/Visibilties_for_-6_m_south_-3_m_east_0_m_up_xx_pol_125.195_MHz.dat') as f:
	correct_v = np.array([np.array([float(x) for x in line.split()]) for line in f])

with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/spherical_result/sphericalharmonics_L20.txt') as f:
	spherical_20 = np.array([np.array([float(x) for x in line.split()]) for line in f])

with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/spherical_result/sphericalharmonics_L15.txt') as f:
	spherical_15 = np.array([np.array([float(x) for x in line.split()]) for line in f])
	
with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/spherical_result/sphericalharmonics_L10_test.txt') as f:
	spherical_10 = np.array([np.array([float(x) for x in line.split()]) for line in f])
	
with open('/home/eric/Dropbox/MIT/UROP/Simulate visibilities/spherical_result/sphericalharmonics_L10_05.txt') as f:
	spherical_10_shift = np.array([np.array([float(x) for x in line.split()]) for line in f])


p1 = spherical_10
p2 = spherical_10_shift

plt.plot([i[0] for i in p1],[i[1] for i in p1],'bo')
plt.plot([i[0] for i in p2],[i[1] for i in p2],'ko')
plt.plot([i[0] for i in p1],[i[2] for i in p1],'ro')
plt.plot([i[0] for i in p2],[i[2] for i in p2],'co')
plt.show()


