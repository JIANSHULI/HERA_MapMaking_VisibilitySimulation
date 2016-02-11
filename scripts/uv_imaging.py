__author__ = 'omniscope'

import numpy as np
import numpy.linalg as la
import scipy.interpolate as si
import fitsio as fit
import healpy.visufunc as hpv
import healpy.pixelfunc as hpf
import healpy.sphtfunc as hps
import healpy.rotator as hpr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import sys, time, os, glob
import simulate_visibilities.simulate_visibilities as sv
from sklearn.decomposition import FastICA
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import resource

nt = 192
nUBL = 78
data = np.fromfile("/home/omniscope/data/GSM_data/absolute_calibrated_data/q3AL_0_abscal_xx_192_78_2016_01_20_avg", dtype='complex64').reshape((nt, nUBL))
ubls = np.fromfile("/home/omniscope/data/GSM_data/absolute_calibrated_data/q3AL_0_abscal_xx_78_3.ubl", dtype='float32').reshape((nUBL, 3))

uv = np.zeros((nt, 15, 15), dtype='complex64')

for u, ubl in enumerate(ubls):
   xi = int(np.round(ubl[0] / 3.))
   yi = int(np.round(ubl[1] / 3.))
   uv[:, xi, yi] = data[:, u]
   uv[:, -xi, -yi] = np.conjugate(data[:, u])

images = np.fft.fft2(uv)
images.astype('complex64').tofile("/home/omniscope/data/GSM_data/absolute_calibrated_data/img_q3AL_0_abscal_xx_192_78_2016_01_20_avg")
