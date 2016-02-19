import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, datetime, warnings
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob

PI = np.pi
TPI = np.pi * 2


tag = "mwa_aug23_eor0" #
datatag = '.dat'#
vartag = ''#''#
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/'



nf = 1
data_filename = glob.glob(datadir + tag + '_xx_*_*' + datatag)[0]
nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('xx_')[-1]
nt = int(nt_nUBL.split('_')[0])
nUBL = int(nt_nUBL.split('_')[1])

tfactor = 4
nt2 = nt / tfactor


for p in ['x', 'y']:
    # get Ni (1/variance) and data
    var_filename = datadir + tag + '_%s%s_%i_%i' % (p, p, nt, nUBL) + vartag + '.var'
    var_filename2 = datadir + tag + '_%s%s_%i_%i' % (p, p, nt2, nUBL) + vartag + '.var'
    data_filename = datadir + tag + '_%s%s_%i_%i' % (p, p, nt, nUBL) + datatag
    data_filename2 = datadir + tag + '_%s%s_%i_%i' % (p, p, nt2, nUBL) + datatag + 't%i'%tfactor

    tf_filename = datadir + tag + '_%s%s_%i_%i.tf' % (p, p, nt, nf)
    tf_filename2 = datadir + tag + '_%s%s_%i_%i.tf' % (p, p, nt2, nf)

    (np.sum(np.fromfile(var_filename, dtype='float32').reshape((nUBL, nt2, tfactor)), axis=-1) / tfactor**2).tofile(var_filename2)
    (np.sum(np.fromfile(data_filename, dtype='complex64').reshape((nUBL, nt2, tfactor)), axis=-1) / tfactor).tofile(data_filename2)
    np.mean(np.fromfile(tf_filename, dtype='complex64').reshape((nt2, tfactor, nf)), axis=1).tofile(tf_filename2)
