__author__ = 'omniscope'

import numpy as np
import numpy.linalg as la
import omnical.calibration_omni as omni
import matplotlib.pyplot as plt
import scipy.interpolate as si
import glob, sys, os, ephem, warnings
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import matplotlib.pyplot as plt
import simulate_visibilities.simulate_visibilities as sv
import time
PI = np.pi
TPI = 2*np.pi

npzs = glob.glob('/home/omniscope/data/GSM_data/absolute_calibrated_data/cygcas*.npz')
print npzs

files = {}
for npz in npzs:
    Q = os.path.basename(npz).split('_')[1]
    files[Q] = np.load(npz)

colors = ['b', 'g', 'r', 'k', 'c']

for i, Q in enumerate(files.keys()):
    iquv = files[Q]['cyg_cas_iquv']
    mask = ~np.any(iquv[:, :, 1:3].reshape((len(iquv), 4)) == 0., axis=-1)
    frac = la.norm(iquv[..., 1:3], axis=-1) / iquv[..., 0]
    vfrac = np.abs(iquv[..., 3]) / iquv[..., 0]
    angle = np.angle(iquv[..., 1] + 1.j * iquv[..., 2]) % TPI / 2 * (180. / PI)
    freqs = files[Q]['freqs']

    plt.subplot(2, 1, 1)
    plt.plot(freqs[mask], frac[mask, 0], colors[i] + '^', label='CygA Linear Fraction')
    plt.plot(freqs[mask], frac[mask, 1], colors[i] + 'o', label='CasA Linear Fraction')
    plt.plot(freqs[mask], vfrac[mask, 0], colors[i] + '^', fillstyle='none', label='CygA Circular Fraction')
    plt.plot(freqs[mask], vfrac[mask, 1], colors[i] + 'o', fillstyle='none', label='CasA Circular Fraction')

    if i == 0:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Fraction")
        # plt.title('polarization fraction')
        plt.ylim([0, .5])
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs[mask], (angle[mask, 0] - 20)%180 + 20, colors[i] + '^', label='CygA Pol Angle')
    plt.plot(freqs[mask], angle[mask, 1], colors[i] + 'o', label='CasA Pol Angle')

    if i == 0:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Angle (degree)")
        plt.ylim([0, 200.])
        # plt.title('polarization angle')
        plt.legend()
plt.show()