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

all_npzs = glob.glob('/home/omniscope/data/GSM_data/absolute_calibrated_data/cygcas*.npz')

npzs = [npz for npz in all_npzs if ('beam' not in npz and 'unpol' not in npz)]
print npzs

files = {}
for npz in npzs:
    Q = os.path.basename(npz).split('_')[1]
    files[Q] = np.load(npz)

colors = ['b', 'g', 'r', 'k', 'c', 'y', 'm']

for i, Q in enumerate(files.keys()):
    iquv = files[Q]['cyg_cas_iquv']
    mask = ~np.any(iquv[:, :, 1:3].reshape((len(iquv), 4)) == 0., axis=-1)
    frac = la.norm(iquv[..., 1:3], axis=-1) / iquv[..., 0]
    qfrac = iquv[..., 1] / iquv[..., 0]
    ufrac = iquv[..., 2] / iquv[..., 0]
    vfrac = np.abs(iquv[..., 3]) / iquv[..., 0]
    angle = np.angle(iquv[..., 1] + 1.j * iquv[..., 2]) % TPI / 2 * (180. / PI)
    freqs = files[Q]['freqs']

    plt.subplot(3, 1, 1)
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

    plt.subplot(3, 1, 2)
    plt.plot(freqs[mask], qfrac[mask, 0], colors[i] + '^', label='CygA Q Fraction')
    plt.plot(freqs[mask], qfrac[mask, 1], colors[i] + 'o', label='CasA Q Fraction')
    plt.plot(freqs[mask], ufrac[mask, 0], colors[i] + '^', fillstyle='none', label='CygA U Fraction')
    plt.plot(freqs[mask], ufrac[mask, 1], colors[i] + 'o', fillstyle='none', label='CasA U Fraction')

    if i == 0:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Fraction")
        # plt.title('polarization fraction')
        plt.ylim([-.2, .2])
        plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(freqs[mask], (angle[mask, 0] - 20)%180 + 20, colors[i] + '^', label='CygA Pol Angle')
    plt.plot(freqs[mask], angle[mask, 1], colors[i] + 'o', label='CasA Pol Angle')

    if i == 0:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Angle (degree)")
        plt.ylim([0, 200.])
        # plt.title('polarization angle')
        plt.legend()
plt.show()

#I plot
flux_func = {}
flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])
for i, Q in enumerate(files.keys()):
    iquv = files[Q]['cyg_cas_iquv']
    mask = ~np.any(iquv[:, :, 1:3].reshape((len(iquv), 4)) == 0., axis=-1)
    freqs = files[Q]['freqs']

    plt.plot(freqs[mask], iquv[mask, 0, 0], colors[i] + '^', label='CygA FLux')
    plt.plot(freqs[mask], flux_func['cyg'](freqs[mask]), colors[i], label='CygA Model FLux')
    plt.plot(freqs[mask], iquv[mask, 1, 0], colors[i] + 'o', label='CasA Flux')
    plt.plot(freqs[mask], flux_func['cas'](freqs[mask]), colors[i], label='CasA Model Flux')

    if i == 0:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Fraction")
        # plt.title('polarization fraction')
        plt.ylim([0, 1.5e4])
        plt.legend()

plt.show()
#
# exit()
# ####cross pol beam trial
#
# beam_npzs = [npz for npz in all_npzs if 'crossbeam' in npz]
# print beam_npzs
#
# files = {}
# for npz in beam_npzs:
#     frac = float(os.path.basename(npz).split('beam')[1][:5])
#     files[frac] = np.load(npz)
#
#
# iquvs = np.zeros((len(beam_npzs), 15, 2, 4))
# for i, frac in enumerate(sorted(files.keys())):
#     print frac
#     iquvs[i] = files[frac]['cyg_cas_iquv']
#
# lin_fraction = la.norm(iquvs[..., 1:3], axis=-1) / iquvs[..., 0]
# cir_fraction = iquvs[..., 3] / iquvs[..., 0]
#
# plt.subplot(2, 2, 1)
# plt.imshow(lin_fraction[..., 0])
# plt.subplot(2, 2, 2)
# plt.imshow(lin_fraction[..., 1])
# plt.subplot(2, 2, 3)
# plt.imshow(cir_fraction[..., 0])
# plt.subplot(2, 2, 4)
# plt.imshow(cir_fraction[..., 1])
# plt.show()
#
# plt.subplot(1, 2, 1)
# plt.plot(lin_fraction[0, ..., 0])
# plt.plot(lin_fraction[-1, ..., 0])
# plt.ylim([0, .15])
# plt.subplot(1, 2, 2)
# plt.plot(lin_fraction[0, ..., 1])
# plt.plot(lin_fraction[-1, ..., 1])
# plt.ylim([0, .15])
# plt.show()
#
#
# ####y beam trial
#
# beam_npzs = [npz for npz in all_npzs if 'ybeam' in npz]
# print beam_npzs
#
# files = {}
# for npz in beam_npzs:
#     frac = float(os.path.basename(npz).split('ybeam')[1][:5])
#     files[frac] = np.load(npz)
#
#
# iquvs = np.zeros((len(beam_npzs), 15, 2, 4))
# for i, frac in enumerate(sorted(files.keys())):
#     print frac
#     iquvs[i] = files[frac]['cyg_cas_iquv']
#
# lin_fraction = la.norm(iquvs[..., 1:3], axis=-1) / iquvs[..., 0]
# cir_fraction = iquvs[..., 3] / iquvs[..., 0]
#
# plt.subplot(2, 2, 1)
# plt.imshow(lin_fraction[..., 0])
# plt.subplot(2, 2, 2)
# plt.imshow(lin_fraction[..., 1])
# plt.subplot(2, 2, 3)
# plt.imshow(cir_fraction[..., 0])
# plt.subplot(2, 2, 4)
# plt.imshow(cir_fraction[..., 1])
# plt.show()