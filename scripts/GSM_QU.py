__author__ = 'omniscope'

import numpy as np
import scipy.interpolate as si
import fitsio as fit
import healpy.visufunc as hpv
import healpy.pixelfunc as hpf
import healpy.sphtfunc as hps
import healpy.rotator as hpr
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import sys, time, os
import simulate_visibilities.simulate_visibilities as sv
from sklearn.decomposition import FastICA


def find_regions(nan_mask):
    nf = nan_mask.shape[0]
    mother_npix = nan_mask.shape[1]
    incomplete_fs = np.arange(nf)[nan_mask.any(axis=1)]
    complete_fs = np.arange(nf)[~nan_mask.any(axis=1)]
    n_incomplete = len(incomplete_fs)#number of incomplete maps
    max_n_region = 2**n_incomplete #maximum number of possible regions
    region_mask_list = []
    region_indices_list = []

    region_sum = np.zeros(mother_npix)
    for n, f in enumerate(incomplete_fs):
        region_sum += nan_mask[f] * 2**n

    for n_reg in range(max_n_region):
        fill_mask = np.array([not bool(int(c)) for c in bin(n_reg)[2:].zfill(n_incomplete)])[::-1]
        matching_mask = (region_sum == n_reg)
        if matching_mask.any():
            region_mask_list.append(matching_mask)
            region_indices_list.append(sorted(np.concatenate((incomplete_fs[fill_mask], complete_fs))))
    return region_indices_list, region_mask_list



###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
mother_nside = 32
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 10. * np.pi / 180.
edge_width = 10. * np.pi / 180.
# mask_name = 'herastrip'
# pixel_mask = (hpf.pix2ang(mother_nside, range(mother_npix), nest=True)[0] > np.pi/2 + np.pi/9) & (hpf.pix2ang(mother_nside, range(mother_npix), nest=True)[0] < np.pi/2 + np.pi/4.5)

mask_name = 'plane20deg'
pixel_mask = np.abs(hpf.pix2ang(mother_nside, range(mother_npix), nest=True)[0] - np.pi / 2) > np.pi / 9


step = .2
remove_cmb = True
show_plots = False

data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i.npz'%(mother_nside, smoothing_fwhm, edge_width, remove_cmb)

data_file = np.load(data_file_name)
freqs = data_file['freqs']
exclude_freqs = [freq for freq in freqs if freq > 40.]#[60.8, 70, 93.5]


idata = data_file['idata']
qudata = data_file['qdata'] + 1.j * data_file['udata']


bad_freqs_mask = np.isnan(qudata).all(axis=1)
freqs = freqs[~bad_freqs_mask]
nf = len(freqs)
idata = idata[~bad_freqs_mask]
qudata = qudata[~bad_freqs_mask]

#####quick digression

def mod(x, m, lower=None):
    if lower is None:
        lower = -m/2.
    return (x-lower)%m + lower
lambda2 = (0.3/freqs)**2

ploti = 0
ploti += 1
rm1 = mod(np.angle(qudata[0] / qudata[2]) / 2, np.pi) / lambda2[0]
rm2 = mod(np.angle(qudata[1] / qudata[2]) / 2, np.pi) / lambda2[1]
def colorcode(i):
    return abs(hpf.pix2ang(mother_nside, i, nest=True)[0] - np.pi/2)/(np.pi/2)
plt.scatter(rm2, rm1, marker='+', c=[(1. - colorcode(i), .5-2*(colorcode(i) - .5)**2, colorcode(i)) for i in range(mother_npix)], label='data')

maxrm = 500.
rm_step = .5
plt.scatter(mod(np.arange(-maxrm, maxrm, rm_step), np.pi/lambda2[1]), mod(np.arange(-maxrm, maxrm, rm_step), np.pi/lambda2[0]), c=[(0, .5+.5*abs(rm)/maxrm, 0) for rm in np.arange(-maxrm, maxrm,rm_step)], label='equal RM guideline')
plt.xlim([-120, 120])
plt.legend()
plt.ylabel('Inferred RM from 1.4GHz map')
plt.xlabel('Inferred RM from 2.3GHz map')
plt.show()

hpv.mollview(np.abs(rm1-rm2), nest=True, max=10, sub=(3,1,1), title='RM diff')
hpv.mollview(np.abs(qudata[0]) / idata[0], nest=True, sub=(3,1,2), min=0, max=.2, title='1.4GHz pol fraction')
hpv.mollview(np.abs(qudata[1]) / idata[1], nest=True, sub=(3,1,3), min=0, max=.2, title='2.3GHz pol fraction')
plt.show()
####end of digression


pixel_mask = pixel_mask & ~(np.isnan(qudata).any(axis=0))

matplotlib.rcParams.update({'font.size': 25})
for n, qud in enumerate(qudata):
    hpv.mollview(np.log10(np.abs(qud)), nest=True, sub=(4, (len(qudata)+1) / 2, n + 1), title="%.1fGHz"%freqs[n])
    hpv.mollview(np.angle(qud) / 2, nest=True, sub=(4, (len(qudata)+1) / 2, (len(qudata)+1) / 2 * 2 + n + 1), cmap=cm.hsv, title="%.1fGHz"%freqs[n])
plt.show()
exclude_freqs_mask = np.array([freq in exclude_freqs for freq in freqs])

freqs = freqs[~exclude_freqs_mask]
nf = len(freqs)
qudata = qudata[~exclude_freqs_mask]
qudata[:, ~pixel_mask] = 0

normalization = np.linalg.norm(qudata, axis=1)
D = qudata / normalization[:, None]



cov = np.einsum('ik,jk->ij', np.conjugate(D), D)

ev, ec = np.linalg.eigh(cov)
principal_maps = np.linalg.inv(np.transpose(np.conjugate(ec)).dot(ec)).dot(np.transpose(np.conjugate(ec)).dot(D))

for n in range(nf):
    hpv.mollview(np.log10(np.abs(principal_maps[n])), nest=True, sub=(4, nf, n + 1), title=ev[n])
    hpv.mollview(np.angle(principal_maps[n]) / 2, nest=True, sub=(4, nf, nf + n + 1), cmap=cm.hsv)
    plt.subplot(4, nf, nf * 2 + n + 1)
    plt.plot(np.abs(ec)[:, n])
    plt.ylim(-1, 1)
    plt.subplot(4, nf, nf * 3 + n + 1)
    plt.plot(np.angle(ec)[:, n] / 2)
plt.show()

for n_principal in range(1, min(6, 1+nf)):
    w_fn = np.copy(ec[:, -1:-n_principal-1:-1])
    x_ni = np.copy(principal_maps[-1:-n_principal-1:-1])
    errors = [np.linalg.norm(D - np.einsum('fn,ni->fi', w_fn, x_ni))]

    niter = 0
    while niter < 1000 and (len(errors) == 1 or errors[-2] - errors[-1] > 1.e-4 * errors[-2]):
        niter += 1
        print niter,
        sys.stdout.flush()

        new_x_ni = np.einsum('mn,fn,fi->mi', np.linalg.inv(np.einsum('fn,fm->nm', np.conjugate(w_fn), w_fn)), np.conjugate(w_fn), D)
        # new_x_ni = np.linalg.inv(np.einsum('fn,fm->nm', np.conjugate(w_fn), w_fn)).dot(np.einsum('fn,fi->ni', np.conjugate(w_fn), D))
        x_ni = step * new_x_ni + (1 - step) * x_ni

        new_w_fn = np.einsum('mn,ni,fi->fm', np.linalg.inv(np.einsum('ni,mi->nm', np.conjugate(x_ni), x_ni)), np.conjugate(x_ni), D)
        w_fn = step * new_w_fn + (1 - step) * w_fn

        re_norm = np.linalg.norm(w_fn, axis=0)
        w_fn /= re_norm[None, :]
        x_ni *= re_norm[:, None]

        D_error = D - np.einsum('fn,ni->fi', w_fn, x_ni)
        errors.append(np.linalg.norm(D_error))


    matplotlib.rcParams.update({'font.size': 6})

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for n in range(n_principal):
        hpv.mollview(np.log10(np.abs(x_ni[n])), nest=True, sub=(4, n_principal, n + 1), title=np.linalg.norm(x_ni[n]))
        hpv.mollview(np.angle(x_ni[n]) / 2, nest=True, sub=(4, n_principal, n_principal + n + 1), cmap=cm.hsv)
        plt.subplot(4, n_principal, n_principal * 2 + n + 1)
        plt.plot(np.log10(freqs), np.abs(w_fn)[:, n], 'g+')
        plt.plot(np.log10(freqs), np.abs(w_fn)[:, n], 'b-')
        plt.ylim(0, 2)
        plt.subplot(4, n_principal, n_principal * 3 + n + 1)
        phase_data = np.angle(w_fn)[:, n] / 2
        for i in range(1, len(phase_data)):
            phase_data[i] = (phase_data[i] - (phase_data[i-1] - np.pi / 2)) % np.pi + (phase_data[i-1] - np.pi / 2)
        phase_data -= int(np.mean(phase_data) / np.pi) * np.pi
        plt.plot(np.log10(freqs), phase_data, 'g+')
        plt.plot(np.log10(freqs), phase_data, 'b-')
        plt.ylim(-np.pi / 2 + np.mean(phase_data), np.pi / 2 + np.mean(phase_data))
    fig.savefig(data_file_name.replace('data_', 'plot_QU_%i_'%len(qudata)).replace('.npz', '_' + mask_name + '_principal_%i_step_%.2f_result_plot.png'%(n_principal, step)), dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for f in range(nf):
        hpv.mollview(np.log10(np.abs(D_error[f]/D[f])), nest=True, sub=(3, 5, f + 1), title=np.linalg.norm(D_error[f]), min=-2, max=-0)
    plt.subplot(3, 5, nf + 2)
    plt.plot(errors)
    fig.savefig(data_file_name.replace('data_', 'plot_QU_%i_'%len(qudata)).replace('.npz', '_' + mask_name + '_principal_%i_step_%.2f_error_plot.png'%(n_principal, step)), dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

