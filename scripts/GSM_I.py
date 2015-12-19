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
mother_nside = 512
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 3. * np.pi / 180.
edge_width = 1. * np.pi / 180.
remove_cmb = True

data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i.npz'%(mother_nside, smoothing_fwhm, edge_width, remove_cmb)


data_file = np.load(data_file_name)
freqs = data_file['freqs']
idata = data_file['idata']


##############################################
##############################################
####start I data processing method 1
##############################################
##############################################
####kick small coverage frequencies until there are regions that contain all frequencies
coverage_order = np.argsort(np.sum(~np.isnan(idata), axis=1))
low_coverage_data = {}
kick_candidate = 0#just increases from 0 to 1 to 2...
removed_mask = np.zeros(len(freqs), dtype='bool')
while np.sum(~np.isnan(idata[~removed_mask]).any(axis=0)) < mother_npix / 10:#when no region contains all frequencies
    kick_f = coverage_order[kick_candidate]
    # kick_freq = freqs[kick_f]
    # low_coverage_data[kick_freq] = new_mother.pop(kick_freq)
    removed_mask[kick_f] = True

    kick_candidate += 1



###put kicked data into add_on, and awkwardly change idata etc to smaller set, will merge back later. doing this awkward back and forth due to unfortunate coding order...
addon_freqs = freqs[removed_mask]
addon_idata = idata[removed_mask]

freqs = freqs[~removed_mask]
idata = idata[~removed_mask]

####seperate regions##
region_indices_list, region_mask_list = find_regions(np.isnan(idata))

region_illustration = np.empty(mother_npix)
for i, mask in enumerate(region_mask_list):
    region_illustration[mask] = len(region_mask_list) - i
hpv.mollview(region_illustration, nest=True)
plt.show()

####PCA to get rough estimates
####get eigen systems##
evs = []#np.zeros((len(region_mask_list), len(freqs)))
ecs = []#np.zeros((len(region_mask_list), len(freqs), len(freqs)))
i_covs = []
normalizations = []
# pix_normalization = np.zeros(mother_npix)
for i, (mask, fs) in enumerate(zip(region_mask_list, region_indices_list)[:10]):

    normalization = np.linalg.norm(idata[fs][:, mask], axis=1)
    normalized_data = idata[fs][:, mask] / normalization[:, None]

    # pix_normalization[mask] = np.linalg.norm(normalized_data, axis=0) / len(normalized_data)**.5

    i_cov = np.einsum('ik,jk->ij', normalized_data, normalized_data) / len(fs)
    ev, ec = np.linalg.eig(i_cov)

    #flip signs of eigenvectors: use first region, which i assume have all freqs, as template, and demand following eigenvectors to pick the sign that make it agree better with the template
    if i > 0:
        same_sign_norm = np.linalg.norm(ecs[0][fs, :len(fs)] - ec, axis=0)
        diff_sign_norm = np.linalg.norm(ecs[0][fs, :len(fs)] + ec, axis=0)
        ec *= (((same_sign_norm < diff_sign_norm) - .5) * 2)[None, :]


    evs.append(ev)
    ecs.append(ec)
    i_covs.append(i_cov)
    normalizations.append(normalization)

[plt.plot(ev) for ev in evs]
plt.show()
for i in range(5):
    plt.subplot(5, 1, i+1)
    for fs, ec in zip(region_indices_list, ecs):
        plot_data = np.copy(ec[:, i])
        plot_data *= plot_data.dot(ecs[0][fs, i]) / plot_data.dot(plot_data)
        # plt.plot(np.log10(freqs[fs]), plot_data)
        plt.plot(fs, plot_data)
        plt.ylim([-.7, .7])
plt.show()


#merge data (i know this is awkward)
all_freqs = sorted(np.concatenate((freqs, addon_freqs)))
addon_freqs_mask = np.array([freq in addon_freqs for freq in all_freqs])

all_idata = np.zeros((len(all_freqs), mother_npix))
all_idata[addon_freqs_mask] = addon_idata
all_idata[~addon_freqs_mask] = idata

all_region_indices_list, all_region_mask_list = find_regions(np.isnan(all_idata))

###iterate through different choices of n_principal
show_plots = False
step_size = 1.
for n_principal in range(6, 7):
    ###get starting point principal maps
    principal_matrix = ecs[0][:, :n_principal]
    principal_maps = np.zeros((n_principal, mother_npix))
    for i, (mask, fs) in enumerate(zip(region_mask_list, region_indices_list)):
        A = principal_matrix[fs]
        Ninv = np.eye(len(fs))#np.linalg.inv(i_covs[0][fs][:, fs])
        principal_maps[:, mask] = np.linalg.inv(A.transpose().dot(Ninv.dot(A))).dot(A.transpose().dot(Ninv.dot(idata[fs][:, mask] / normalizations[0][fs, None])))
    principal_fits = principal_matrix.dot(principal_maps)

    #use starting point principal maps to get weights for add-on maps
    addon_weights = np.zeros((len(addon_freqs), n_principal))
    for f, freq in enumerate(addon_freqs):
        addon_mask = ~np.isnan(addon_idata[f])
        A = np.transpose(principal_maps[:, addon_mask])
        addon_weights[f] = np.linalg.inv(np.transpose(A).dot(A)).dot(np.transpose(A).dot(addon_idata[f][addon_mask]))
    addon_normalization = np.linalg.norm(addon_weights.dot(principal_maps), axis=1) / np.linalg.norm(principal_fits[-1])
    addon_weights /= addon_normalization[:, None]

    ########################################
    ############Numerical Fitting###########
    ##########################################
    normalization = np.zeros(len(all_freqs))
    normalization[addon_freqs_mask] = addon_normalization
    normalization[~addon_freqs_mask] = normalizations[0]

    w_nf = np.zeros((n_principal, len(all_freqs)))
    w_nf[:, addon_freqs_mask] = np.transpose(addon_weights)
    w_nf[:, ~addon_freqs_mask] = np.transpose(ecs[0][:, :n_principal])

    xbar_ni = np.copy(principal_maps)

    x_fit = np.transpose(w_nf).dot(xbar_ni)

    errors = []
    current_error = 1#placeholder
    error = 1e12#placeholder
    niter = 0
    while abs((error - current_error)/current_error) > 1e-4 and niter <= 50:
        niter += 1
        print niter,
        sys.stdout.flush()
        current_error = error

        normalization *= np.linalg.norm(x_fit, axis=1) / np.mean(np.linalg.norm(x_fit, axis=1))
        x_fi = all_idata / normalization[:, None]

        #for w
        w_nf_ideal = np.zeros_like(w_nf)
        for f in range(len(all_freqs)):
            valid_mask = ~np.isnan(x_fi[f])
            A = np.transpose(xbar_ni)[valid_mask]
            b = x_fi[f, valid_mask]
            w_nf_ideal[:, f] = np.linalg.inv(np.einsum('ki,kj->ij', A, A)).dot(np.transpose(A).dot(b))
        w_nf = w_nf_ideal * step_size + w_nf * (1 - step_size)

        print '.',
        sys.stdout.flush()

        #for map:
        xbar_ni_ideal = np.zeros_like(xbar_ni)
        for i, (mask, fs) in enumerate(zip(all_region_mask_list, all_region_indices_list)):
            tm = time.time()
            A = np.transpose(w_nf)[fs]
            # print time.time() - tm,
            # tm = time.time(); sys.stdout.flush()
            Ninv = np.eye(len(fs))#np.linalg.inv(i_covs[0][fs][:, fs])
            # print time.time() - tm,
            # tm = time.time(); sys.stdout.flush()
            xbar_ni_ideal[:, mask] = np.linalg.inv(np.einsum('ki,kj->ij', A, Ninv.dot(A))).dot(np.transpose(A).dot(Ninv.dot(x_fi[:, mask][fs])))
            # print time.time() - tm,
            # tm = time.time(); sys.stdout.flush()
        xbar_ni = xbar_ni_ideal * step_size + xbar_ni * (1 - step_size)

        x_fit = np.transpose(w_nf).dot(xbar_ni)
        error = np.nansum((x_fit - x_fi).flatten()**2)
        errors.append(error)
    re_norm = np.linalg.norm(w_nf, axis=1)
    w_nf /= re_norm[:, None]
    xbar_ni *= re_norm[:, None]


    matplotlib.rcParams.update({'font.size': 6})

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    plt.subplot(3, 1, 1)
    plt.plot(errors)
    plt.subplot(3, 1, 2)
    plt.plot(np.nanmean((x_fit-x_fi)**2, axis=1))
    hpv.mollview(np.log10(np.nanmean((x_fit-x_fi)**2, axis=0)), nest=True, sub=(3,1,3))
    fig.savefig(data_file_name.replace('data_', 'plot_%i+%i_'%(len(idata), len(addon_idata))).replace('.npz', '_principal_%i_step_%.2f_error_plot.png'%(n_principal, step_size)), dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for i in range(n_principal):
        cmap = cm.gist_rainbow_r
        cmap.set_under('w')
        cmap.set_bad('gray')
        plot_data_lin = xbar_ni[i] * np.sign(xbar_ni[i, hpf.vec2pix(mother_nside, 1, 0, 0, nest=True)])
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin) / np.log(10.)

        hpv.mollview(plot_data, nest=True, sub=(3, n_principal, i + 1), min=np.percentile(plot_data, 2), max=np.percentile(plot_data, 98), cmap=cmap, title='%.3e'%np.linalg.norm(plot_data_lin))

        plt.subplot(3, n_principal, i + 1 + n_principal)
        plt.plot(np.log10(all_freqs), w_nf[i], 'r+')
        interp_x = np.arange(np.log10(all_freqs[0]), np.log10(all_freqs[-1]), .01)
        interp_y = si.interp1d(np.log10(all_freqs), w_nf[i], kind='slinear')(interp_x)
        plt.plot(interp_x, interp_y, 'b-')
        plt.ylim([-1.5, 1.5])

        plt.subplot(3, n_principal, i + 1 + 2 * n_principal)
        plt.plot(np.log10(all_freqs), np.log10(w_nf[i] * normalization), '+')
        plt.xlim([np.log10(all_freqs[0]), np.log10(all_freqs[-1])])
        plt.ylim(-5, 8)
    fig.savefig(data_file_name.replace('data_', 'plot_%i+%i_'%(len(idata), len(addon_idata))).replace('.npz', '_principal_%i_step_%.2f_result_plot.png'%(n_principal, step_size)), dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for f in range(len(all_freqs)):
        hpv.mollview(np.log10(np.abs(x_fit[f] - x_fi[f])), nest=True, title='%.3fGHz'%all_freqs[f], sub=(4, (len(all_freqs) - 1) / 4 + 1, f + 1), min=-5, max=-2)
    fig.savefig(data_file_name.replace('data_', 'plot_%i+%i_'%(len(idata), len(addon_idata))).replace('.npz', '_principal_%i_step_%.2f_error_plot2.png'%(n_principal, step_size)), dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    ##########################################
    ###put in interferometer measurements#####
    ##########################################


# ##############################################
# ##############################################
# ####start I data processing method 2
# ##############################################
# ##############################################
# D = all_idata / normalization[:, None]
# D[np.isnan(D)] = 0.
# DD = np.einsum('ik,jk->ij', D, D)
# CC = np.einsum('ik,jk->ij', np.array(D!=0, dtype='float32'), np.array(D!=0, dtype='float32'))
# ev2, ec2 = np.linalg.eigh(DD / CC)