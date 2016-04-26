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

def make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, n_principal, show_plots=True, vis_freqs=None, vis_ws=None, vis_norms=None):
    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    plt.gcf().set_size_inches(w=10, h=5)
    for i in range(n_principal):
        cmap = cm.gist_rainbow_r
        cmap.set_under('w')
        cmap.set_bad('gray')

        sign_flip = np.sign(w_nf[i, np.argmax(np.abs(w_nf[i]))])#np.sign(xbar_ni[i, hpf.vec2pix(mother_nside, 1, 0, 0, nest=True)])

        plot_data_lin = xbar_ni[i] * sign_flip
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin * 1 / (np.median(np.abs(plot_data_lin)))) #/ np.log(10.)

        hpv.mollview(plot_data, nest=True, sub=(3, n_principal, i + 1), min=np.percentile(plot_data, 2), max=np.percentile(plot_data, 98), cmap=cmap, title='%.3e'%la.norm(plot_data_lin))

        plt.subplot(3, n_principal, i + 1 + n_principal)
        plt.plot(np.log10(all_freqs), w_nf[i] * sign_flip, 'r+')
        interp_x = np.arange(np.log10(all_freqs[0]), np.log10(all_freqs[-1]), .01)
        interp_y = w_estimates[i](interp_x)
        plt.plot(interp_x, interp_y * sign_flip, 'b-')
        if vis_freqs is not None:
            plt.plot(np.log10(vis_freqs), vis_ws[:, i] * sign_flip, 'g+')
        plt.xlim([np.log10(all_freqs[0]) - .5, np.log10(all_freqs[-1]) + .5])
        plt.ylim([-1.5, 1.5])

        plt.subplot(3, n_principal, i + 1 + 2 * n_principal)
        plt.plot(np.log10(all_freqs), np.log10(w_nf[i] * normalization * sign_flip), '+')
        if vis_freqs is not None:
            plt.plot(np.log10(vis_freqs), np.log10(vis_ws[:, i] * vis_norms * sign_flip), 'g+')
        plt.xlim([np.log10(all_freqs[0]) - .5, np.log10(all_freqs[-1]) + .5])
        plt.ylim(-5, 8)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

# def get_estimated_idata(w_nf, normalization, xbar_ni, removed_freq):
#     w_estimates = [si.InterpolatedUnivariateSpline(np.log10(all_freqs), np.arcsinh(normalization * w_nf[i]), k=spline_k) for i in range(n_principal)]
#     logf = np.log10(removed_freq)
#     ws = [np.sinh(w_estimate(logf)) for w_estimate in w_estimates]
#     return xbar_ni.transpose().dot(ws)
def get_estimated_idata(w_nf, normalization, xbar_ni, removed_freq):
    w_estimates = [si.InterpolatedUnivariateSpline(np.log10(all_freqs), w_nf[i], k=spline_k) for i in range(n_principal)]
    norm_estimate = si.InterpolatedUnivariateSpline(np.log10(all_freqs), np.log10(normalization), k=spline_k)
    logf = np.log10(removed_freq)
    norm = 10.**(norm_estimate(logf))
    ws = [w_estimate(logf) for w_estimate in w_estimates]
    return norm * xbar_ni.transpose().dot(ws)
###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
mother_nside = 64
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 5 * np.pi / 180.#3.6 * np.pi / 180.#
edge_width = 3. * np.pi / 180.
remove_cmb = True
I_only = True
version = 3.0
spline_k = 1
step_size = 1.#.2
max_iter = 500

n_principal = 6
pick_map_mask = np.array([True] * 29)
error_weighting = 'remove_pt'#'none'#'inv_error'#'remove_pt'
rm_percentile = 99
n_trial = 2
common_coverage_fraction = 20

gsm1_mask = np.array([True] * 3 + [False,False,True,False,False] + [True] * 3 + [False,True,True,False,True,False,True] + [False]*10)
# n_principal = 5
# pick_map_mask = gsm1_mask
# error_weighting = 'remove_pt'#'none'#'inv_error'#'remove_pt'
# n_trial = 0
# common_coverage_fraction = 2000

data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(mother_nside, smoothing_fwhm, edge_width, remove_cmb, not I_only, version)
print data_file_name


data_file = np.load(data_file_name)
original_freqs = np.concatenate((data_file['freqs'][:10], data_file['freqs'][11:-2]))[pick_map_mask]
original_idata = np.concatenate((data_file['idata'][:10], data_file['idata'][11:-2]))[pick_map_mask]

###############################
#give data correct units
kB = 1.38065e-23
c = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB
def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / c)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * c / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / c)**2 * kB
    return K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

TRJ_mask = original_freqs < 20
TCMB_mask = (original_freqs >= 20) & (original_freqs < 500)
original_idata[TRJ_mask] = K_RJ2MJysr(1., original_freqs[TRJ_mask] * 1e9)[:, None] * original_idata[TRJ_mask]
original_idata[TCMB_mask] = K_CMB2MJysr(1., original_freqs[TCMB_mask] * 1e9)[:, None] * original_idata[TCMB_mask]

###############################
##start loop
estimated_data = []
latitude_masks = [np.ones(mother_npix, dtype=bool)]
n_latitude_regions = 2
gal_lat = np.abs(np.pi / 2 - hpf.pix2ang(mother_nside, range(mother_npix), nest=True)[0])
lat_width = np.pi / 2 / n_latitude_regions
for n_lat in range(n_latitude_regions):
    latitude_masks.append((gal_lat >= n_lat * lat_width) & (gal_lat < (n_lat + 1) * lat_width))
estimation_errors = [[] for i in range(1 + n_latitude_regions)]
estimation_renorm_errors = [[] for i in range(1 + n_latitude_regions)]
pseudo_estimation_errors = [[] for i in range(1 + n_latitude_regions)]
for remove_f in range(-1, len(original_idata)):
    removed_idata = np.copy(original_idata[remove_f])
    removed_freq = np.copy(original_freqs[remove_f])
    idata = original_idata[np.array([f for f in range(len(original_idata)) if f != remove_f])]
    freqs = original_freqs[np.array([f for f in range(len(original_freqs)) if f != remove_f])]
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
    while np.sum(~np.isnan(idata[~removed_mask]).any(axis=0)) < mother_npix / common_coverage_fraction:#when no region contains all frequencies
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


    ####PCA to get rough estimates
    ####get eigen systems##
    mask = region_mask_list[0]
    fs = region_indices_list[0]
    normalization0 = la.norm(idata[np.ix_(fs, mask)], axis=1)
    normalized_data = idata[np.ix_(fs, mask)] / normalization0[:, None]
    i_cov0 = np.einsum('ik,jk->ij', normalized_data, normalized_data) / len(fs)
    ev0, ec0 = la.eig(i_cov0)



    #merge data (i know this is awkward)
    all_freqs = sorted(np.concatenate((freqs, addon_freqs)))
    addon_freqs_mask = np.array([freq in addon_freqs for freq in all_freqs])

    all_idata = np.zeros((len(all_freqs), mother_npix))
    all_idata[addon_freqs_mask] = addon_idata
    all_idata[~addon_freqs_mask] = idata

    all_region_indices_list, all_region_mask_list = find_regions(np.isnan(all_idata))


    ######################################################
    ###iterate through different choices of n_principal###
    #######################################################
    show_plots = False

    ###get starting point principal maps
    principal_matrix = ec0[:, :n_principal]
    principal_maps = np.zeros((n_principal, mother_npix))
    for i, (mask, fs) in enumerate(zip(region_mask_list, region_indices_list)):
        A = principal_matrix[fs]
        Ninv = np.eye(len(fs))
        #Ninv = la.inv(i_covs[0][fs][:, fs])
        principal_maps[:, mask] = la.inv(A.transpose().dot(Ninv.dot(A))).dot(A.transpose().dot(Ninv.dot(idata[np.ix_(fs, mask)] / normalization0[fs, None])))
    principal_fits = principal_matrix.dot(principal_maps)

    #use starting point principal maps to get weights for add-on maps
    addon_weights = np.zeros((len(addon_freqs), n_principal))
    for f, freq in enumerate(addon_freqs):
        addon_mask = ~np.isnan(addon_idata[f])
        A = np.transpose(principal_maps[:, addon_mask])
        addon_weights[f] = la.inv(np.transpose(A).dot(A)).dot(np.transpose(A).dot(addon_idata[f, addon_mask]))
    addon_normalization = la.norm(addon_weights.dot(principal_maps), axis=1) / la.norm(principal_fits[-1])
    addon_weights /= addon_normalization[:, None]

    ########################################
    ############Numerical Fitting###########
    ##########################################
    normalization = np.zeros(len(all_freqs))
    normalization[addon_freqs_mask] = addon_normalization
    normalization[~addon_freqs_mask] = normalization0

    w_nf = np.zeros((n_principal, len(all_freqs)))
    w_nf[:, addon_freqs_mask] = np.transpose(addon_weights)
    w_nf[:, ~addon_freqs_mask] = np.transpose(ec0[:, :n_principal])

    xbar_ni = np.copy(principal_maps)

    x_fit = np.transpose(w_nf).dot(xbar_ni)
    point_source_mask = np.zeros(mother_npix, dtype=bool)
    x_fi = all_idata / normalization[:, None]
    for trial in range(n_trial):#when trial goes to 1 we remove worst fitting pixels to remove point sources such as cyg and cas
        if trial == 1:
            previous_error_map = np.nanmean((x_fit-x_fi)**2, axis=0)
            if error_weighting == 'remove_pt':
                point_source_mask = previous_error_map >= np.percentile(previous_error_map, rm_percentile)
            elif error_weighting == 'inv_error':
                x_fi = x_fi / previous_error_map**.5
                x_fit = x_fit / previous_error_map**.5
                xbar_ni = xbar_ni / previous_error_map**.5
            elif error_weighting == 'none':
                break
        errors = []
        current_error = 1#placeholder
        error = 1e12#placeholder
        niter = 0
        while abs((error - current_error)/current_error) > 1e-4 and niter <= max_iter:
            niter += 1
            print niter,
            sys.stdout.flush()
            current_error = error

            normalization *= la.norm(x_fit[:, ~point_source_mask], axis=1) / np.mean(la.norm(x_fit[:, ~point_source_mask], axis=1))
            x_fi = all_idata / normalization[:, None]

            #for w
            tm = time.time()
            w_nf_ideal = np.zeros_like(w_nf)
            for f in range(len(all_freqs)):

                if f > 0 and (np.isnan(x_fi[f]) == np.isnan(x_fi[f-1])).all():
                    pass
                else:#only recompute A and AtAi if mask changes
                    valid_mask = ~(np.isnan(x_fi[f]) | point_source_mask)
                    A = np.transpose(xbar_ni)[valid_mask]
                    AtA = np.einsum('ki,kj->ij', A, A)
                    AtAi = la.inv(AtA)
                b = x_fi[f, valid_mask]
                w_nf_ideal[:, f] = AtAi.dot(np.transpose(A).dot(b))
            w_nf = w_nf_ideal * step_size + w_nf * (1 - step_size)

            print '.',#time.time() - tm,
            sys.stdout.flush()

            #for map:
            xbar_ni_ideal = np.zeros_like(xbar_ni)
            for i, (mask, fs) in enumerate(zip(all_region_mask_list, all_region_indices_list)):
                tm = time.time()
                A = np.transpose(w_nf)[fs]
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()
                # Ninv = np.eye(len(fs))
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()
                # xbar_ni_ideal[:, mask] = la.inv(np.transpose(A).dot(Ninv.dot(A))).dot(np.transpose(A).dot(Ninv)).dot(x_fi[np.ix_(fs, mask)])

                xbar_ni_ideal[:, mask] = la.inv(np.einsum('ki,kj->ij', A, A)).dot(np.transpose(A).dot(x_fi[np.ix_(fs, mask)]))
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()
            xbar_ni = xbar_ni_ideal * step_size + xbar_ni * (1 - step_size)

            x_fit = np.transpose(w_nf).dot(xbar_ni)
            error = np.nansum((x_fit - x_fi)[:, ~point_source_mask].flatten()**2)
            errors.append(error)
        if trial == 1 and error_weighting == 'inv_error':
            x_fi = x_fi * previous_error_map**.5
            x_fit = x_fit * previous_error_map**.5
            xbar_ni = xbar_ni * previous_error_map**.5
    ev2, ec2 = la.eigh(np.einsum('ni,mi->nm', xbar_ni, xbar_ni))
    xbar_ni = la.inv(ec2).dot(xbar_ni)
    w_nf = np.transpose(ec2).dot(w_nf)

    re_norm = la.norm(w_nf, axis=1)
    w_nf /= re_norm[:, None]
    xbar_ni *= re_norm[:, None]
    # print la.eigh(w_nf.dot(np.transpose(w_nf)))[0]

    ##end of component separatin. now calculate estimates
    norm_estimate = si.InterpolatedUnivariateSpline(np.log10(all_freqs), np.log10(normalization), k=spline_k)


    w_estimates = [si.InterpolatedUnivariateSpline(np.log10(all_freqs), w_nf[i], k=spline_k) for i in range(n_principal)]

    if remove_f >= 0:
        estimated_removal = get_estimated_idata(w_nf, normalization, xbar_ni, removed_freq)

        for l, lat_mask in enumerate(latitude_masks):
            mask = lat_mask&~(point_source_mask|np.isnan(removed_idata)|np.isnan(estimated_removal))
            renorm = np.sum((estimated_removal * removed_idata)[mask]) / np.sum(estimated_removal * estimated_removal * mask)
            estimated_removal2 = estimated_removal * renorm


            estimation_error = la.norm((removed_idata - estimated_removal)[mask]) / la.norm(removed_idata[mask])
            estimation_renorm_error = la.norm((removed_idata - estimated_removal2)[mask]) / la.norm(removed_idata[mask])

            estimation_errors[l].append(estimation_error)
            estimation_renorm_errors[l].append(estimation_renorm_error)
        # estimated_data.append(estimated_removal)
        print removed_freq, estimation_error, estimation_renorm_error, renorm
    else:
        make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, n_principal)
        systematic_errors = [(np.nanmean((x_fit-x_fi)[:, lat_mask&~point_source_mask]**2, axis=1) / np.nanmean(x_fi[:, lat_mask&~point_source_mask]**2, axis=1))**.5 for lat_mask in latitude_masks]

        for rm_f in range(len(original_freqs)):
            xbar_ni_rm1 = np.zeros_like(xbar_ni)
            for i, (mask, fs) in enumerate(zip(all_region_mask_list, all_region_indices_list)):
                fs = np.array([tmpf for tmpf in fs if tmpf != rm_f])
                A = np.transpose(w_nf)[fs]
                xbar_ni_rm1[:, mask] = la.inv(np.einsum('ki,kj->ij', A, A)).dot(np.transpose(A).dot(x_fi[np.ix_(fs, mask)]))
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()

            removed_idata = original_idata[rm_f]
            estimated_removal = get_estimated_idata(w_nf, normalization, xbar_ni_rm1, original_freqs[rm_f])
            for l, lat_mask in enumerate(latitude_masks):
                estimation_error = (np.nansum((removed_idata - estimated_removal)[lat_mask&~point_source_mask]**2) / np.nansum(removed_idata[lat_mask&~point_source_mask]**2))**.5
                pseudo_estimation_errors[l].append(estimation_error)

        import matplotlib
        matplotlib.rcParams.update({'font.size': 25})
        for plot_i, logfreq in enumerate(np.arange(-1.7, 4., .03)):
            if logfreq < 0:
                title = '%.1f MHz'%(1e3 * 10**logfreq)
            elif logfreq < 3:
                title = '%.1f GHz'%(10**logfreq)
            else:
                title = '%.1f THz'%(1e-3 * 10**logfreq)
            pltdata = np.arcsinh(np.transpose(xbar_ni).dot([w_estimates[i](logfreq) for i in range(n_principal)]) / np.percentile(np.abs(np.transpose(xbar_ni).dot([w_estimates[i](logfreq) for i in range(n_principal)])), .2))
            hpv.mollview(pltdata, nest=True, title=title, min=np.percentile(pltdata, 1), max=np.percentile(pltdata, 99), cbar=False)
            plt.savefig('/home/omniscope/gif_dir/%04i.png'%plot_i)
            plt.clf()

def cplot(x, y, fmt, mksize, fillstyle='none'):
    for a, b, msize in zip(x, y, mksize):
        plt.plot(a, b, fmt, markersize=msize, fillstyle=fillstyle)

for i in range(1 + n_latitude_regions):
    plt.subplot(1, 1 + n_latitude_regions, i+1)
    
    mksize = (400. * np.sum(~np.isnan(original_idata), axis=-1) / mother_npix)**.5
    plt.plot(np.log10(original_freqs), systematic_errors[i], 'b-', label='systematic error')
    cplot(np.log10(original_freqs), systematic_errors[i], 'bo', mksize)
    plt.plot(np.log10(original_freqs), pseudo_estimation_errors[i], 'g-', label='pseudo estimation error')
    cplot(np.log10(original_freqs), pseudo_estimation_errors[i], 'go', mksize)
    plt.plot(np.log10(original_freqs), estimation_renorm_errors[i], 'm--', label='renormed estimation errors')
    cplot(np.log10(original_freqs), estimation_renorm_errors[i], 'mo', mksize)
    plt.plot(np.log10(original_freqs), estimation_errors[i], 'k--', label='estimation errors')
    cplot(np.log10(original_freqs), estimation_errors[i], 'ko', mksize)
    plt.ylim([0, .3])
    plt.xlim([-4, 4])
    plt.title((['all'] + (np.arange(n_latitude_regions) * np.pi / 2 / n_latitude_regions).tolist())[i])
    plt.legend()
plt.show()

gsm1_errs=np.array([
0.091,
0.080,
0.094,
0.084,
0.187,
0.147,
0.111,
0.062,
0.091,
0.270,
0.766,
0.099,
0.082,
0.102,
0.088,
0.133,
0.160,
0.070,
0.022,
0.052,
0.067,
0.164,
0.094,
0.136,
0.095,
0.072,
0.144,
0.158,
0.083,
0.013,
0.042,
0.071,
0.095,
0.148,
0.210,
0.108,
0.125,
0.180,
0.165,
0.073,
0.012,
0.039,
0.082,
0.124,
0.111,
0.306,
0.158,
0.170,
0.157,
0.170,
0.062,
0.011,
0.034,
0.079,
0.136,
np.nan,
0.451,
0.202,
0.190,
0.129,
0.167,
0.050,
0.011,
0.028,
0.067,
0.135,
0.098,
0.144,
0.109,
0.115,
0.144,
0.155,
0.062,
0.013,
0.032,
0.068,
0.121

]).reshape((7, 11))# 6 sky regions clean to dirty, then overall

i = 0
mksize = (400. * np.sum(~np.isnan(original_idata), axis=-1) / mother_npix)**.5
lw = 3

# plt.plot(original_freqs, estimation_errors[i], 'k--', label='estimation errors')
# cplot(original_freqs, estimation_errors[i], 'ko', mksize)
plt.plot(original_freqs, systematic_errors[i], 'b-', label='systematic error', linewidth=lw)
cplot(original_freqs, systematic_errors[i], 'bo', mksize)
plt.plot(original_freqs, pseudo_estimation_errors[i], 'g-', label='pseudo estimation error', linewidth=lw)
cplot(original_freqs, pseudo_estimation_errors[i], 'go', mksize)
plt.plot(original_freqs, estimation_renorm_errors[i], 'm-', label='renormed estimation errors', linewidth=lw)
cplot(original_freqs, estimation_renorm_errors[i], 'mo', mksize)

plt.plot(original_freqs[gsm1_mask], gsm1_errs[-1].transpose(), 'g--', linewidth=lw)
plt.ylim([0,.3])
plt.xlim([.5e-3, 8e3])
plt.show()

#
# hpv.mollview(np.log10(removed_idata * ~point_source_mask), nest=True, sub=(2,1,1))#, min=1, max=3)
# hpv.mollview(np.log10(estimated_removal * ~point_source_mask), nest=True, sub=(2,1,2))#, min=1, max=3)
# plt.show()
# make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, n_principal)

# w_der = w_nf[:, 1:] - w_nf[:, :-1]
# w_der2 = w_der[:, 1:] - w_der[:, :-1]
# _, M = la.eigh(w_der2.dot(w_der2.transpose()))
# make_result_plot(all_freqs, M.transpose().dot(w_nf), M.transpose().dot(xbar_ni), w_estimates, normalization, n_principal)