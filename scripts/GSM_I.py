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

def make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, tag, n_principal, show_plots, vis_freqs=None, vis_ws=None, vis_norms=None):
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
    fig.savefig(plot_filename_base + tag + '.png', dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()
###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
mother_nside = 64
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 3. * np.pi / 180.
edge_width = 3. * np.pi / 180.
remove_cmb = True

n_principal_range = range(6, 7)

include_visibility = False
vis_Qs = ["q0C_*_abscal", "q1AL_*_abscal", "q2C_*_abscal", "q3AL_*_abscal", "q4AL_*_abscal"]  # L stands for lenient in flagging
datatag = '_2016_01_20_avg'
vartag = '_2016_01_20_avg'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
vis_tags = []
for vis_Q in vis_Qs:
    filenames = glob.glob(datadir + vis_Q + '_xx*' + datatag)
    vis_tags = vis_tags + [os.path.basename(fn).split('_xx')[0] for fn in filenames]

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
hpv.mollview(region_illustration, nest=True, title="Frequency Coverage Map")
plt.show()

####PCA to get rough estimates
####get eigen systems##
evs = []#np.zeros((len(region_mask_list), len(freqs)))
ecs = []#np.zeros((len(region_mask_list), len(freqs), len(freqs)))
i_covs = []
normalizations = []
# pix_normalization = np.zeros(mother_npix)
for i, (mask, fs) in enumerate(zip(region_mask_list, region_indices_list)[:10]):

    normalization = la.norm(idata[np.ix_(fs, mask)], axis=1)
    normalized_data = idata[np.ix_(fs, mask)] / normalization[:, None]

    # pix_normalization[mask] = la.norm(normalized_data, axis=0) / len(normalized_data)**.5

    i_cov = np.einsum('ik,jk->ij', normalized_data, normalized_data) / len(fs)
    ev, ec = la.eig(i_cov)

    #flip signs of eigenvectors: use first region, which i assume have all freqs, as template, and demand following eigenvectors to pick the sign that make it agree better with the template
    if i > 0:
        same_sign_norm = la.norm(ecs[0][fs, :len(fs)] - ec, axis=0)
        diff_sign_norm = la.norm(ecs[0][fs, :len(fs)] + ec, axis=0)
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


######################################################
###iterate through different choices of n_principal###
#######################################################
show_plots = False
step_size = 1.
for n_principal in n_principal_range:
    ###get starting point principal maps
    principal_matrix = ecs[0][:, :n_principal]
    principal_maps = np.zeros((n_principal, mother_npix))
    for i, (mask, fs) in enumerate(zip(region_mask_list, region_indices_list)):
        A = principal_matrix[fs]
        Ninv = np.eye(len(fs))#la.inv(i_covs[0][fs][:, fs])
        principal_maps[:, mask] = la.inv(A.transpose().dot(Ninv.dot(A))).dot(A.transpose().dot(Ninv.dot(idata[np.ix_(fs, mask)] / normalizations[0][fs, None])))
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
    normalization[~addon_freqs_mask] = normalizations[0]

    w_nf = np.zeros((n_principal, len(all_freqs)))
    w_nf[:, addon_freqs_mask] = np.transpose(addon_weights)
    w_nf[:, ~addon_freqs_mask] = np.transpose(ecs[0][:, :n_principal])

    xbar_ni = np.copy(principal_maps)

    x_fit = np.transpose(w_nf).dot(xbar_ni)

    for trial in range(1):#when trial goes to 1 we remove worst fitting pixels to remove point sources such as cyg and cas
        if trial == 1:
            previous_error_map = np.nanmean((x_fit-x_fi)**2, axis=0)
            all_idata[:, previous_error_map >= np.percentile(previous_error_map, 95)] = 0
            xbar_ni[:, previous_error_map >= np.percentile(previous_error_map, 95)] = 0
        errors = []
        current_error = 1#placeholder
        error = 1e12#placeholder
        niter = 0
        while abs((error - current_error)/current_error) > 1e-4 and niter <= 50:
            niter += 1
            print niter,
            sys.stdout.flush()
            current_error = error

            normalization *= la.norm(x_fit, axis=1) / np.mean(la.norm(x_fit, axis=1))
            x_fi = all_idata / normalization[:, None]

            #for w
            w_nf_ideal = np.zeros_like(w_nf)
            for f in range(len(all_freqs)):
                valid_mask = ~np.isnan(x_fi[f])
                A = np.transpose(xbar_ni)[valid_mask]
                b = x_fi[f, valid_mask]
                w_nf_ideal[:, f] = la.inv(np.einsum('ki,kj->ij', A, A)).dot(np.transpose(A).dot(b))
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
                Ninv = np.eye(len(fs))#la.inv(i_covs[0][fs][:, fs])
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()
                xbar_ni_ideal[:, mask] = la.inv(np.transpose(A).dot(Ninv.dot(A))).dot(np.transpose(A).dot(Ninv)).dot(x_fi[np.ix_(fs, mask)])
                # print time.time() - tm,
                # tm = time.time(); sys.stdout.flush()
            xbar_ni = xbar_ni_ideal * step_size + xbar_ni * (1 - step_size)

            x_fit = np.transpose(w_nf).dot(xbar_ni)
            error = np.nansum((x_fit - x_fi).flatten()**2)
            errors.append(error)

    ev2, ec2 = la.eigh(np.einsum('ni,mi->nm', xbar_ni, xbar_ni))
    xbar_ni = la.inv(ec2).dot(xbar_ni)
    w_nf = np.transpose(ec2).dot(w_nf)

    re_norm = la.norm(w_nf, axis=1)
    w_nf /= re_norm[:, None]
    xbar_ni *= re_norm[:, None]
    # print la.eigh(w_nf.dot(np.transpose(w_nf)))[0]

    w_estimates = [si.interp1d(np.log10(all_freqs), w_nf[i], kind='slinear', assume_sorted=False) for i in range(n_principal)]

    result_filename = data_file_name.replace('data_', 'result_%i+%i_'%(len(idata), len(addon_idata))).replace('.npz', '_principal_%i_step_%.2f'%(n_principal, step_size))
    np.savez(result_filename, w_nf=w_nf, x_ni=xbar_ni, normalization=normalization)

    ##################################################
    #####play with changing basis WMinv MX####################
    ##################################################

    ###########manually isolate components###############
    if n_principal == 6:
        M = np.eye(n_principal)

        M1inv = np.eye(n_principal)
        manual_spike_freq_ranges = [[1, 2], [10, 1000], [0, 10], [50, 1e9], [500, 1e9]]
        manual_spike_ranges = [np.arange(len(all_freqs))[(np.array(all_freqs) >= freq_range[0]) & (np.array(all_freqs) <= freq_range[1])] for freq_range in manual_spike_freq_ranges]
        # manual_spike_ranges = [np.arange(4, 8), np.arange(8, 22), np.arange(0, 8), np.arange(13, 29), np.arange(22, 29)]
        manual_spike_principals = np.array([1, 0, 4, 2, 3]) - 6
        for spike_fs, spike_principal in zip(manual_spike_ranges, manual_spike_principals):
            if spike_principal >= -n_principal:
                non_spike_fs = np.array([i for i in range(len(all_freqs)) if i not in spike_fs])
                non_spike_principals = np.array([i for i in range(-n_principal, 0) if i != spike_principal])
                spike_A = np.transpose(np.delete(w_nf[:, non_spike_fs], spike_principal, axis=0))
                M1inv[non_spike_principals, spike_principal] = -la.inv(np.transpose(spike_A).dot(spike_A)).dot(np.transpose(spike_A).dot(w_nf[spike_principal, non_spike_fs]))
        M1 = la.inv(M1inv)
        w_nf1 = np.transpose(M1inv).dot(w_nf)
        M = M1.dot(M)
        print la.cond(M)


        Minv2 = np.eye(n_principal)
        for i in range(n_principal):
            if i not in manual_spike_principals:
                spike_principal = i
                spike_fs = []
                non_spike_fs = np.arange(len(all_freqs))
                non_spike_principals = np.array([i for i in range(n_principal) if i != spike_principal])
                spike_A = np.transpose(w_nf1[np.ix_(manual_spike_principals, non_spike_fs)])
                Minv2[manual_spike_principals, spike_principal] = -la.inv(np.transpose(spike_A).dot(spike_A)).dot(np.transpose(spike_A).dot(w_nf1[spike_principal, non_spike_fs]))
        M2 = la.inv(Minv2)
        xbar_ni2 = M2.dot(M1.dot(xbar_ni))
        M = M2.dot(M)
        print la.cond(M)


        M3 = np.eye(n_principal)
        if n_principal >= 6:
            galactic_plane_mask = np.abs(hpf.pix2ang(mother_nside, range(mother_npix), nest=True)[0] - np.pi / 2) < np.pi/36
            cmb_A = np.transpose(xbar_ni2[np.ix_(np.arange(-n_principal, 0) != -6, galactic_plane_mask)])
            regulation = np.ones(n_principal - 1)
            regulation[-5] = 100
            regulation[-2] = 100
            M3[-6, np.arange(-n_principal, 0) != -6] = -la.inv(np.transpose(cmb_A).dot(cmb_A) + np.diag(regulation)).dot(np.transpose(cmb_A).dot(xbar_ni2[0, galactic_plane_mask]))
        Minv3 = la.inv(M3)
        M = M3.dot(M)
        w_nf3 = np.transpose(la.inv(M)).dot(w_nf)
        xbar_ni3 = M3.dot(xbar_ni2)
        print la.cond(M)


        # for n in range(20):
        #     M4 = np.eye(n_principal)
        #     for p in [-4, -1]:
        #         negative_mask = xbar_ni_final[p] < 0
        #         if negative_mask.any():
        #             negative_A = np.transpose(xbar_ni3[np.ix_(np.arange(-n_principal, 0) != p, negative_mask)])
        #             regulation = np.zeros(n_principal - 1)
        #             M4[p, np.arange(-n_principal, 0) != p] = -la.inv(np.transpose(negative_A).dot(negative_A) + np.diag(regulation)).dot(np.transpose(negative_A).dot(xbar_ni3[p, negative_mask]))
        #     xbar_ni3 = M4.dot(xbar_ni3)
        #     M = M4.dot(M)
        #     print la.cond(M), la.norm(xbar_ni3[-4][xbar_ni3[-4] < 0]), la.norm(xbar_ni3[-1][xbar_ni3[-1] < 0])


        Minv = la.inv(M)
        w_nf_final = np.transpose(Minv).dot(w_nf)
        xbar_ni_final = M.dot(xbar_ni)
        final_renorm = la.norm(w_nf_final, axis=-1)
        w_nf_final /= final_renorm[:, None]
        xbar_ni_final *= final_renorm[:, None]
        w_estimates_final = [si.interp1d(np.log10(all_freqs), w_nf_final[i], kind='slinear', assume_sorted=False) for i in range(n_principal)]
    #
    # ####auto-identifying ranges######
    # eigen_thresh = 0.05
    # f_starts = []
    # for i in range(len(all_freqs) - 1, 0, -1):
    #     non_spike_fs = np.arange(i)
    #     ev_ascending, _ = la.eigh(w_nf[:, non_spike_fs].dot(np.transpose(w_nf[:, non_spike_fs])))
    #     if ev_ascending[len(f_starts)] / ev_ascending[-1] < eigen_thresh and ev_ascending[len(f_starts) + 1] / ev_ascending[-1] >= eigen_thresh:
    #         f_starts.append(i)
    #
    # f_ends = np.zeros_like(f_starts)
    # n_degens = np.zeros_like(f_starts)
    # for i, f_start in enumerate(f_starts):
    #     for f_end in range(f_start + 1, len(all_freqs) + 1):
    #         non_spike_fs = range(f_start) + range(f_end, len(all_freqs))
    #         ev_ascending, _ = la.eigh(w_nf[:, non_spike_fs].dot(np.transpose(w_nf[:, non_spike_fs])))
    #         if ev_ascending[0] / ev_ascending[-1] < eigen_thresh:
    #             f_ends[i] = f_end
    #             n_degens[i] = np.sum(ev_ascending / ev_ascending[-1] < eigen_thresh)
    #             break
    #
    # # for f_end in range(1, len(all_freqs)):
    # #     non_spike_fs = range(f_end, len(all_freqs))
    # #     ev_ascending, _ = la.eigh(w_nf[:, non_spike_fs].dot(np.transpose(w_nf[:, non_spike_fs])))
    # #     if ev_ascending[np.sum(np.array(f_ends) <= f_end)] / ev_ascending[-1] < eigen_thresh:
    # #         f_starts.append(0)
    # #         f_ends = np.concatenate((f_ends, [f_end]))
    # #         n_degens = np.concatenate((n_degens, [np.sum(ev_ascending / ev_ascending[-1] < eigen_thresh)]))
    # #         break
    # M6inv = np.eye(n_principal)
    # for spike_principal, (f_start, f_end) in enumerate(zip(f_starts, f_ends)):
    #     non_spike_fs = range(f_start) + range(f_end, len(all_freqs))
    #     print np.argmax(la.norm(w_nf[:, f_start:f_end])), np.argmin(la.norm(w_nf[:, non_spike_fs]))
    #     non_spike_principals = np.array([i for i in range(n_principal) if i != spike_principal])
    #     spike_A = np.transpose(np.delete(w_nf[:, non_spike_fs], spike_principal, axis=0))
    #     M6inv[non_spike_principals, spike_principal] = -la.inv(np.transpose(spike_A).dot(spike_A)).dot(np.transpose(spike_A).dot(w_nf[spike_principal, non_spike_fs]))
    #
    # M6 = la.inv(M6inv)
    # xbar_ni6 = M6.dot(xbar_ni)
    # w_nf6 = np.transpose(M6inv).dot(w_nf)
    # w_estimates6 = [si.interp1d(np.log10(all_freqs), w_nf6[i], kind='slinear', assume_sorted=False) for i in range(n_principal)]

    ##################################################
    #####make plots####################
    ##################################################
    matplotlib.rcParams.update({'font.size': 5})
    plot_filename_base = result_filename.replace('result', 'plot')

    make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, '_result_plot', n_principal, show_plots)
    if n_principal == 6:
        make_result_plot(all_freqs, w_nf_final, xbar_ni_final, w_estimates_final, normalization, '_result_plot_M5', n_principal, show_plots)

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    plt.subplot(3, 1, 1)
    plt.plot(errors)
    plt.subplot(3, 1, 2)
    plt.plot(np.nanmean((x_fit-x_fi)**2, axis=1))
    hpv.mollview(np.log10(np.nanmean((x_fit-x_fi)**2, axis=0)), nest=True, sub=(3,1,3))
    fig.savefig(plot_filename_base + '_error_plot.png', dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for f in range(len(all_freqs)):
        hpv.mollview(np.log10(np.abs(x_fit[f] - x_fi[f])), nest=True, title='%.3fGHz'%all_freqs[f], sub=(4, (len(all_freqs) - 1) / 4 + 1, f + 1), min=-5, max=-2)
    fig.savefig(plot_filename_base + '_error_plot2.png', dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

    # ##########################################
    # ###put in interferometer measurements#####
    # ##########################################
    if include_visibility and n_principal == 6:
        ###prepare big matrices stuff for interferometer data###
        # tag = "q3_abscalibrated"
        # datatag = '_seccasa_polcor.rad'
        # vartag = '_seccasa_polcor'
        # datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
        # nt = 440
        # nf = 1
        # nUBL = 75
        # nside = mother_nside
        # bnside = 8

        vis_freqs = np.zeros(len(vis_tags))
        vis_norms = np.zeros(len(vis_tags))
        vis_norms_final = np.zeros(len(vis_tags))
        vis_ws = np.zeros((len(vis_tags), n_principal)) + np.nan
        vis_ws_final = np.zeros((len(vis_tags), n_principal)) + np.nan
        for tag_i, tag in enumerate(vis_tags):
            print tag
            sys.stdout.flush()
            # nt = {"q3A_abscal": 253, "q3AL_abscal": 368}[tag]
            nf = 1
            # nUBL = 78

            nside = mother_nside
            bnside = 16

            C = .299792458
            kB = 1.3806488 * 1.e-23

            A_vis = {}
            vis_data = {}
            Ni = {}
            data_shape = {}
            ubls = {}
            ubl_sort = {}
            tmasks = {}
            for p in ['x', 'y']:
                pol = p+p
                data_filename = glob.glob(datadir + tag + '_%s%s_*_*'%(p, p) + datatag)[0]
                nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('%s%s_'%(p, p))[-1]
                nt = int(nt_nUBL.split('_')[0])
                nUBL = int(nt_nUBL.split('_')[1])

                #tf file
                tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
                tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
                tlist = np.real(tflist[:, 0])
                flist = np.imag(tflist[0, :])    #will be later assuming flist only has 1 element
                vis_freq = flist[0] / 1e3
                print vis_freq
                vis_freqs[tag_i] = vis_freq
                ubl_length_thresh = 1.4 * 0.3 / vis_freq * min(mother_nside, 1. / (smoothing_fwhm + 1e-12)) / 10

                #tf mask file, 0 means flagged bad data
                try:
                    tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
                    tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
                    tmasks[p] = np.array(tfmlist[:,0].astype('bool'))
                    #print tmasks[p]
                except:
                    print "No mask file found"
                    tmasks[p] = np.ones_like(tlist).astype(bool)
                #print vis_freq, tlist

                #ubl file
                ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
                ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
                ubl_mask = la.norm(ubls[p], axis=1) <= ubl_length_thresh
                print "%i UBLs to include"%len(ubls[p])


                #read A matrix computed by GSM_make_A_matrix.py
                A_filename = datadir + tag + '_%s%s_%i_%i.Agsm'%(p, p, len(tlist)*len(ubls[p]), 12*nside**2) #.Agsm is nested galactic for this script, whereas .A is ring equtorial

                print "Reading A matrix from %s"%A_filename
                sys.stdout.flush()
                A_vis[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(ubls[p]), len(tlist), 12*nside**2))[np.ix_(ubl_mask, tmasks[p])].reshape((np.sum(ubl_mask)*len(tlist[tmasks[p]]), 12*nside**2))

                #get Ni (1/variance) and data
                var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, vartag)
                Ni[p] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[np.ix_(tmasks[p], ubl_mask)].transpose().flatten() * (1.e-26*(C/vis_freq)**2/2/kB/(4*np.pi/(12*nside**2)))**2)

                vis_data[p] = (np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[np.ix_(tmasks[p], ubl_mask)].transpose().flatten()*1.e-26*(C/vis_freq)**2/2/kB/(4*np.pi/(12*nside**2))).conjugate()#there's a conjugate convention difference
                data_shape[p] = (np.sum(ubl_mask), np.sum(tmasks[p]))
                ubls[p] = ubls[p][ubl_mask]
                ubl_sort[p] = np.argsort(la.norm(ubls[p], axis=1))

            print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
            sys.stdout.flush()

            vis_data = np.concatenate((np.real(vis_data['x']), np.real(vis_data['y']), np.imag(vis_data['x']), np.imag(vis_data['y'])))
            Ni = np.concatenate((Ni['x'], Ni['y']))
            Ni = np.concatenate((Ni * 2, Ni * 2))

            A_vis = np.concatenate((np.real(A_vis['x']), np.real(A_vis['y']), np.imag(A_vis['x']), np.imag(A_vis['y']))).astype('float32') / 2 #factor of 2 for polarization: each pol only receives half energy
            # AtNi_vis = np.transpose(A_vis).dot(Ni)
            # AtBib_vis = AtNi_vis.dot(vis_data)
            #need to convert galactic ring to equatorial nest

            print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
            sys.stdout.flush()

            # AtA_filename = datadir + tag + '_%i_%i.AtA'%(len(tlist)*len(ublss), 12*nside**2)
            # if os.path.isfile(AtA_filename):
            #     print "Reading AtA from " + AtA_filename
            #     sys.stdout.flush()
            #     AtA = np.fromfile(AtA_filename, dtype='float32').reshape((12*nside**2, 12*nside**2))
            # else:
            #     tm = time.time()
            #     AtA = np.einsum('ki,kj->ij', A_vis, A_vis)
            #     print (time.time() - tm) / 60., "mins for AtA"
            #     sys.stdout.flush()
            #     AtA.astype('float32').tofile(AtA_filename)
            #
            #

            vis_w_estimates = np.array([we(np.log10(vis_freq)) for we in w_estimates])
            vis_w_estimates_final = np.array([we(np.log10(vis_freq)) for we in w_estimates_final])
            vis_map_estimate_final = np.transpose(xbar_ni_final).dot(vis_w_estimates_final)
            vis_data_estimate_final = A_vis.dot(vis_map_estimate_final)

            vis_relevant_mask = np.abs(vis_w_estimates_final) >= .0#1

            additive_calibrate_matrix = np.zeros((len(vis_data), 2 * (data_shape['x'][0] + data_shape['y'][0]) + np.sum(vis_relevant_mask)))
            row = 0
            col = 0
            for ri in range(2):
                for shp in [data_shape['x'], data_shape['y']]:
                    for i in range(shp[0]):
                        additive_calibrate_matrix[row:row+shp[1], col] = 1
                        col += 1
                        row += shp[1]

            #for pre-physics version
            additive_calibrate_matrix[:, -np.sum(vis_relevant_mask):] = np.einsum('ji,ni->jn', A_vis, xbar_ni[vis_relevant_mask])
            additive_sol = la.inv(np.einsum('ki,k,kj->ij', additive_calibrate_matrix, Ni, additive_calibrate_matrix)).dot(np.transpose(additive_calibrate_matrix).dot(Ni * vis_data))
            vis_data_fit = additive_calibrate_matrix.dot(additive_sol)

            vis_norm = la.norm(np.transpose(xbar_ni[vis_relevant_mask]).dot(additive_sol[-np.sum(vis_relevant_mask):])) / np.mean(la.norm(x_fit, axis=1))
            Sinv = np.zeros((additive_calibrate_matrix.shape[1], additive_calibrate_matrix.shape[1]))
            for i, col in enumerate(range(-np.sum(vis_relevant_mask), 0)):
                Sinv[col, col] = (vis_norm * vis_w_estimates[i]) ** -2.
            additive_sol = la.inv(Sinv + np.einsum('ki,k,kj->ij', additive_calibrate_matrix, Ni, additive_calibrate_matrix)).dot(np.transpose(additive_calibrate_matrix).dot(Ni * vis_data))
            vis_data_fit = additive_calibrate_matrix.dot(additive_sol)

            vis_norm = la.norm(np.transpose(xbar_ni[vis_relevant_mask]).dot(additive_sol[-np.sum(vis_relevant_mask):])) / np.mean(la.norm(x_fit, axis=1))
            vis_w = additive_sol[-np.sum(vis_relevant_mask):] / vis_norm
            vis_norms[tag_i] = vis_norm
            vis_ws[tag_i][vis_relevant_mask] = vis_w
            # print tag_i, vis_data[:10], vis_w, vis_ws[tag_i][vis_relevant_mask]

            #for post-physics version
            additive_calibrate_matrix[:, -np.sum(vis_relevant_mask):] = np.einsum('ji,ni->jn', A_vis, xbar_ni_final[vis_relevant_mask])
            additive_sol = la.inv(np.einsum('ki,k,kj->ij', additive_calibrate_matrix, Ni, additive_calibrate_matrix)).dot(np.transpose(additive_calibrate_matrix).dot(Ni * vis_data))
            vis_data_fit = additive_calibrate_matrix.dot(additive_sol)

            vis_norm = la.norm(np.transpose(xbar_ni_final[vis_relevant_mask]).dot(additive_sol[-np.sum(vis_relevant_mask):])) / np.mean(la.norm(x_fit, axis=1))
            Sinv = np.zeros((additive_calibrate_matrix.shape[1], additive_calibrate_matrix.shape[1]))
            for i, col in enumerate(range(-np.sum(vis_relevant_mask), 0)):
                Sinv[col] = (vis_norm * vis_w_estimates_final[i]) ** -2.
            additive_sol = la.inv(Sinv + np.einsum('ki,k,kj->ij', additive_calibrate_matrix, Ni, additive_calibrate_matrix)).dot(np.transpose(additive_calibrate_matrix).dot(Ni * vis_data))
            vis_data_fit = additive_calibrate_matrix.dot(additive_sol)

            vis_norm = la.norm(np.transpose(xbar_ni_final[vis_relevant_mask]).dot(additive_sol[-np.sum(vis_relevant_mask):])) / np.mean(la.norm(x_fit, axis=1))
            vis_w = additive_sol[-np.sum(vis_relevant_mask):] / vis_norm
            vis_norms_final[tag_i] = vis_norm
            vis_ws_final[tag_i][vis_relevant_mask] = vis_w
            # print tag_i, vis_data[:10], vis_w, vis_ws[tag_i][vis_relevant_mask]


            # scale_fit = additive[-1]
            # additive_term = {}
            # qaz = additive[:len(additive) / 2] + 1.j * additive[len(additive) / 2:-1]
            # additive_term['x'] = qaz[:data_shape['x'][0]]
            # additive_term['y'] = qaz[data_shape['x'][0]:]
            #
            # vis_data_fit_error = {}
            # qaz = (vis_data_fit - vis_data)[:len(vis_data) / 2] + 1.j * (vis_data_fit - vis_data)[len(vis_data) / 2:]
            # vis_data_fit_error['x'] = qaz[:data_shape['x'][0] * data_shape['x'][1]].reshape(data_shape['x'])
            # vis_data_fit_error['y'] = qaz[data_shape['x'][0] * data_shape['x'][1]:].reshape(data_shape['y'])
            #
            # plt.plot(additive_term['x'])
            #
            #
            if len(vis_tags) <= 3:
                dbg_matrix = np.copy(additive_calibrate_matrix[:, :-n_principal+1])
                dbg_matrix[:, -1] = vis_data_estimate_final
                dbg = la.inv(np.einsum('ki,k,kj->ij', dbg_matrix, Ni, dbg_matrix)).dot(np.transpose(dbg_matrix).dot(Ni * vis_data))

                fig = plt.Figure(figsize=(2000, 1000))
                fig.set_canvas(plt.gcf().canvas)
                plt.gcf().set_size_inches(w=30, h=10)
                t_plot = (tlist[tmasks['x']] - 10.)%24 + 10
                for i, bl in enumerate(ubl_sort['x']):
                    plt.subplot(8, 10, i + 1)
                    pstart, pend = (bl * data_shape['x'][1], (bl + 1) * data_shape['x'][1])
                    plt.plot(t_plot, vis_data[pstart:pend])
                    plt.plot(t_plot, vis_data_fit[pstart:pend])
                    plt.plot(t_plot, dbg[-1] * vis_data_estimate_final[pstart:pend])
                    plt.title('(%.1f, %.1f) %.1f'%(ubls['x'][bl][0], ubls['x'][bl][1], la.norm(ubls['x'][bl])))
                    plt.xlim([15, 30])
                fig.savefig(plot_filename_base + datatag + '_vis_fit.png', dpi=1000)
                fig.clear()
                plt.gcf().clear()


        make_result_plot(all_freqs, w_nf, xbar_ni, w_estimates, normalization, '_vis_result_plot1', n_principal, show_plots, vis_freqs=vis_freqs, vis_ws=vis_ws, vis_norms=vis_norms)
        make_result_plot(all_freqs, w_nf_final, xbar_ni_final, w_estimates_final, normalization, '_vis_result_plot2', n_principal, show_plots, vis_freqs=vis_freqs, vis_ws=vis_ws_final, vis_norms=vis_norms_final)
        for i in range(n_principal):
            plt.subplot(1, n_principal, i+1)
            plt.plot(sorted(vis_freqs), vis_ws[np.argsort(vis_freqs), i], 'go')
            plt.plot(sorted(vis_freqs), w_estimates[i](sorted(np.log10(vis_freqs))), 'b-')
            plt.ylim([-1.5, 1.5])
        plt.show()
        for i in range(n_principal):
            plt.subplot(1, n_principal, i+1)
            plt.plot(sorted(vis_freqs), vis_ws_final[np.argsort(vis_freqs), i], 'go')
            plt.plot(sorted(vis_freqs), w_estimates_final[i](sorted(np.log10(vis_freqs))), 'b-')
            plt.ylim([-1.5, 1.5])
        plt.show()

        good_vis_mask = vis_ws_final[:, 4] >= 0
        good_vis_freqs = vis_freqs[good_vis_mask]
        vis_maps = vis_ws_final[good_vis_mask].dot(xbar_ni_final)
        for plot_i, plot_freq in enumerate(np.arange(.13, .172, .001)):
            fi = np.argmin(np.abs(good_vis_freqs - plot_freq))
            hpv.mollview(np.log10(vis_maps[fi]), nest=True, title='%.1f MHz'%(good_vis_freqs[fi] * 1e3), min=-2.5, max=-1)
            plt.savefig('/home/omniscope/gif_dir/%04i.png'%plot_i)
            plt.clf()

        for plot_i, logfreq in enumerate(np.arange(-2., 4., .03)):
            pltdata = np.log10(np.abs(np.transpose(xbar_ni).dot([w_estimates[i](logfreq) for i in range(n_principal)])))
            hpv.mollview(pltdata, nest=True, title='%.2f GHz'%(10**logfreq), min=np.percentile(pltdata, 5), max=np.percentile(pltdata, 98))
            plt.savefig('/home/omniscope/gif_dir/%04i.png'%plot_i)
            plt.clf()

        for i in range(n_principal):
            plt.subplot(1, n_principal, i+1)
            plt.plot(sorted(good_vis_freqs), vis_ws_final[good_vis_mask][np.argsort(good_vis_freqs), i], 'go')
            plt.plot(sorted(good_vis_freqs), w_estimates_final[i](sorted(np.log10(good_vis_freqs))), 'b-')
            plt.ylim([-1.5, 1.5])
        plt.show()

        np.savez(result_filename, freqs=all_freqs, w_nf=w_nf, x_ni=xbar_ni, normalization=normalization, w_nf_final=w_nf_final, x_ni_final=xbar_ni_final, vis_freqs=vis_freqs, vis_ws=vis_ws, vis_norms=vis_norms, vis_ws_final=vis_ws_final, vis_norms_final=vis_norms_final)
    ##########################################
    ###ICA#####can get cmb very well
    ##########################################
    ica = FastICA(n_components=n_principal)

    #ica on pca data with gap filled by pca results
    ica_data = np.copy(x_fi)
    ica_data[np.isnan(ica_data)] = x_fit[np.isnan(ica_data)]
    xbar_ica = ica.fit_transform(np.transpose(ica_data))

    fig = plt.Figure(figsize=(200, 100))
    fig.set_canvas(plt.gcf().canvas)
    for i in range(n_principal):
        cmap = cm.gist_rainbow_r
        cmap.set_under('w')
        cmap.set_bad('gray')
        plot_data_lin = xbar_ica[:, i] * np.sign(xbar_ica[hpf.vec2pix(mother_nside, 1, 0, 0, nest=True), i])
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin) / np.log(10.)

        hpv.mollview(plot_data, nest=True, sub=(3, n_principal, i + 1), min=np.percentile(plot_data, 2), max=np.percentile(plot_data, 98), cmap=cmap, title='%.3e'%la.norm(plot_data_lin))

    for i in range(n_principal):
        plt.subplot(3, n_principal, i + n_principal + 1)
        plt.plot(np.log10(all_freqs), ica.mixing_[:, i] * np.sign(np.mean(ica.mixing_[:, i])))
        plt.xlim([np.log10(all_freqs[0]), np.log10(all_freqs[-1])])
        plt.ylim([0,3])

    for i in range(n_principal):
        plt.subplot(3, n_principal, i + 2 * n_principal + 1)
        plt.plot(np.log10(all_freqs), np.log10(normalization * ica.mixing_[:, i] * np.sign(np.mean(ica.mixing_[:, i]))))
        plt.xlim([np.log10(all_freqs[0]), np.log10(all_freqs[-1])])
        # plt.ylim([0,3])
    fig.savefig(plot_filename_base + '_ica_plot.png', dpi=1000)
    if show_plots:
        plt.show()
    fig.clear()
    plt.gcf().clear()

# ##############################################
# ##############################################
# ####start I data processing method 2
# ##############################################
# ##############################################
# D = all_idata / normalization[:, None]
# D[np.isnan(D)] = 0.
# DD = np.einsum('ik,jk->ij', D, D)
# CC = np.einsum('ik,jk->ij', np.array(D!=0, dtype='float32'), np.array(D!=0, dtype='float32'))
# ev2, ec2 = la.eigh(DD / CC)