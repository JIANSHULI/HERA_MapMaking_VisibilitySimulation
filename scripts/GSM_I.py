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
import sys, time, os
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


###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
mother_nside = 64
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 0#3. * np.pi / 180.
edge_width = 2. * np.pi / 180.
remove_cmb = True

n_principal_range = range(6, 7)
include_visibility = True

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

if include_visibility:
    ###prepare big matrices stuff for interferometer data###
    tag = "q3_abscalibrated"
    datatag = '_seccasa_polcor.rad'
    vartag = '_seccasa_polcor'
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
    nt = 440
    nf = 1
    nUBL = 75
    nside = mother_nside
    bnside = 8

    C = .299792458
    kB = 1.3806488 * 1.e-23

    A_vis = {}
    vis_data = {}
    Ni = {}
    data_shape = {}
    ubls = {}
    ubl_sort = {}
    for p in ['x', 'y']:
        pol = p+p

        #tf file
        tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
        tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
        tlist = np.real(tflist[:, 0])
        flist = np.imag(tflist[0, :])    #will be later assuming flist only has 1 element
        vis_freq = flist[0] / 1e3
        print vis_freq
        ubl_length_thresh = 1.4 * 0.3 / vis_freq * mother_nside / 10

        #tf mask file, 0 means flagged bad data
        try:
            tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
            tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
            tmask = np.array(tfmlist[:,0].astype('bool'))
            #print tmask
        except:
            print "No mask file found"
            tmask = np.zeros_like(tlist).astype(bool)
        #print vis_freq, tlist

        #ubl file
        ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
        ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
        ubl_mask = la.norm(ubls[p], axis=1) <= ubl_length_thresh
        print "%i UBLs to include"%len(ubls[p])


        #compute A matrix
        A_filename = datadir + tag + '_%s%s_%i_%i.Agsm'%(p, p, len(tlist)*len(ubls[p]), 12*nside**2) #.Agsm is nested galactic for this script, whereas .A is ring equtorial

        print "Reading A matrix from %s"%A_filename
        sys.stdout.flush()
        A_vis[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(ubls[p]), len(tlist), 12*nside**2))[np.ix_(ubl_mask, tmask)].reshape((np.sum(ubl_mask)*len(tlist[tmask]), 12*nside**2))

        #get Ni (1/variance) and data
        var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL,vartag)
        Ni[p] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[np.ix_(tmask, ubl_mask)].transpose().flatten() * (1.e-26*(C/vis_freq)**2/2/kB/(4*np.pi/(12*nside**2)))**2)
        data_filename = datadir + tag + '_%s%s_%i_%i'%(p, p, nt, nUBL) + datatag
        vis_data[p] = (np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[np.ix_(tmask, ubl_mask)].transpose().flatten()*1.e-26*(C/vis_freq)**2/2/kB/(4*np.pi/(12*nside**2))).conjugate()#there's a conjugate convention difference
        data_shape[p] = (np.sum(ubl_mask), np.sum(tmask))
        ubls[p] = ubls[p][ubl_mask]
        ubl_sort[p] = np.argsort(la.norm(ubls[p], axis=1))

    print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
    sys.stdout.flush()

    vis_data = np.concatenate((np.real(vis_data['x']), np.real(vis_data['y']), np.imag(vis_data['x']), np.imag(vis_data['y'])))
    Ni = np.concatenate((Ni['x'], Ni['y']))
    Ni = np.concatenate((Ni/2, Ni/2))

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

    w_estimates = [si.interp1d(np.log10(all_freqs), w_nf[i], kind='slinear', assume_sorted=False) for i in range(n_principal)]

    result_filename = data_file_name.replace('data_', 'result_%i+%i_'%(len(idata), len(addon_idata))).replace('.npz', '_principal_%i_step_%.2f'%(n_principal, step_size))
    np.savez(result_filename, w_nf=w_nf, x_ni=xbar_ni, normalization=normalization)

    matplotlib.rcParams.update({'font.size': 5})
    plot_filename_base = result_filename.replace('result', 'plot')


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

    for i in range(n_principal):
        cmap = cm.gist_rainbow_r
        cmap.set_under('w')
        cmap.set_bad('gray')
        plot_data_lin = xbar_ni[i] * np.sign(xbar_ni[i, hpf.vec2pix(mother_nside, 1, 0, 0, nest=True)])
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin) / np.log(10.)

        hpv.mollview(plot_data, nest=True, sub=(3, n_principal, i + 1), min=np.percentile(plot_data, 2), max=np.percentile(plot_data, 98), cmap=cmap, title='%.3e'%la.norm(plot_data_lin))

        plt.subplot(3, n_principal, i + 1 + n_principal)
        plt.plot(np.log10(all_freqs), w_nf[i], 'r+')
        interp_x = np.arange(np.log10(all_freqs[0]), np.log10(all_freqs[-1]), .01)
        interp_y = w_estimates[i](interp_x)
        plt.plot(interp_x, interp_y, 'b-')
        plt.ylim([-1.5, 1.5])

        plt.subplot(3, n_principal, i + 1 + 2 * n_principal)
        plt.plot(np.log10(all_freqs), np.log10(w_nf[i] * normalization), '+')
        plt.xlim([np.log10(all_freqs[0]), np.log10(all_freqs[-1])])
        plt.ylim(-5, 8)
    fig.savefig(plot_filename_base + '_result_plot.png', dpi=1000)
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
    if include_visibility:
        vis_map_estimate = np.transpose(xbar_ni).dot([we(np.log10(vis_freq)) for we in w_estimates])
        vis_data_estimate = A_vis.dot(vis_map_estimate)

        additive_calibrate_matrix = np.zeros((len(vis_data), 2 * (data_shape['x'][0] + data_shape['y'][0]) + n_principal))
        row = 0
        col = 0
        for ri in range(2):
            for shp in [data_shape['x'], data_shape['y']]:
                for i in range(shp[0]):
                    additive_calibrate_matrix[row:row+shp[1], col] = 1
                    col += 1
                    row += shp[1]

        additive_calibrate_matrix[:, -n_principal:] = np.einsum('ji,ni->jn', A_vis, xbar_ni)
        additive = la.inv(np.einsum('ki,k,kj->ij', additive_calibrate_matrix, Ni, additive_calibrate_matrix)).dot(np.transpose(additive_calibrate_matrix).dot(Ni * vis_data))
        vis_data_fit = additive_calibrate_matrix.dot(additive)

        vis_norm = la.norm(np.transpose(xbar_ni).dot(additive[-n_principal:])) / np.mean(la.norm(x_fit, axis=1))

        vis_w = additive[-n_principal:] / vis_norm

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
        fig = plt.Figure(figsize=(2000, 1000))
        fig.set_canvas(plt.gcf().canvas)
        plt.gcf().set_size_inches(w=30, h=10)
        for i, bl in enumerate(ubl_sort['x']):
            plt.subplot(8, 10, i + 1)
            pstart, pend = (bl * data_shape['x'][1], (bl + 1) * data_shape['x'][1])
            plt.plot(vis_data[pstart:pend])
            plt.plot(vis_data_fit[pstart:pend])
            plt.title('(%.1f, %.1f) %.1f'%(ubls['x'][bl][0], ubls['x'][bl][1], la.norm(ubls['x'][bl])))
        fig.savefig(plot_filename_base + '_vis_fit.png', dpi=1000)
        fig.clear()
        plt.gcf().clear()


        fig = plt.Figure(figsize=(200, 100))
        fig.set_canvas(plt.gcf().canvas)

        for i in range(n_principal):
            plt.subplot(1, n_principal, 1 + i)
            plt.plot(np.log10(all_freqs), w_nf[i], 'r+')
            interp_x = np.arange(np.log10(all_freqs[0]), np.log10(all_freqs[-1]), .01)
            interp_y = w_estimates[i](interp_x)
            plt.plot(interp_x, interp_y, 'b-')
            plt.plot([np.log10(vis_freq)], vis_w[i], 'g+')
            plt.ylim([-1.5, 1.5])

        fig.savefig(plot_filename_base + '_vis_result_plot.png', dpi=1000)
        if show_plots:
            plt.show()
        fig.clear()
        plt.gcf().clear()

        # cholesky_name = AtA_filename.replace('AtA', os.path.basename(result_filename)) + '.icholesky'
        # if os.path.isfile(cholesky_name):
        #     print "Loading", cholesky_name
        #     sys.stdout.flush()
        #     sv.InverseCholeskyMatrix.fromfile(cholesky_name, mother_npix * n_principal, 'float32')
        # else:
        #     A1 = sps.bmat([[sps.diags([~np.isnan(all_idata[f])], [0], format='dia', dtype='float32') * w_nf[n, f] for n in range(n_principal)] for f in range(len(all_freqs))], format='csc')
        #     A1tA1 = A1.T * A1
        #
        #     print "Computing A2tA2"
        #     sys.stdout.flush()
        #     #A2 = [A_vis * w_estimates[n](np.log10(vis_freq)) for n in range(n_principal)]
        #     A2tA2 = np.bmat([[w_estimates[n](np.log10(vis_freq)) * w_estimates[m](np.log10(vis_freq)) * AtA for m in range(n_principal)] for n in range(n_principal)])
        #
        #     print "Computing BtB"
        #     sys.stdout.flush()
        #     BtB = A1tA1.todense() + A2tA2
            # del A2tA2
            # del A1tA1
            #
            # print "Computing BtB cholesky"
            # sys.stdout.flush()
            # tm = time.time()
            # BtBi = sv.InverseCholeskyMatrix(BtB)
            # print (time.time() - tm) / 60., "minutes for Cholesky"
            # sys.stdout.flush()
            # BtBi.tofile(cholesky_name)

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
    #
    # ##########################################
    # ###seperate physical components#####
    # ##########################################
    # X = np.einsum('ni,mi->nm', xbar_ni, xbar_ni)
    # A = np.zeros((n_principal**2, n_principal**2))
    # for xa in range(n_principal):
    #     for xb in range(n_principal):
    #         nrow = xa * n_principal + xb
    #         for mi in range(n_principal):
    #             for mj in range(n_principal):
    #                 ncol = mi * n_principal + mj
    #                 if mi != xa:
    #                     A[nrow, ncol] = X[xb, mj]
    #                 else:
    #                     A[nrow, ncol] = -X[xb, mj]
    #

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