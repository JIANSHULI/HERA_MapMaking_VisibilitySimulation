__author__ = 'omniscope'

import numpy as np
import numpy.linalg as la
import sys
from matplotlib import cm
import healpy.pixelfunc as hpf
import healpy as hp
import scipy.interpolate as si
try:
    import healpy.visufunc as hpv
except:
    pass
import matplotlib.pyplot as plt

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
        fill_mask = np.array([not bool(int(cc)) for cc in bin(n_reg)[2:].zfill(n_incomplete)])[::-1]
        matching_mask = (region_sum == n_reg)
        if matching_mask.any():
            region_mask_list.append(matching_mask)
            region_indices_list.append(sorted(np.concatenate((incomplete_fs[fill_mask], complete_fs))))
    return region_indices_list, region_mask_list

kB = 1.38065e-23
C = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / C)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * C / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / C)**2 * kB
    return  K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def Bnu(T, nu):
    return 2 * h * nu**3 / (C**2 * (np.exp(hoverk * nu / T) - 1))

def dust_spectrum(T, beta, nu):
    return Bnu(T, nu) * nu**beta

def get_derivative(x):
    result = np.zeros_like(x)
    result[1:] = x[1:] - x[:-1]
    return result

def get_derivative2(x):
    return get_derivative(get_derivative(x))
########################################
#load data
result_filename = '/mnt/data0/omniscope/polarized foregrounds/result_25+4_nside_64_smooth_8.73E-02_edge_5.24E-02_rmvcmb_1_UV0_v3.0_principal_6_step_1.00_err_remove_pt.npz'
# result_filename = '/mnt/data0/omniscope/polarized foregrounds/result_25+4_nside_128_smooth_6.28E-02_edge_5.24E-02_rmvcmb_1_UV0_v3.0_principal_6_step_1.00_err_remove_pt.npz'
f = np.load(result_filename)
w_nfo = f['w_nf']#n_principal by frequency
w_nf = f['w_nf'][:, 1:]#n_principal by frequency
x_ni = f['x_ni']#n_principal by pixel
freqs = f['freqs'][1:]#GHz
freqso = f['freqs']#GHz
# ps_mask = f['ps_mask']
# x_ni *= (1-ps_mask)
n_f = len(freqs)
n_principal = len(w_nf)
nside = hpf.npix2nside(x_ni.shape[1])
########################################
normalizationo = f['normalization']
normalizationo[freqso < 20] = K_RJ2MJysr(normalizationo[freqso < 20], freqso[freqso < 20] * 1e9)
normalizationo[(freqso >= 20) & (freqso < 500)] = K_CMB2MJysr(normalizationo[(freqso >= 20) & (freqso < 500)], freqso[(freqso >= 20) & (freqso < 500)] * 1e9)

normalization = normalizationo[1:]

################################################
#plot orthogonal results
cmap = cm.gist_rainbow_r
cmap.set_under('w')
cmap.set_bad('gray')
def plot_components(M=np.eye(n_principal)):
    w_nf_local = M.dot(w_nf)
    x_ni_local = la.inv(M).transpose().dot(x_ni)
    for n in range(n_principal):


        sign_flip = np.sign(np.nanmedian(x_ni_local[n]))
        # sign_flip = np.sign(w_nf_local[n, np.argmax(np.abs(w_nf_local[n]))])

        plot_data_lin = x_ni_local[n] * sign_flip
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin * 1 / (np.median(np.abs(plot_data_lin))))
        try:
            hpv.mollview(plot_data, nest=True, sub=(5, n_principal, n + 1), cmap=cmap, title="%.4f"%la.norm(plot_data_lin), max=np.percentile(plot_data, 98), min=np.percentile(plot_data, 2), )
        except:
            print "NEED HEALPY PACKAGE FOR PLOTTING!"
        plt.subplot(5, n_principal, n_principal + n + 1)
        plt.plot(np.log10(freqs), sign_flip * w_nf_local[n])
        plt.plot(np.log10(freqs), sign_flip * w_nf_local[n], 'r+')
        plt.ylim([-1, 1])
        plt.subplot(5, n_principal, 2 * n_principal + n + 1)
        plt.plot(np.log10(freqs), np.log10(np.abs(w_nf_local[n] * normalization * la.norm(plot_data_lin))))
        plt.plot(np.log10(freqs), np.log10(np.abs(w_nf_local[n] * normalization * la.norm(plot_data_lin))), 'r+')
        plt.subplot(5, n_principal, 3 * n_principal + n + 1)
        plt.plot(np.log10(freqs), sign_flip * get_derivative(w_nf_local[n]))
        plt.plot(np.log10(freqs), sign_flip * get_derivative(w_nf_local[n]), 'r+')
        plt.ylim([-1, 1])
        plt.subplot(5, n_principal, 4 * n_principal + n + 1)
        plt.plot(np.log10(freqs), sign_flip * get_derivative2(w_nf_local[n]))
        plt.plot(np.log10(freqs), sign_flip * get_derivative2(w_nf_local[n]), 'r+')
        plt.ylim([-1, 1])

    plt.show()

def plot_components_publication(M=np.eye(n_principal)):
    w_nf_local = M.dot(w_nf)
    x_ni_local = la.inv(M).transpose().dot(x_ni)
    for n in range(n_principal):

        sign_flip = np.sign(np.nanmedian(x_ni_local[n]))
        # sign_flip = np.sign(w_nf_local[n, np.argmax(np.abs(w_nf_local[n]))])

        plot_data_lin = x_ni_local[n] * sign_flip
        # if i == 0:
        #     plot_data = np.log10(plot_data)
        # else:
        plot_data = np.arcsinh(plot_data_lin * 1 / (np.median(np.abs(plot_data_lin))))
        try:
            hpv.mollview(plot_data, nest=True, sub=((n_principal + 1) / 2, 2, n + 1), cmap=cmap, title="%.4f"%la.norm(plot_data_lin), max=np.percentile(plot_data, 98), min=np.percentile(plot_data, 2), )
        except:
            print "NEED HEALPY PACKAGE FOR PLOTTING!"
    plt.show()

eigen_values = np.zeros((n_f, n_f, n_principal))
eigen_vecs = np.zeros((n_f, n_f, n_principal, n_principal))
ex_eigen_values = np.zeros((n_f, n_f, n_principal))#eigenvalues for sections excluding f0-f1
ex_eigen_vecs = np.zeros((n_f, n_f, n_principal, n_principal))#eigenvecs for sections excluding f0-f1
for fstart in range(n_f):
    for fend in range(n_f):
        f_range_mask = np.zeros(n_f, dtype='bool')
        f_range_mask[fstart:fend+1] = True
        tmp_w_nf = w_nf[:, f_range_mask]
        eigen_values[fstart, fend], eigen_vecs[fstart, fend] = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))
        tmp_w_nf = w_nf[:, ~f_range_mask]
        ex_eigen_values[fstart, fend], ex_eigen_vecs[fstart, fend] = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))

plot_components()
###STEP 1
###excluding range eigenvalue approach: pick out clean modes
w_nf_intermediate = np.copy(w_nf)
fs_intermediate = np.arange(n_f)
M = np.eye(n_principal)
project_M = np.eye(n_principal)
max_component_range = 12
max_thresh = 1e-2
start_n_principal = 0
for i in range(n_principal):
    project_range = [0, max_component_range]
    thresh = 1e-4
    project_M = np.eye(n_principal - i)
    while np.allclose(project_M, np.eye(n_principal - i)) and thresh <= max_thresh:

        for f0, fstart in enumerate(fs_intermediate):
            for f1, fend in enumerate(fs_intermediate):
                if f1-f0 < project_range[1] - project_range[0]:
                    f_range_mask = np.ones(len(fs_intermediate), dtype='bool')
                    f_range_mask[f0:f1+1] = False
                    tmp_w_nf = w_nf_intermediate[:, f_range_mask]
                    ev, ec = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))
                    if ev[0] / np.sum(ev) <= thresh:
                        project_range = [f0, f1]
                        project_M = ec.transpose()
        thresh *= 2
    if thresh > max_thresh:
        break
    print thresh, fs_intermediate[project_range]
    if la.norm(np.array([10, 19]) - np.array(fs_intermediate[project_range])) <= 5:
        cmb_principal = i
    project_M = project_M / la.norm(project_M.dot(w_nf_intermediate), axis=-1)[:, None]

    # new_w_nf = project_M.dot(w_nf_intermediate)
    # isolated_mode = new_w_nf[0]
    # isolate_matrix = np.eye(n_principal - i)
    # isolate_matrix[1:, 0] = -new_w_nf[1:].dot(isolated_mode) / isolated_mode.dot(isolated_mode)
    # project_M = isolate_matrix.dot(project_M)

    M[i:] = project_M.dot(M[i:])
    w_nf_intermediate = project_M.dot(w_nf_intermediate)[1:]
    start_n_principal += 1
    # fs_intermediate = np.concatenate((fs_intermediate[:project_range[0]], fs_intermediate[project_range[1]+1:]))
plot_components(M)

# ##STEP 2
# ##dominating eigenvalue approach: not very effective
# # M = np.eye(n_principal)
# # start_n_principal = 2
# w_nf_intermediate = M.dot(w_nf)[start_n_principal:]
# fs_intermediate = np.arange(n_f)
# # project_M = np.eye(n_principal)
# for i in range(n_principal - start_n_principal):
#
#     thresh = 1e-3
#     project_range = [-1, -1]
#     for f0, fstart in enumerate(fs_intermediate):
#         for f1, fend in enumerate(fs_intermediate):
#             if f1-f0 > project_range[1] - project_range[0]:
#                 tmp_w_nf = w_nf_intermediate[:, f0:f1+1]
#                 ev, ec = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))
#                 if ev[-1] / np.sum(ev) >= 1 - thresh:
#                     project_range = [f0, f1]
#                     project_M = ec.transpose()
#     print fs_intermediate[project_range]
#     project_M = project_M / la.norm(project_M.dot(w_nf_intermediate), axis=-1)[:, None]
#     M[start_n_principal:n_principal-i] = project_M.dot(M[start_n_principal:n_principal-i])
#     w_nf_intermediate = project_M.dot(w_nf_intermediate)[:n_principal-start_n_principal-i-1]
#     fs_intermediate = np.concatenate((fs_intermediate[:project_range[0]], fs_intermediate[project_range[1]+1:]))
# plot_components(M)

#2nd STEP: cmb
x_ni_intermediate = np.transpose(la.inv(M)).dot(x_ni)
cmb_m = np.eye(n_principal)#remove foreground in CMB
# cmb_principal = 1
non_cmb_principal_mask = np.array([False, False, False] + [True] * (n_principal - 3))#np.arange(n_principal) != cmb_principal#
# if abs(np.min(x_ni_intermediate[cmb_principal])) > np.max(x_ni_intermediate[cmb_principal]):
#     plane_mask = (x_ni_intermediate[cmb_principal] <  -1.5*np.max(x_ni_intermediate[cmb_principal])) & (x_ni_intermediate[cmb_principal] > np.percentile(x_ni_intermediate[cmb_principal], .5))
# else:
#     plane_mask = (x_ni_intermediate[cmb_principal] >  1.5*abs(np.min(x_ni_intermediate[cmb_principal]))) & (x_ni_intermediate[cmb_principal] < np.percentile(x_ni_intermediate[cmb_principal], 99.5))
# plane_mask = np.ones_like(plane_mask)
# At_cmb = x_ni_intermediate[np.ix_(non_cmb_principal_mask, plane_mask)]
# cmb_m[cmb_principal, non_cmb_principal_mask] = -la.inv(At_cmb.dot(np.transpose(At_cmb))).dot(At_cmb.dot(x_ni_intermediate[cmb_principal, plane_mask]))
At_cmb = x_ni_intermediate[non_cmb_principal_mask]
cmb_m[cmb_principal, non_cmb_principal_mask] = -la.inv(At_cmb.dot(np.transpose(At_cmb))).dot(At_cmb.dot(x_ni_intermediate[cmb_principal]))

M = la.inv(cmb_m.transpose()).dot(M)
M = M / la.norm(M.dot(w_nf), axis=-1)[:, None]
plot_components(M)


#STep 3: manual

manual_spike_freq_ranges = [[10,-10]] * n_principal
# manual_spike_freq_ranges[0] = [0, 30]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[2] = [.5, 3]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[3] = [1e3, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[4] = [100, 1e3]#[100, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[5] = [10, 100]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_ranges = [np.arange(len(freqs))[(np.array(freqs) >= freq_range[0]) & (np.array(freqs) <= freq_range[1])] for freq_range in manual_spike_freq_ranges]

for i1, p1 in enumerate([0]):#, 1, 2, 3, 4, 5]):
    for i2, p2 in enumerate([0]):#, 1, 2, 3, 4, 5]):

        M1 = np.diag(la.norm(la.inv(M).transpose().dot(x_ni),axis=-1))
        niter = 0
        while niter < 500 and (niter == 0 or la.norm(M2 - np.eye(n_principal)) > 1e-3 * n_principal):
            niter += 1
            w_nf_intermediate = M1.dot(M).dot(w_nf)
            M2 = np.eye(n_principal)
            reg = 0#.0001
            step = .2
            for spike_principal, spike_fs in enumerate(manual_spike_ranges):
                if len(spike_fs) > 0:
                    non_spike_fs = np.array([i for i in range(0, len(freqs)) if i not in spike_fs])
                    non_spike_principals = np.array([i for i in range(n_principal) if i != spike_principal and i != cmb_principal])
                    spike_A = np.transpose(w_nf_intermediate[np.ix_(non_spike_principals, non_spike_fs)])

                    importance_weight = (np.abs(w_nf_intermediate[spike_principal, non_spike_fs]) / np.sum(np.abs(w_nf_intermediate[:, non_spike_fs]), axis=0))
                    distance_weight = np.min(np.abs([np.log10(freqs) - np.log10(manual_spike_freq_ranges[spike_principal][0]), np.log10(freqs) - np.log10(manual_spike_freq_ranges[spike_principal][1])]), axis=0)[non_spike_fs] / 4.
                    Ni = importance_weight**p1 * distance_weight**p2
                    # Ni = np.ones(len(non_spike_fs))
                    M2[spike_principal, non_spike_principals] = -step * la.inv((np.transpose(spike_A)*Ni).dot(spike_A) + np.eye(len(non_spike_principals)) * reg).dot(np.transpose(spike_A).dot(w_nf_intermediate[spike_principal, non_spike_fs] * Ni))
            M2 = M2 * la.norm(la.inv(M2.dot(M1).dot(M)).transpose().dot(x_ni), axis=-1)[:, None]
            M1 = M2.dot(M1)
        print niter
        plt.subplot(6, 6, i1 * 6 + i2 + 1)
        plt.plot(np.log10(freqs), np.log10(np.abs(M1.dot(M).dot(w_nf).transpose() * normalization[:, None])))
        plt.title((p1, p2))
        plt.ylim([-1, 5])
plt.show()

plot_components(M1.dot(M))
# #
# # manual_spike_freq_ranges = [[10,-10]] * n_principal
# # manual_spike_freq_ranges[2] = [100, 1e6]#[100, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
# # manual_spike_ranges = [np.arange(len(freqs))[(np.array(freqs) >= freq_range[0]) & (np.array(freqs) <= freq_range[1])] for freq_range in manual_spike_freq_ranges]
# # w_nf_intermediate = M1.dot(M).dot(w_nf)
# # M2 = np.eye(n_principal)
# # reg = 0#.0001
# # for spike_principal, spike_fs in enumerate(manual_spike_ranges):
# #     if len(spike_fs) > 0:
# #         non_spike_fs = np.array([i for i in range(0, len(freqs)) if i not in spike_fs])
# #         non_spike_principals = np.array([i for i in range(n_principal) if i != spike_principal and i != cmb_principal])
# #         spike_A = np.transpose(w_nf_intermediate[np.ix_(non_spike_principals, non_spike_fs)])
# #         M2[spike_principal, non_spike_principals] = -la.inv(np.transpose(spike_A).dot(spike_A) + np.eye(len(non_spike_principals)) * reg).dot(np.transpose(spike_A).dot(w_nf_intermediate[spike_principal, non_spike_fs]))
# # M2 = M2 / la.norm(M2.dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
# # plot_components(M2.dot(M1).dot(M))
# #
# # second_der_w = np.array([get_derivative2(w) for w in M1.dot(M).dot(w_nf)])
# # manual_spike_freq_ranges = [[10,-10]] * n_principal
# # manual_spike_freq_ranges[3] = [100, 400]#[100, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
# # manual_spike_freq_ranges[5] = [100, 400]#[100, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
# # manual_spike_ranges = [np.arange(len(freqs))[(np.array(freqs) < freq_range[0]) | (np.array(freqs) > freq_range[1])] for freq_range in manual_spike_freq_ranges]
# # w_nf_intermediate = second_der_w
# # M2 = np.eye(n_principal)
# # reg = 0
# # for spike_principal, spike_fs in enumerate(manual_spike_ranges):
# #     if len(spike_fs) < len(freqs):
# #         non_spike_fs = np.array([i for i in range(0, len(freqs)) if i not in spike_fs])
# #         non_spike_principals = [4]#np.array([i for i in range(n_principal) if i != spike_principal and i != cmb_principal])
# #         spike_A = np.transpose(w_nf_intermediate[np.ix_(non_spike_principals, non_spike_fs)])
# #         M2[spike_principal, non_spike_principals] = -la.inv(np.transpose(spike_A).dot(spike_A) + np.eye(len(non_spike_principals)) * reg).dot(np.transpose(spike_A).dot(w_nf_intermediate[spike_principal, non_spike_fs]))
# # M2 = M2 / la.norm(M2.dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
# # M2[3,4] *= 4
# # M2 = M2 / la.norm(M2.dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
# # plot_components(M2.dot(M1).dot(M))
# #
# # manual_spike_freq_ranges = [[10,-10]] * n_principal
# # # manual_spike_freq_ranges[0] = [0, 30]#, [1, 1000], [10, 1e5], [-10, -10]]
# # manual_spike_freq_ranges[4] = [10, 1e6]#[100, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
# # manual_spike_ranges = [np.arange(len(freqs))[(np.array(freqs) >= freq_range[0]) & (np.array(freqs) <= freq_range[1])] for freq_range in manual_spike_freq_ranges]
# # w_nf_intermediate = M2.dot(M1).dot(M).dot(w_nf)
# # M3 = np.eye(n_principal)
# # reg = 0#.0001
# # for spike_principal, spike_fs in enumerate(manual_spike_ranges):
# #     if len(spike_fs) > 0:
# #         non_spike_fs = np.array([i for i in range(0, len(freqs)) if i not in spike_fs])
# #         non_spike_principals = [0, 2]#np.array([i for i in range(n_principal) if i != spike_principal and i != cmb_principal])
# #         spike_A = np.transpose(w_nf_intermediate[np.ix_(non_spike_principals, non_spike_fs)])
# #         M3[spike_principal, non_spike_principals] = -la.inv(np.transpose(spike_A).dot(spike_A) + np.eye(len(non_spike_principals)) * reg).dot(np.transpose(spike_A).dot(w_nf_intermediate[spike_principal, non_spike_fs]))
# # M3 = M3 / la.norm(M3.dot(M2).dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
# # plot_components(M3.dot(M2).dot(M1).dot(M))
#
# MM = np.eye(n_principal);
# MM[3, 4] = 1.1; MM[4, 5] = .48; MM[4,3] = .06; MM[0, 2] = .03;
# MM2 = np.eye(n_principal);
# MM2[3, 4] = -.98; MM2[5,4] = .2; MM2[4, 0] = .2; MM2[4,2] = -.05; MM2[5, 2] = .01;
# # semifinal_M = MM2.dot(MM).dot(M1).dot(M) / la.norm(MM2.dot(MM).dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
#
# semifinal_M = MM2.dot(MM).dot(M1).dot(M) * la.norm(la.inv(MM2.dot(MM).dot(M1).dot(M)).transpose().dot(x_ni), axis=-1)[:, None]
# # for i in range(n_principal):
# #     semifinal_M[i] *= np.sign(np.mean(semifinal_M.dot(w_nf)))
# plot_components(semifinal_M)
#
# plot_components_publication()
#
# a = 1.216
# b = 0#0.19
# d = -.16
# e = -.605
# qazM = np.eye(n_principal)
# qazM[5, 3] = a
# qazM[3, 4] = b
# qazM[4, 5] = d
# qazM[4, 0] = e
# final_M = qazM.dot(semifinal_M) * la.norm(la.inv(qazM.dot(semifinal_M)).transpose().dot(x_ni), axis=-1)[:, None]
# # plot_components_publication(final_M)
#
# qazM = np.eye(n_principal)
# qazM[4, 5] = -10**-.27
# final_M = qazM.dot(final_M) * la.norm(la.inv(qazM.dot(final_M)).transpose().dot(x_ni), axis=-1)[:, None]
final_M = M1.dot(M)
plot_components_publication(final_M)


final_w_nf = final_M.dot(w_nfo)

#plotting
c = ['b', 'g', 'r', 'c', 'm', 'k']
labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
for m, plot_M in enumerate([np.eye(n_principal) * la.norm(x_ni, axis=-1), final_M]):

    for n, i in enumerate(range(n_principal)):#n, i same thing, just a relic
        if m == 0:
            mask = np.arange(len(freqso))
            label = 'Component ' + str(i)
        else:
            if i < 3 and i != cmb_principal:
                mask = freqso < 10
            elif i == 1:
                mask = (freqso > 20) & (freqso < 1000)
            elif i not in [0, 2, 4]:
                mask = freqso > 10
            else:
                mask = np.arange(len(freqso))
            label = labels[n]
        plt.plot(np.log10(freqso)[mask], np.log10(np.abs(normalizationo * plot_M.dot(w_nfo)[i]))[mask], c[i]+'-', label=label)
        plt.plot(np.log10(freqso)[mask], np.log10(np.abs(normalizationo * plot_M.dot(w_nfo)[i]))[mask], c[i]+'o')
        if i == 1:
            plt.plot(np.log10(freqso)[mask], np.log10(Bnu(2.75, freqso * 1e9))[mask], c[i]+'--')
    plt.legend(loc='upper left')
    plt.xlim([-2.3, 4])
    plt.ylim([-1.5, 5])
    plt.show()

def fit_power(freq, amp, relative_error=None, plot=False, log=False):
    if relative_error is None:
        relative_error = np.ones_like(amp)
    b = np.log10(amp)
    A = np.ones((len(freq), 2))
    A[:, 0] = np.log10(freq)
    Ni = 1. / relative_error**2
    AtAi = la.inv((A.transpose() * Ni).dot(A))
    x = AtAi.dot((A.transpose() * Ni).dot(b))
    error = (A.dot(x) - b) * Ni**.5
    noise = la.norm(error) / (len(freq) - 2)**.5
    if plot:
        if log:
            plt_x = 10**A[:, 0]
        else:
            plt_x = A[:, 0]
        plt.errorbar(plt_x, b, yerr=noise * relative_error, fmt='bo')
        plt.plot(plt_x, A.dot(x), 'g-')
        plt.show()
    return x, np.diagonal(AtAi)**.5 * noise

for m, plot_M in enumerate([final_M]):
    for n, i in enumerate(range(n_principal)):#n, i same thing, just a relic
        if i < 3 and i != cmb_principal:
            mask = freqso < 10
        elif i == cmb_principal:
            mask = (freqso > 20) & (freqso < 500)
        elif i == 3:
            mask = freqso > 150
        elif i == 4:
            mask = freqso > 50
        else:
            mask = (freqso > 10) & (freqso < 100)
        label = labels[n]
        local_x = freqso[mask]
        local_norm = np.log10(normalizationo)[mask]
        local_y = np.log10(np.abs(normalizationo * plot_M.dot(w_nfo)[i]))[mask]
        plt.plot(np.log10(local_x), local_y, c[i]+'-', label=label)
        plt.plot(np.log10(local_x), local_y, c[i]+'o')

        if i == 0:
            pars = [0,0]
            for j, sync_mask in enumerate([local_x < 0.5, local_x > 0.3]):
                pars[j], err = fit_power(local_x[sync_mask], 10.**local_y[sync_mask])
                print labels[i], pars[j], err
            plt.plot(np.log10(local_x), np.min([pars[j][1] + pars[j][0] * np.log10(local_x) for j in range(2)], axis=0), c[i]+'--')
        elif i == 1:
            # plt.plot(np.log10(local_x), np.log10(1e18 * Bnu(2.75, freqso * 1e9))[mask], c[i]+'--')
            # plt.plot(np.log10(local_x), np.log10(10**15.85 * freqso * Bnu(2.75, freqso * 1e9))[mask], c[i]+'--')
            plt.plot(np.log10(local_x), np.log10(10**16.64 * freqso**.65 * Bnu(2.75, freqso * 1e9))[mask], c[i]+'--')
        elif i == 3 or i == 4:
            # plt.plot(np.log10(local_x), -.4 + np.log10(1e-13 * dust_spectrum(15.72, 2.70, freqso * 1e9))[mask], c[i]+'--')
            # plt.plot(np.log10(local_x), -12.7 + np.log10(1e-13 * dust_spectrum(15.72, 3.70, freqso * 1e9))[mask], c[i]+'--')

            best_Tdust = 5
            best_beta = 1
            amp = 1
            std = np.std(np.log10(dust_spectrum(best_Tdust, best_beta, local_x * 1e9)) - local_y)
            for Tdust in np.arange(5, 30, .01):
                for beta in np.arange(1, 4, .01):
                    log_ratio = np.log10(dust_spectrum(Tdust, beta, local_x * 1e9)) - local_y
                    if np.std(log_ratio) < std:
                        std = np.std(log_ratio)
                        best_Tdust = Tdust
                        best_beta = beta
                        best_offset = -np.mean(log_ratio)
            print labels[i], best_Tdust, best_beta
            plt.plot(np.log10(local_x), best_offset + np.log10(dust_spectrum(best_Tdust, best_beta, local_x * 1e9)), c[i]+'--')
            # plt.plot(np.log10(local_x), -12.7 + np.log10(10**-15.5 * dust_spectrum(15.72, 3.90, freqso * 1e9))[mask], c[i]+'--')


            # plt.plot(np.log10(local_x), .12 - 1.2 * np.log10(local_x), c[i]+'--')
        # elif i == 4:
        #     # plt.plot(np.log10(local_x), .2 + np.log10(dust_spectrum(9.15, 1.67, freqso * 1e9))[mask], c[i]+'--')
        #     plt.plot(np.log10(local_x), -1.1 + np.log10(dust_spectrum(9.15*1.72, 1.67, freqso * 1e9))[mask], c[i]+'--')
        elif i == 5:
            pars = fit_power(local_x, 10.**local_y)
            plt.plot(np.log10(local_x), pars[0][1] + pars[0][0] * np.log10(local_x), c[i]+'--')
            print labels[i], pars
    plt.legend(loc='upper left')
    plt.xlim([-2.3, 4])
    plt.ylim([-1.5, 5])
    plt.show()

# np.savez(result_filename.replace('result_', 'GSM_'), f=freqso, W=np.transpose(final_M.dot(w_nfo)), M=np.transpose(la.inv(final_M)).dot(x_ni))

#############################
#####high res version######
#############################

high_res_nside = 1024
high_res = 14./60*np.pi/180.
low_res = .8*np.pi/180.
high_f_file = np.load('/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(high_res_nside, high_res, 3*np.pi/180., True, False, 3.0))
low_f_file = np.load('/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(high_res_nside, low_res, 3*np.pi/180., True, False, 3.0))

# high_f
high_f_principals = np.array([1, 3, 4, 5])
high_f_mask = np.array([(f in high_f_file['freqs']) and (f > 20 and f <= 3e3) for f in freqso])
high_f_data = high_f_file['idata'][np.array([f in freqso[high_f_mask] for f in high_f_file['freqs']])] / np.load(result_filename)['normalization'][high_f_mask][:, None]

high_res_x_ni = np.zeros((n_principal, 12 * high_res_nside**2))
for high_f_f_list, high_f_map_mask in zip(find_regions(np.isnan(high_f_data))[0], find_regions(np.isnan(high_f_data))[1]):
    A = final_w_nf.transpose()[np.ix_(high_f_mask, high_f_principals)][high_f_f_list]
    b = high_f_data[np.ix_(high_f_f_list, high_f_map_mask)]
    high_res_x_ni[np.ix_(high_f_principals, high_f_map_mask)] = la.inv(np.transpose(A).dot(A)).dot(A.transpose().dot(b))

    # local_fit = A.dot(high_res_x_ni[np.ix_(high_f_principals, high_f_map_mask)])
    # for f, bb in enumerate(b):
    #     plt_data = np.zeros(12 * high_res_nside**2)
    #     plt_data[high_f_map_mask] = local_fit[f]
    #     hpv.mollview(np.log10(plt_data), nest=True, sub=(len(b),2,2*f+1))
    #     plt_data[high_f_map_mask] = bb
    #     hpv.mollview(np.log10(plt_data), nest=True, sub=(len(b),2,2*f+2))
    # plt.show()

high_res_fit = np.transpose(final_w_nf).dot(high_res_x_ni)

#low_f
low_f_principals = np.array([0, 2])
low_f_mask = np.array([f in [5, 8] for f in range(len(freqso))])#np.array([(f in low_f_file['freqs']) and (f <= 20) for f in freqso])
low_f_data = low_f_file['idata'][np.array([f in freqso[low_f_mask] for f in low_f_file['freqs']])] / np.load(result_filename)['normalization'][low_f_mask][:, None]

A = np.transpose(final_w_nf)[np.ix_(low_f_mask, low_f_principals)]
b = low_f_data - [hp.reorder(hp.smoothing(hp.reorder(d, n2r=True), fwhm=low_res), r2n=True) for d in high_res_fit[low_f_mask]]
high_res_x_ni[low_f_principals] = la.inv(np.transpose(A).dot(A)).dot(A.transpose().dot(b))

#plot all components
for i in range(n_principal):
    qaz = np.arcsinh(high_res_x_ni[i])
    hpv.mollview(qaz, sub=(2, 3, i+1), nest=True, min=np.percentile(qaz, 2), max=np.percentile(qaz, 98));
plt.show()

w_estimates = si.interp1d(np.log10(freqso), np.arcsinh(final_w_nf * normalizationo), axis=-1, bounds_error=False)
sys.exit(0)
###########
#movie
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
for plot_i, logfreq in enumerate(np.arange(-2, 4., .03)):
    pltdata = np.arcsinh(np.transpose(high_res_x_ni).dot(np.sinh(w_estimates(logfreq))))
    if logfreq < 0:
        title = '%.1f MHz'%(1e3 * 10**logfreq)
    elif logfreq < 3:
        title = '%.1f GHz'%(10**logfreq)
    else:
        title = '%.1f THz'%(1e-3 * 10**logfreq)
    hpv.mollview(pltdata, nest=True, title=title, min=np.percentile(pltdata, 1), max=np.percentile(pltdata, 99), cbar=False)
    plt.savefig('/home/omniscope/gif_dir/%04i.png'%plot_i)
    plt.clf()

sys.exit(0)


post_final_M = np.eye(n_principal)
post_final_M[-1, -3:-1] = [-0.06463659, -0.09126348]#[-0.67153722, -0.24290523]
post_final_M.dot(final_w_nf)

for m, plot_M in enumerate([post_final_M.dot(final_M)]):

    for n, i in enumerate(range(n_principal)):
        if m == 0:
            mask = np.arange(len(freqso))
            label = 'Component ' + str(i)
        else:
            if i < 3 and i != cmb_principal:
                mask = freqso < 10
            elif i == 1:
                mask = (freqso > 20) & (freqso < 1000)
            elif i not in [0, 2, 4]:
                mask = freqso > 10
            else:
                mask = np.arange(len(freqso))
            label = labels[n]
        plt.plot(np.log10(freqso)[mask], np.log10(np.abs(normalizationo * plot_M.dot(w_nfo)[i]))[mask], c[i]+'-', label=label)
        plt.plot(np.log10(freqso)[mask], np.log10(np.abs(normalizationo * plot_M.dot(w_nfo)[i]))[mask], c[i]+'o')
    plt.legend(loc='upper left')
    plt.xlim([-2.3, 4])
    plt.ylim([-1.5, 5])
    plt.show()


sys.exit(0)



a = 1.216
b = 0#0.19
d = -.16
for n, e in enumerate(np.arange(-.65, -.55, .005)):
    qazM = np.eye(n_principal)
    qazM[5, 3] = a
    qazM[3, 4] = b
    qazM[4, 5] = d
    qazM[4, 0] = e
    final_M = qazM.dot(semifinal_M) * la.norm(la.inv(qazM.dot(semifinal_M)).transpose().dot(x_ni), axis=-1)[:, None]

    plt.subplot(4, 5, n+1)
    c = ['b', 'g', 'r', 'c', 'm', 'k']
    for i in range(n_principal):
        if i < 3 and i != cmb_principal:
            mask = freqs < 30
        elif i not in [0, 2, 4]:
            mask = freqs > 2
        else:
            mask = np.arange(len(freqs))
        plt.plot(np.log10(freqs)[mask], np.log10(np.abs(normalization * final_M.dot(w_nf)[i]))[mask], c[i]+'-')
        plt.plot(np.log10(freqs)[mask], np.log10(np.abs(normalization * final_M.dot(w_nf)[i]))[mask], c[i]+'o')
    plt.title(e)
plt.show()





######################################################
##quick example of using eigen values in w_nf to search for modes that are limited in frequency range
##as I shrink the range of frequencies, the number of non-zero eigen values decreases
eigen_values = np.zeros((n_f, n_principal))
for f_end in range(n_f):
    tmp_w_nf = w_nf[:, :f_end+1]
    eigen_values[f_end], evector = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))
plt.subplot(1, 2, 1)
plt.imshow(eigen_values, interpolation='none')

eigen_values = np.zeros((n_f, n_principal))
for f_start in range(n_f):
    tmp_w_nf = w_nf[:, f_start:]
    eigen_values[f_start], evector = la.eigh(tmp_w_nf.dot(np.transpose(tmp_w_nf)))
plt.subplot(1, 2, 2)
plt.imshow(eigen_values, interpolation='none')
plt.show()
