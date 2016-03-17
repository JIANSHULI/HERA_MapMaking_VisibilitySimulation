__author__ = 'omniscope'

import numpy as np
import numpy.linalg as la
import sys
from matplotlib import cm
import healpy.pixelfunc as hpf
try:
    import healpy.visufunc as hpv
except:
    pass
import matplotlib.pyplot as plt
kB = 1.38065e-23
c = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / c)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * c / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / c)**2 * kB
    return  K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy


########################################
#load data
result_filename = '/mnt/data0/omniscope/polarized foregrounds/result_25+4_nside_128_smooth_6.28E-02_edge_5.24E-02_rmvcmb_1_UV0_v3.0_principal_6_step_1.00_err_remove_pt.npz'
f = np.load(result_filename)
w_nf = f['w_nf']#n_principal by frequency
x_ni = f['x_ni']#n_principal by pixel
freqs = f['freqs']#GHz
# ps_mask = f['ps_mask']
# x_ni *= (1-ps_mask)
n_f = len(freqs)
n_principal = len(w_nf)
nside = hpf.npix2nside(x_ni.shape[1])
########################################
normalization = f['normalization']
normalization[freqs < 20] = K_RJ2MJysr(normalization[freqs < 20], freqs[freqs < 20] * 1e9)
normalization[(freqs >= 20) & (freqs < 500)] = K_CMB2MJysr(normalization[(freqs >= 20) & (freqs < 500)], freqs[(freqs >= 20) & (freqs < 500)] * 1e9)
# plt.plot(np.log10(freqs), normalization)
# plt.show()

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
            hpv.mollview(plot_data, nest=True, sub=(3, n_principal, n + 1), cmap=cmap, title="%.4f"%la.norm(plot_data_lin))
        except:
            print "NEED HEALPY PACKAGE FOR PLOTTING!"
        plt.subplot(3, n_principal, n_principal + n + 1)
        plt.plot(np.log10(freqs), sign_flip * w_nf_local[n])
        plt.plot(np.log10(freqs), sign_flip * w_nf_local[n], 'r+')
        plt.ylim([-1, 1])
        plt.subplot(3, n_principal, 2 * n_principal + n + 1)
        plt.plot(np.log10(freqs), np.log10(np.abs(w_nf_local[n] * normalization * la.norm(plot_data_lin))))
        plt.plot(np.log10(freqs), np.log10(np.abs(w_nf_local[n] * normalization * la.norm(plot_data_lin))), 'r+')

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

###normal inclusive eigenvalue approach: not very effective
# eigenvalue_fractions = eigen_values / np.sum(eigen_values, axis=-1)[..., None]
# eigenvalue_cumu_fractions = np.cumsum(eigenvalue_fractions, axis=-1)
#
# for i in range(n_principal-1):
#     plt.subplot(2, n_principal - 1, i+1)
#     plt.imshow(np.log10(eigen_values[..., -i-1] / eigen_values[..., -i-2]), interpolation='none', vmax=3, vmin=1)
#     plt.title('ratio of %ist and %ist eigen'%(i+1, i+2))
#
#
# #calculate how many eigen modes it takes to explain 99%
# for j, thresh in enumerate([5e-2, 1e-2, 1e-3, 1e-4]):
#     n_modes = np.zeros((n_f, n_f), dtype='int')
#
#     for fstart in range(n_f):
#         for fend in range(n_f):
#             for i in range(n_principal - 1):
#                 if eigenvalue_cumu_fractions[fstart, fend, i] >= thresh:
#                     n_modes[fstart, fend] = n_principal - i
#                     break
#                 elif eigenvalue_cumu_fractions[fstart, fend, i] < thresh and eigenvalue_cumu_fractions[fstart, fend, i + 1] >= thresh:
#                     n_modes[fstart, fend] = n_principal - 1 - i
#                     break
#     plt.subplot(2, n_principal - 1, j+n_principal)
#     plt.imshow(n_modes, interpolation='none')
#     plt.title('number of modes to exclude only %.1e'%thresh)
# plt.show()
#
###STEP 2
###dominating eigenvalue approach: not very effective
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

#LAST STEP: cmb
x_ni_intermediate = np.transpose(la.inv(M)).dot(x_ni)
cmb_m = np.eye(n_principal)#remove foreground in CMB
cmb_principal = 1
non_cmb_principal_mask = np.array([False, False, False, True, True, True])#np.arange(n_principal) != cmb_principal
if abs(np.min(x_ni_intermediate[cmb_principal])) > np.max(x_ni_intermediate[cmb_principal]):
    plane_mask = (x_ni_intermediate[cmb_principal] <  -1.5*np.max(x_ni_intermediate[cmb_principal])) & (x_ni_intermediate[cmb_principal] > np.percentile(x_ni_intermediate[cmb_principal], .5))
else:
    plane_mask = (x_ni_intermediate[cmb_principal] >  1.5*abs(np.min(x_ni_intermediate[cmb_principal]))) & (x_ni_intermediate[cmb_principal] < np.percentile(x_ni_intermediate[cmb_principal], 99.5))
At_cmb = x_ni_intermediate[np.ix_(non_cmb_principal_mask, plane_mask)]
cmb_m[cmb_principal, non_cmb_principal_mask] = -la.inv(At_cmb.dot(np.transpose(At_cmb))).dot(At_cmb.dot(x_ni_intermediate[cmb_principal, plane_mask]))

M = la.inv(cmb_m.transpose()).dot(M)
M = M / la.norm(M.dot(w_nf), axis=-1)[:, None]
plot_components(M)


#STep 4: manual
manual_spike_freq_ranges = [[10,-10]] * n_principal
# manual_spike_freq_ranges[0] = [0, 30]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[2] = [0, 10]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[3] = [10, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[4] = [10, 1e6]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_freq_ranges[5] = [10, 1e3]#, [1, 1000], [10, 1e5], [-10, -10]]
manual_spike_ranges = [np.arange(len(freqs))[(np.array(freqs) >= freq_range[0]) & (np.array(freqs) <= freq_range[1])] for freq_range in manual_spike_freq_ranges]
w_nf_intermediate = M.dot(w_nf)
M1 = np.eye(n_principal)
for spike_principal, spike_fs in enumerate(manual_spike_ranges):
    if len(spike_fs) > 0:
        non_spike_fs = np.array([i for i in range(0, len(freqs)) if i not in spike_fs])
        non_spike_principals = np.array([i for i in range(n_principal) if i != spike_principal and i != cmb_principal])
        spike_A = np.transpose(w_nf_intermediate[np.ix_(non_spike_principals, non_spike_fs)])
        M1[spike_principal, non_spike_principals] = -la.inv(np.transpose(spike_A).dot(spike_A) + np.eye(len(non_spike_principals)) * .0001).dot(np.transpose(spike_A).dot(w_nf_intermediate[spike_principal, non_spike_fs]))
M1 = M1 / la.norm(M1.dot(M).dot(w_nf), axis=-1)[:, None]
plot_components(M1.dot(M))

# M2 = np.eye(n_principal)
# qaz = M1.dot(M).dot(w_nf)[-3:, 10:]
# M2[-3:, -3:] = la.eigh(qaz.dot(qaz.transpose()))[1].transpose()
# M2 = M2 / la.norm(M2.dot(M1).dot(M).dot(w_nf), axis=-1)[:, None]
# plot_components(M2.dot(M1).dot(M))
#

sys.exit(0)
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
