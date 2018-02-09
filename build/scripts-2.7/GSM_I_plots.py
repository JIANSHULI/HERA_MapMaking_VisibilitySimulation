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

###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
mother_nside = 128
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 3.6 * np.pi / 180.#5 * np.pi / 180.
edge_width = 3. * np.pi / 180.
remove_cmb = True
I_only = True
version = 3.0


n_principal_range = range(5, 10)
error_weighting = 'remove_pt'#'none'#'inv_error'#'remove_pt'

include_visibility = False
vis_Qs = ["q0AL_*_abscal", "q0C_*_abscal", "q1AL_*_abscal", "q2AL_*_abscal", "q2C_*_abscal", "q3AL_*_abscal", "q4AL_*_abscal"]  # L stands for lenient in flagging
datatag = '_2016_01_20_avg_unpol'
vartag = '_2016_01_20_avg_unpol'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
vis_tags = []
for vis_Q in vis_Qs:
    filenames = glob.glob(datadir + vis_Q + '_xx*' + datatag)
    vis_tags = vis_tags + [os.path.basename(fn).split('_xx')[0] for fn in filenames]

data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(mother_nside, smoothing_fwhm, edge_width, remove_cmb, not I_only, version)
print data_file_name


data_file = np.load(data_file_name)
freqs = np.concatenate((data_file['freqs'][:10], data_file['freqs'][11:-2]))
idata = np.concatenate((data_file['idata'][:10], data_file['idata'][11:-2]))


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
#normalization
########################################
data_file2 = np.load('/mnt/data0/omniscope/polarized foregrounds/result_25+4_nside_64_smooth_8.73E-02_edge_5.24E-02_rmvcmb_1_UV0_v3.0_principal_6_step_1.00_err_remove_pt.npz')
# data_file2 = np.load('/mnt/data0/omniscope/polarized foregrounds/result_25+4_nside_128_smooth_6.28E-02_edge_5.24E-02_rmvcmb_1_UV0_v3.0_principal_6_step_1.00_err_remove_pt.npz')
normalization = data_file2['normalization']
normalization[freqs < 20] = K_RJ2MJysr(normalization[freqs < 20], freqs[freqs < 20] * 1e9)
normalization[(freqs >= 20) & (freqs < 500)] = K_CMB2MJysr(normalization[(freqs >= 20) & (freqs < 500)], freqs[(freqs >= 20) & (freqs < 500)] * 1e9)


print
plt.plot(np.log10(freqs), np.log10(normalization * la.norm(np.transpose(data_file2['w_nf']).dot(data_file2['x_ni']), axis=-1)), 'b-')
for i in range(len(freqs)):
    plt.plot(np.log10(freqs[i]), np.log10((normalization * la.norm(np.transpose(data_file2['w_nf']).dot(data_file2['x_ni']), axis=-1))[i]), 'bo', fillstyle='none', markersize=(100. * np.sum(~np.isnan(idata[i])) / idata.shape[1])**.5)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Surface brightness (MJy/sr)')
plt.xlim([-2.2, 3.9])
plt.ylim([0.6, 4.6])
plt.show()


##start plotting
cmap = cm.gist_rainbow_r
cmap.set_under('w')
cmap.set_bad('gray')
for i, f in enumerate(freqs):

    plot_data = np.arcsinh(idata[i]) / np.log10(np.e)
    if f < 1:
        title = '%.1f MHz'%(1e3 * f)
    elif f < 1e3:
        title = '%.1f GHz'%(f)
    else:
        title = '%.1f THz'%(f * 1e-3)
    hpv.mollview(plot_data, nest=True, sub=(6, 5, i + 1), min=np.percentile(plot_data[~np.isnan(plot_data)], 2), max=np.percentile(plot_data[~np.isnan(plot_data)], 98), cmap=cmap, title=title, cbar=False)
plt.show()