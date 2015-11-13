__author__ = 'omniscope'

import numpy as np
import scipy.interpolate as si
import fitsio as fit
import healpy.visufunc as hpv
import healpy.pixelfunc as hpf
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

def pol_frac(I, Q, U, V=0):
    return (Q**2 + U**2 + V**2)**.5 / I

def pol_angle(I, Q, U, V=0):
    return np.arctan2(Q, U) / 2

def plot_dataset(data_set):

    if len(data_set) <= 5:
        nrow = len(data_set)
        ncol = 3

    elif len(data_set) <= 10:
        nrow = (len(data_set) + 1) / 2
        ncol = 6

    else:
        nrow = (len(data_set) + 2) / 3
        ncol = 9

    iplot = 0
    for f in sorted(data_set.keys()):
        d = data_set[f]
        if d.ndim == 2:
            iplot += 1
            plot_d = np.log10(d[0])
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.2fGHz I, n%i"%(f, (len(plot_d)/12)**.5))

            iplot += 1
            plot_d = pol_frac(d[0], d[1], d[2])
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.2fGHz Q, n%i"%(f, (len(plot_d)/12)**.5))

            iplot += 1
            hpv.mollview(pol_angle(d[0], d[1], d[2]), sub=(nrow, ncol, iplot), nest=True, cmap=cm.hsv, title="%.2fGHz U, n%i"%(f, (len(plot_d)/12)**.5))
        elif d.ndim== 1:
            iplot += 1
            plot_d = np.log10(d)
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.2fGHz I, n%i"%(f, (len(plot_d)/12)**.5))
            iplot += 2
        else:
            raise ValueError("Shape problem.")
    plt.show()

def equirectangular2heapix(data, nside, data_x=None, data_y=None, nest=True):
    if data_x is None:
        delx = 2*np.pi / (data.shape[1] - 1)
        data_x = np.arange(0, 2*np.pi+delx/100., delx)
    if data_y is None:
        dely = np.pi / (data.shape[0] - 1)
        data_y = np.arange(np.pi, -dely/100., -dely)
    if data.shape != (len(data_y), len(data_x)):
        raise ValueError("Input shape mismatch between %s and (%i, %i)"%(data.shape, len(data_y), len(data_x)))
    inter_f = si.interp2d(data_x, data_y, data)

    result = np.empty(12*nside**2, dtype=data.dtype)

    heal_thetas, heal_phis = hpf.pix2ang(nside, range(12*nside**2), nest=nest)
    unique_heal_thetas = np.unique(heal_thetas)

    for heal_theta in unique_heal_thetas:
        theta_mask = heal_thetas == heal_theta

        #doing some complicated juggling bc interp function automatically sort the list input and output according to that implicitly re-arranged inuput list
        qaz_phis = heal_phis[theta_mask] % (np.pi*2)
        qaz = np.zeros_like(heal_phis[theta_mask])
        qaz[np.argsort(qaz_phis)] = inter_f(np.sort(qaz_phis), heal_theta).flatten()

        result[theta_mask] = qaz
    return result

def ud_grade(m, nside, nest=True):
    if nside != hpf.npix2nside(len(m)):
        if nest:
            order_in='NESTED'
        else:
            order_in='RING'
        bad_mask = hpf.ud_grade((~(np.isnan(m) | np.isinf(m))).astype('int'), nside, order_in=order_in, pess=True) <= 0
        result = hpf.ud_grade(m, nside, order_in=order_in, pess=True)
        result[bad_mask] = np.nan
        return result
    return m

def merge_map(maps, nside=None, nest=True, verbose=False, renormalize=False):
    if nside is None:
        nside = 4096
        for m in maps:
            nside = min(nside, hpf.npix2nside(len(m)))



    filled_mask = np.zeros(hpf.nside2npix(nside), dtype=bool)
    result = np.zeros(hpf.nside2npix(nside), dtype=maps[0].dtype)

    for m in maps:
        m = ud_grade(m, nside, nest=nest)
        #valid in m
        valid_mask = ~(np.isnan(m)|np.isinf(m))

        #pixels to be taken from m, earlier m takes priority and will not be over-written
        fill_mask = valid_mask&(~filled_mask)
        if verbose:
            print "%.1f%% valid"%(100. * np.sum(valid_mask) / len(valid_mask)),
            print "%.1f%% to be filled"%(100. * np.sum(fill_mask) / len(fill_mask))

        if renormalize:
            overlap_mask = valid_mask&filled_mask
            if overlap_mask.any():
                factor = m[overlap_mask].dot(result[overlap_mask]) / m[overlap_mask].dot(m[overlap_mask])
                if verbose:
                    print "renormalizing by ", factor
                m *= factor

        #fill pixel and mask
        result[fill_mask] = m[fill_mask]
        filled_mask[fill_mask] = True
    result[~filled_mask] = np.nan

    return result


plot_individual = False

###########################
###1.4G: DRAO+villa Elisa+CHIPASS+LAB
###########################

drao_elisa_iqu_syn = {}

chipass = fit.read("/home/omniscope/data/polarized foregrounds/lambda_chipass_healpix_r10.fits")['TEMPERATURE']
chipass[chipass <= 0] = np.nan
# lab = fit.read("/home/omniscope/data/polarized foregrounds/LAB_fullvel.fits")['TEMPERATURE'].flatten()
# lab = lab[hpf.nest2ring(int((len(lab)/12)**.5), range(len(lab)))]

stockert = fit.read("/home/omniscope/data/polarized foregrounds/stocker_villa_elisa.bin")[0]
stockert = np.roll(stockert, 720, axis=1)[:, ::-1]
stockert = equirectangular2heapix(stockert, 256)

elisa = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/Elisa_POL_%s.bin"%key)[0], 256) for key in ['I', 'Q', 'U']])
elisa[elisa > 1000] = np.nan

drao = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/DRAO_POL_%s.bin"%key)[0], 256) for key in ['I', 'Q', 'U']])
drao[drao > 1000] = np.nan

drao_elisa_iqu_syn[1.3945] = chipass
drao_elisa_iqu_syn[1.435] = elisa
drao_elisa_iqu_syn[1.41] = drao
drao_elisa_iqu_syn[1.42] = stockert
# drao_elisa_iqu_syn[1.4276] = lab

if plot_individual:
    plot_dataset(drao_elisa_iqu_syn)

all_1400 = {1.41: np.array([
    merge_map([chipass, stockert], verbose=True, renormalize=True),
    merge_map([drao[1], elisa[1]]),
    merge_map([drao[2], elisa[2]]),
])}

if plot_individual:
    plot_dataset(all_1400)

###########################
###CMB######
###############
T_CMB = 2.725
dipole_correction = {'I_Stokes': T_CMB, 'Q_Stokes': 0, 'U_Stokes': 0, 'I_STOKES': T_CMB, 'Q_STOKES': 0, 'U_STOKES': 0, 'TEMPERATURE': T_CMB, 'Q_POLARISATION': 0, 'U_POLARISATION': 0}
####
planck_iqu = {}
planck_iqu[30.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/LFI_SkyMap_030_1024_R2.01_full.fits")[key] + dipole_correction[key] for key in ['I_Stokes', 'Q_Stokes', 'U_Stokes']])
planck_iqu[44.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/LFI_SkyMap_044_1024_R2.01_full.fits")[key] + dipole_correction[key] for key in ['I_Stokes', 'Q_Stokes', 'U_Stokes']])
planck_iqu[70.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/LFI_SkyMap_070_1024_R2.01_full.fits")[key] + dipole_correction[key] for key in ['I_Stokes', 'Q_Stokes', 'U_Stokes']])
planck_iqu[100.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_100_2048_R2.02_full.fits")[key] + dipole_correction[key] for key in ['I_STOKES', 'Q_STOKES', 'U_STOKES']])
planck_iqu[143.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_143_2048_R2.02_full.fits")[key] + dipole_correction[key] for key in ['I_STOKES', 'Q_STOKES', 'U_STOKES']])
planck_iqu[217.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_217_2048_R2.02_full.fits")[key] + dipole_correction[key] for key in ['I_STOKES', 'Q_STOKES', 'U_STOKES']])
planck_iqu[353.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_353_2048_R2.02_full.fits")[key] + dipole_correction[key] for key in ['I_STOKES', 'Q_STOKES', 'U_STOKES']])
planck_iqu[545.0] = fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_545_2048_R2.02_full.fits")['I_STOKES']
planck_iqu[857.0] = fit.read("/home/omniscope/data/polarized foregrounds/HFI_SkyMap_857_2048_R2.02_full.fits")['I_STOKES']
if plot_individual:
    plot_dataset(planck_iqu)

#####
wmap_iqu = {}
wmap_iqu[22.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_K_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']])
wmap_iqu[30.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_Ka_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']])
wmap_iqu[40.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_Q_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']])
wmap_iqu[60.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_V_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']])
wmap_iqu[90.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_W_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']])
if plot_individual:
    plot_dataset(wmap_iqu)

# ###WMAP POL 22G
# cmb_iqu_syn = {}
# cmb_iqu_syn[22.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_mcmc_base_k_synch_%s_9yr_v5.fits"%key)['BESTFIT'] for key in ['temp', 'stk_q', 'stk_u']])
#
# ###PLANCK POL 30G
# cmb_iqu_syn[30.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/COM_CompMap_Synchrotron-commander_0256_R2.00.fits")['I_ML']]
#                                 + [fit.read("/home/omniscope/data/polarized foregrounds/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits")[key] for key in ['Q_ML_FULL', 'U_ML_FULL']])
# plot_dataset(cmb_iqu_syn)

#########################
###mother file#########################
#########################
motherfile = {}
motherfile_data = np.fromfile("/home/omniscope/data/polarized foregrounds/motherfile_3145728_16_float64.bin", dtype='float64').reshape((3145728, 16))[hpf.nest2ring(512, range(3145728))]
motherfile[.045] = motherfile_data[:, -9]
motherfile[2.33] = motherfile_data[:, -1]
motherfile[.035] = motherfile_data[:, -11]
motherfile[.022] = motherfile_data[:, -13]
motherfile[.082] = motherfile_data[:, -2]
motherfile[.010] = motherfile_data[:, -16]
if plot_individual:
    plot_dataset(motherfile)

#########################
###Haslam#########################
#########################
haslam = fit.read("/home/omniscope/data/polarized foregrounds/haslam408_dsds_Remazeilles2014.fits")['TEMPERATURE'].flatten()
haslam = {.408: haslam[hpf.nest2ring(int((len(haslam)/12)**.5), range(len(haslam)))]}
if plot_individual:
    plot_dataset(haslam)


####PNM+87GB
#plt.imshow(np.log10(fit.read("/home/omniscope/data/polarized foregrounds/PMN+87GB.fits")[0,0]));plt.show()


new_mother = {}
mother_nside = 256
for dict in [motherfile, all_1400, haslam, wmap_iqu, planck_iqu]:
    for f, data in dict.iteritems():
        if data.shape[-1] == hpf.nside2npix(mother_nside):
            new_mother[f] = np.copy(data)
        else:
            if data.ndim == 2:
                new_mother[f] = np.array([ud_grade(d, mother_nside) for d in data])
            else:
                new_mother[f] = np.array(ud_grade(data, mother_nside))
plot_dataset(new_mother)