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
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.3fGHz I, n%i"%(f, (len(plot_d)/12)**.5))

            iplot += 1
            plot_d = pol_frac(d[0], d[1], d[2])
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.3fGHz pol frac, n%i"%(f, (len(plot_d)/12)**.5))

            iplot += 1
            hpv.mollview(pol_angle(d[0], d[1], d[2]), sub=(nrow, ncol, iplot), nest=True, cmap=cm.hsv, title="%.3fGHz angle, n%i"%(f, (len(plot_d)/12)**.5))
        elif d.ndim== 1:
            iplot += 1
            plot_d = np.log10(d)
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.3fGHz I, n%i"%(f, (len(plot_d)/12)**.5))
            iplot += 2
        else:
            raise ValueError("Shape problem.")
    plt.show()

def equirectangular2heapix(data, nside, data_x=None, data_y=None, nest=True, manual_phi_correction=0):
    if data_x is None:
        delx = 2*np.pi / (data.shape[1] - 1)
        data_x = np.arange(0, 2*np.pi+delx/100., delx)
    if data_y is None:
        dely = np.pi / (data.shape[0] - 1)
        data_y = np.arange(np.pi, -dely/100., -dely)
    if data.shape != (len(data_y), len(data_x)):
        raise ValueError("Input shape mismatch between %s and (%i, %i)"%(data.shape, len(data_y), len(data_x)))
    inter_f = si.interp2d(sorted(data_x), sorted(data_y), data[np.ix_(np.argsort(data_y), np.argsort(data_x))])

    result = np.empty(12*nside**2, dtype=data.dtype)

    heal_thetas, heal_phis = hpf.pix2ang(nside, range(12*nside**2), nest=nest)
    unique_heal_thetas = np.unique(heal_thetas)

    for heal_theta in unique_heal_thetas:
        theta_mask = heal_thetas == heal_theta

        #doing some complicated juggling bc interp function automatically sort the list input and output according to that implicitly re-arranged inuput list
        qaz_phis = (heal_phis[theta_mask] + manual_phi_correction) % (np.pi*2)
        qaz = np.zeros_like(heal_phis[theta_mask])
        qaz[np.argsort(qaz_phis)] = inter_f(np.sort(qaz_phis), heal_theta).flatten()

        result[theta_mask] = qaz
    #     if np.abs(heal_theta - np.pi/2.) < 5*np.pi/180.:
    #         print np.isnan(qaz).all()
    # print np.isnan(data).all()
    # print data_x
    # print data_y
    return result

def ud_grade(m, nside, nest=True):
    if nside != hpf.npix2nside(len(m)):
        if nest:
            order_in = 'NESTED'
        else:
            order_in = 'RING'

        bad_mask = (np.isnan(m) | np.isinf(m) | (m == 0))
        if bad_mask.all():
            return np.zeros(hpf.nside2npix(nside)) * np.nan
        bad_mask = hpf.ud_grade(bad_mask.astype('float'), nside, order_in=order_in, pess=True) > 0

        result = hpf.ud_grade(m, nside, order_in=order_in, pess=True)
        result[bad_mask] = np.nan
        return result
    return m

def smoothing(m, fwhm, nest=True):
    if fwhm <= 0:
        return m
    nside = hpf.npix2nside(len(m))
    if nest:
        return hps.smoothing(m[hpf.ring2nest(nside, np.arange(hpf.nside2npix(nside)))], fwhm=fwhm)[hpf.nest2ring(nside, np.arange(hpf.nside2npix(nside)))]
    else:
        return hps.smoothing(m, fwhm=fwhm)

def preprocess(m, final_nside, nest=True, fwhm=0, edge_width=0, smooth_thresh=1e-2):
    #fwhm and edge_witdh are both in radians
    #smooth_thresh decides what relative error is accepted when smoothing smears 0s into edge data points

    result = np.copy(m)

    #create initial mask
    bad_mask = np.isnan(result) | (result == 0) | np.isinf(result)

    #expand mask to push out edge by edge_width
    edge_expand_bad_mask = smoothing(bad_mask, fwhm=edge_width, nest=nest) > .25
    bad_mask = bad_mask | edge_expand_bad_mask

    #smooth
    result[bad_mask] = 0
    result = smoothing(result, fwhm=fwhm, nest=nest)
    smooth_bad_mask = smoothing(bad_mask, fwhm=fwhm, nest=nest) > smooth_thresh
    bad_mask = bad_mask | smooth_bad_mask
    result[bad_mask] = np.nan

    #regrid
    result = ud_grade(result, final_nside, nest=nest)
    return result

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
###########################
###OVER ALL PARAMETERS
###########################
###########################
extract_nside = 512
mother_nside = 32
mother_npix = hpf.nside2npix(mother_nside)
smoothing_fwhm = 10. * np.pi / 180.
edge_width = 10. * np.pi / 180.
remove_cmb = True

dataset = {}
###1.4GHz###
stockert = fit.read("/home/omniscope/data/polarized foregrounds/stocker_villa_elisa.bin")[0]
stockert[:, :-1] = np.roll(stockert[:, :-1], 720, axis=1)[:, ::-1]
stockert[:, -1] = stockert[:, 0]
stockert = equirectangular2heapix(stockert, extract_nside)

reich_q = equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.q.lb.fits")[0], extract_nside, data_x=(np.pi+np.arange(np.pi * 2 + 1e-9, 0, -np.pi/720))%(np.pi*2+1e-9))
reich_u = -equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.u.lb.fits")[0], extract_nside, data_x=(np.pi+np.arange(np.pi * 2 + 1e-9, 0, -np.pi/720))%(np.pi*2+1e-9))#reich is in IAU convention whose U is minus sign the CMB healpix convention

dataset[1.42] = np.array([
            stockert, #merge_map([chipass, stockert], verbose=True, renormalize=True),
            reich_q, #merge_map([drao[1], elisa[1]]),
            reich_u, #merge_map([drao[2], elisa[2]]),
        ]) / 1.e3

###########################
###S-PASS 9' 2.3GHz 224MHz BW
#########################
dataset[2.3] = np.array([fit.read('/home/omniscope/data/polarized foregrounds/spass_hmap_m_1111_%s.fits'%IQU)['UNKNOWN1'].flatten()[hpf.nest2ring(1024, range(hpf.nside2npix(1024)))] for IQU in ['I', 'Q', 'U']])
dataset[2.3][2] = -dataset[2.3][2] # IAU convention to CMB convention
###########################
###CMB######
###############

#####wmap
dataset[22.8] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_K_v5.fits")[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3

plot_dataset(dataset)
