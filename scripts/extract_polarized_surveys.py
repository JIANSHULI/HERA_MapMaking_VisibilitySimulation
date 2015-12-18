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
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.3fGHz Q, n%i"%(f, (len(plot_d)/12)**.5))

            iplot += 1
            hpv.mollview(pol_angle(d[0], d[1], d[2]), sub=(nrow, ncol, iplot), nest=True, cmap=cm.hsv, title="%.3fGHz U, n%i"%(f, (len(plot_d)/12)**.5))
        elif d.ndim== 1:
            iplot += 1
            plot_d = np.log10(d)
            plot_mask = ~(np.isnan(plot_d) | np.isinf(plot_d))
            hpv.mollview(plot_d, sub=(nrow, ncol, iplot), nest=True, min=np.percentile(plot_d[plot_mask], 5), max=np.percentile(plot_d[plot_mask], 95), title="%.3fGHz I, n%i"%(f, (len(plot_d)/12)**.5))
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

    nside = hpf.npix2nside(len(m))
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

plot_individual = False


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


if os.path.isfile(data_file_name):
    data_file = np.load(data_file_name)
    freqs = data_file['freqs']
    idata = data_file['idata']
    qdata = data_file['qdata']
    udata = data_file['udata']

else:
    ###########################
    ###parkes 85 and 150mhz
    #########################

    parkes_85 = fit.read("/home/omniscope/data/polarized foregrounds/parkes_85mhz.bin")[0]
    parkes_85[:, :-1] = np.roll(parkes_85[:, :-1], 180, axis=1)[:, ::-1]
    parkes_85[:, -1] = parkes_85[:, 0]
    parkes_85[parkes_85 > 1.4e4] = -1e-9
    parkes_85[parkes_85 <= 0] = -1e-9
    parkes_85 = equirectangular2heapix(parkes_85, mother_nside)
    parkes_85[parkes_85 <= 0] = np.nan

    parkes_150 = fit.read("/home/omniscope/data/polarized foregrounds/parkes_150mhz.bin")[0]
    parkes_150[:, :-1] = np.roll(parkes_150[:, :-1], 180, axis=1)[:, ::-1]
    parkes_150[:, -1] = parkes_150[:, 0]
    parkes_150[parkes_150 > 7.e3] = -1e-9
    parkes_150[parkes_150 <= 0] = -1e-9
    parkes_150 = equirectangular2heapix(parkes_150, mother_nside)
    parkes_150[parkes_150 <= 0] = np.nan

    parkes = {.085: parkes_85,
              .15: parkes_150,
              }
    if plot_individual:
        plot_dataset(parkes)
    ###########################
    ###1.4G: DRAO+villa Elisa+CHIPASS+LAB
    ###########################

    drao_elisa_iqu_syn = {}

    chipass = fit.read("/home/omniscope/data/polarized foregrounds/lambda_chipass_healpix_r10.fits")['TEMPERATURE']
    chipass[chipass <= 0] = np.nan
    # lab = fit.read("/home/omniscope/data/polarized foregrounds/LAB_fullvel.fits")['TEMPERATURE'].flatten()
    # lab = lab[hpf.nest2ring(int((len(lab)/12)**.5), range(len(lab)))]

    stockert = fit.read("/home/omniscope/data/polarized foregrounds/stocker_villa_elisa.bin")[0]
    stockert[:, :-1] = np.roll(stockert[:, :-1], 720, axis=1)[:, ::-1]
    stockert[:, -1] = stockert[:, 0]
    stockert = equirectangular2heapix(stockert, mother_nside)

    elisa = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/Elisa_POL_%s.bin"%key)[0], mother_nside) for key in ['I', 'Q', 'U']])
    elisa[elisa > 1000] = np.nan

    drao = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/DRAO_POL_%s.bin"%key)[0], mother_nside) for key in ['I', 'Q', 'U']])
    drao[drao > 1000] = np.nan

    reich_q = equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.q.lb.fits")[0], mother_nside)
    reich_u = equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.u.lb.fits")[0], mother_nside)

    drao_elisa_iqu_syn[1.3945] = chipass
    drao_elisa_iqu_syn[1.435] = elisa
    drao_elisa_iqu_syn[1.41] = drao
    drao_elisa_iqu_syn[1.42] = stockert
    # drao_elisa_iqu_syn[1.4276] = lab

    if plot_individual:
        plot_dataset(drao_elisa_iqu_syn)

    all_1400 = {
        1.42:
            np.array([
                stockert, #merge_map([chipass, stockert], verbose=True, renormalize=True),
                reich_q, #merge_map([drao[1], elisa[1]]),
                reich_u, #merge_map([drao[2], elisa[2]]),
            ]) / 1.e3,
        1.3945: chipass / 1.e3}

    if plot_individual:
        plot_dataset(all_1400)

    ###########################
    ###CMB######
    ###############
    T_CMB = 2.725 * (1 - remove_cmb) #if remove cmb, dont add cmb temp into wmap/planck
    dipole_correction = {'I_Stokes': T_CMB, 'Q_Stokes': 0, 'U_Stokes': 0, 'I_STOKES': T_CMB, 'Q_STOKES': 0, 'U_STOKES': 0, 'TEMPERATURE': T_CMB, 'Q_POLARISATION': 0, 'U_POLARISATION': 0}

    ####planck
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

    #####wmap
    wmap_iqu = {}
    wmap_iqu[22.8] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_K_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3
    wmap_iqu[33.0] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_Ka_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3
    wmap_iqu[40.7] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_Q_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3
    wmap_iqu[60.8] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_V_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3
    wmap_iqu[93.5] = np.array([fit.read("/home/omniscope/data/polarized foregrounds/wmap_band_iqumap_r9_9yr_W_v5.fits")[key] + dipole_correction[key] for key in ['TEMPERATURE', 'Q_POLARISATION', 'U_POLARISATION']]) / 1.e3
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
    # motherfile = {}
    # motherfile_data = np.fromfile("/home/omniscope/data/polarized foregrounds/motherfile_3145728_16_float64.bin", dtype='float64').reshape((3145728, 16))[hpf.nest2ring(512, range(3145728))]
    # for i in range(motherfile_data.shape[1]):
    #     motherfile[i] = motherfile_data[:, i]
    # plot_dataset(motherfile)

    motherfile = {}
    motherfile_data = np.fromfile("/home/omniscope/data/polarized foregrounds/motherfile_3145728_16_float64.bin", dtype='float64').reshape((3145728, 16))[hpf.nest2ring(512, range(3145728))]
    motherfile[.045] = motherfile_data[:, -9]
    motherfile[2.33] = motherfile_data[:, -1]
    # motherfile[.0345] = motherfile_data[:, -11]
    # motherfile[.408] = motherfile_data[:, -3]
    motherfile[.022] = motherfile_data[:, -13]
    motherfile[.82] = motherfile_data[:, -2]
    motherfile[.010] = motherfile_data[:, -16]
    if plot_individual:
        plot_dataset(motherfile)

    ###########################
    ###high freq THz
    #########################
    # iras = {}
    # for freq in [3, 5]:#, 12]:
    #     d = np.zeros((5401, 10801)) - 1e9
    #     d[(5401-3)/2:(5401-3)/2+301] = fit.read("/home/omniscope/data/polarized foregrounds/iras_%ithz.bin"%freq)[0]
    #     d[:, :-1] = np.roll(d[:, :-1], 5400, axis=1)[:, ::-1]
    #     d[:, -1] = d[:, 0]
    #     d[d > np.nanmax(d) * .8] = -1e9
    #     d[d <= 0] = -1e9
    #     d = equirectangular2heapix(d, mother_nside)
    #     d[d <= 0] = np.nan
    #     iras[freq * 1.e3] = d
    #
    #
    # if plot_individual:
    #     plot_dataset(iras)

    ############
    ###iris####
    ############
    iris = {}
    for i, freq in enumerate([3.e3, 5.e3, 12.e3, 25.e3]):
        iris[freq] = fit.read("/home/omniscope/data/polarized foregrounds/IRIS_nohole_%i_1024.fits"%(4-i))['f0'].flatten()[hpf.nest2ring(1024, range(hpf.nside2npix(1024)))]

    theta_g, phi_g = hpf.pix2ang(1024, range(12*1024**2), nest=True)
    theta_e, phi_e = hpr.Rotator(coord='ge')(theta_g, phi_g)
    iris_zodiacal_mask_widths = [0, .2, .7, .7]
    for i, freq in enumerate([3.e3, 5.e3, 12.e3, 25.e3]):
        iris[freq][np.abs(theta_e-np.pi/2) <= iris_zodiacal_mask_widths[i]] = np.nan

    if plot_individual:
        plot_dataset(iris)

    ############
    ####akari####
    ############
    akari = {}
    for i, freq in enumerate([3.e5/160, 3.e5/140, 3.e5/90]):#, 3.e5/65]):
        akari[freq] = fit.read("/home/omniscope/data/polarized foregrounds/akari_%s_1_4096.fits"%(['160', 'WideL', 'WideS', '65'][i]))['UNKNOWN1'].flatten()
        akari[freq][akari[freq] == 0] = np.nan
    ###common mask for akari scan patterns
    common_akari_mask = np.zeros(len(akari[akari.keys()[0]]), dtype='bool')
    for freq, data in akari.iteritems():
        common_akari_mask = common_akari_mask|np.isnan(data)
    for freq in akari.keys():
        akari[freq][common_akari_mask] = np.nan
    ###remove zodiacal light contaminated region for 90um (and 65um, but it's excluded anyways)
    theta_g, phi_g = hpf.pix2ang(4096, range(12*4096**2), nest=True)
    theta_e, phi_e = hpr.Rotator(coord='ge')(theta_g, phi_g)
    # theta_z, phi_z = hpr.Rotator(rot=(0, 0, 90))(theta_e, phi_e)
    akari_zodiacal_mask = np.abs(theta_e-np.pi/2) <= .45
    akari[3.e5/90][akari_zodiacal_mask] = np.nan

    if plot_individual:
        plot_dataset(akari)
    #########################
    ###Haslam#########################
    #########################
    haslam = fit.read("/home/omniscope/data/polarized foregrounds/haslam408_dsds_Remazeilles2014.fits")['TEMPERATURE'].flatten()
    haslam = {.408: haslam[hpf.nest2ring(int((len(haslam)/12)**.5), range(len(haslam)))]}
    if plot_individual:
        plot_dataset(haslam)


    ####PNM+87GB
    #plt.imshow(np.log10(fit.read("/home/omniscope/data/polarized foregrounds/PMN+87GB.fits")[0,0]));plt.show()


    #########################
    ###Create new mother#########################
    #########################


    new_mother = {}
    for dict in [
        motherfile,
        all_1400,
        haslam,
        wmap_iqu,
        planck_iqu,
        #parkes,
        #iras,
        ]:
        for f, data in dict.iteritems():
            print f
            sys.stdout.flush()

            if data.ndim == 2:
                new_mother[f] = np.array([preprocess(d, mother_nside, fwhm=smoothing_fwhm, edge_width=edge_width) for d in data])
            else:
                new_mother[f] = preprocess(data, mother_nside, fwhm=smoothing_fwhm, edge_width=edge_width)
    plot_dataset(new_mother)


    freqs = np.array(sorted(new_mother.keys()))
    idata = np.zeros((len(freqs), mother_npix))
    qdata = np.zeros((len(freqs), mother_npix))
    udata = np.zeros((len(freqs), mother_npix))
    for f, freq in enumerate(freqs):
        data = new_mother[freq]
        if data.ndim == 2:
            idata[f] = data[0]
            qdata[f] = data[1]
            udata[f] = data[2]
        else:
            idata[f] = data
            qdata[f] += np.nan
            udata[f] += np.nan

    np.savez(data_file_name, freqs=freqs, idata=idata, qdata=qdata, udata=udata)



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
addon_qdata = qdata[removed_mask]
addon_udata = udata[removed_mask]

freqs = freqs[~removed_mask]
idata = idata[~removed_mask]
qdata = qdata[~removed_mask]
udata = udata[~removed_mask]

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