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

#it's a pretty bad crime to not diffretiate file name based on what data set are included...
data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i.npz'%(mother_nside, smoothing_fwhm, edge_width, remove_cmb)


if os.path.isfile(data_file_name):
    raise Exception(data_file_name + " already exist.")
else:
    ns = mother_nside * 2
    while ns <= extract_nside:
        larger_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i.npz'%(ns, smoothing_fwhm, edge_width, remove_cmb)
        if os.path.isfile(larger_file_name):
            print 'Found ' + larger_file_name
            data_file = np.load(larger_file_name)
            freqs = data_file['freqs']
            idata = [ud_grade(m, mother_nside) for m in data_file['idata']]
            qdata = [ud_grade(m, mother_nside) for m in data_file['qdata']]
            udata = [ud_grade(m, mother_nside) for m in data_file['udata']]
            np.savez(data_file_name, freqs=freqs, idata=idata, qdata=qdata, udata=udata)
            sys.exit(0)
        ns *= 2
###########################
###parkes 85 and 150mhz
#########################

parkes_85 = fit.read("/home/omniscope/data/polarized foregrounds/parkes_85mhz.bin")[0]
parkes_85[:, :-1] = np.roll(parkes_85[:, :-1], 180, axis=1)[:, ::-1]
parkes_85[:, -1] = parkes_85[:, 0]
parkes_85[parkes_85 > 1.4e4] = -1e-9
parkes_85[parkes_85 <= 0] = -1e-9
parkes_85 = equirectangular2heapix(parkes_85, extract_nside)
parkes_85[parkes_85 <= 0] = np.nan

parkes_150 = fit.read("/home/omniscope/data/polarized foregrounds/parkes_150mhz.bin")[0]
parkes_150[:, :-1] = np.roll(parkes_150[:, :-1], 180, axis=1)[:, ::-1]
parkes_150[:, -1] = parkes_150[:, 0]
parkes_150[parkes_150 > 7.e3] = -1e-9
parkes_150[parkes_150 <= 0] = -1e-9
parkes_150 = equirectangular2heapix(parkes_150, extract_nside)
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
stockert = equirectangular2heapix(stockert, extract_nside)

elisa = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/Elisa_POL_%s.bin"%key)[0], extract_nside) for key in ['I', 'Q', 'U']])
elisa[elisa > 1000] = np.nan

drao = np.array([equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/DRAO_POL_%s.bin"%key)[0], extract_nside) for key in ['I', 'Q', 'U']])
drao[drao > 1000] = np.nan

reich_q = equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.q.lb.fits")[0], extract_nside, data_x=(np.pi+np.arange(np.pi * 2 + 1e-9, 0, -np.pi/720))%(np.pi*2+1e-9))
reich_u = -equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.u.lb.fits")[0], extract_nside, data_x=(np.pi+np.arange(np.pi * 2 + 1e-9, 0, -np.pi/720))%(np.pi*2+1e-9))#reich is in IAU convention whose U is minus sign the CMB healpix convention

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
###S-PASS 9' 2.3GHz 224MHz BW
#########################
spass = {}
spass[2.3] = np.array([fit.read('/home/omniscope/data/polarized foregrounds/spass_hmap_m_1111_%s.fits'%IQU)['UNKNOWN1'].flatten()[hpf.nest2ring(1024, range(hpf.nside2npix(1024)))] for IQU in ['I', 'Q', 'U']])
spass[2.3][2] = -spass[2.3][2] # IAU convention to CMB convention
if plot_individual:
    plot_dataset(spass)
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
#     d = equirectangular2heapix(d, extract_nside)
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
    spass,
    haslam,
    wmap_iqu,
    planck_iqu,
    #parkes,
    #iras,
    iris,
    akari,
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


