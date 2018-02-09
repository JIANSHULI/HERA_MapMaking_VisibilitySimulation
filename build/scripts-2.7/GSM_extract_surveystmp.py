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
PI = np.pi
TPI = PI * 2
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

def read_equirectangular(file_name, unit_scale=1, ):
    f = fit.read_header(file_name)
    data = fit.read(file_name)[0]
    data[:, :-1] = np.roll(data[:, :-1], (f['NAXIS1'] - 1) / 2, axis=1)[:, ::-1]
    data[:, -1] = data[:, 0]
    dmax = f['DATAMAX']
    dmin = f['DATAMIN']
    data[data > dmax] = -1e6 * dmax
    data[data < dmin] = -1e6 * dmax
    data = sv.equirectangular2heapix(data)
    data[data <= 0] = np.nan
    return unit_scale * data

plot_individual = False

DEGREE = PI / 180.
ARCMIN = DEGREE / 60.
###########################
###########################
###OVER ALL PARAMETERS
###########################
###########################
version = 3.0
I_only = True
mother_nside = 1024#64
mother_npix = hpf.nside2npix(mother_nside)
target_fwhm = .8 * DEGREE#3.6 * DEGREE#
min_edge_width = 3 * DEGREE
remove_cmb = True

#it's a pretty bad crime to not diffretiate file name based on what data set are included...
data_file_name = '/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(mother_nside, target_fwhm, min_edge_width, remove_cmb, not I_only, version)


if os.path.isfile(data_file_name):
    raise Exception(data_file_name + " already exist.")
else:
    ns = mother_nside * 2
    while ns <= 4096:
        larger_file_names = ['/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(ns, target_fwhm, min_edge_width, remove_cmb, not I_only, version)]
        if I_only:
            larger_file_names.append('/mnt/data0/omniscope/polarized foregrounds/data_nside_%i_smooth_%.2E_edge_%.2E_rmvcmb_%i_UV%i_v%.1f.npz'%(ns, target_fwhm, min_edge_width, remove_cmb, True, version))
        for larger_file_name in larger_file_names:
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

resolutions = {}


#########################
###Stockert 2.72#########################
#########################
data_x = np.arange(PI/2, -PI/2-.00001, -DEGREE/8)%(2*PI)
data_y = np.arange(PI/2+DEGREE*50, PI/2-DEGREE*50-.00001, -DEGREE/8)%(2*PI)
stockert11cm = fit.read('/home/omniscope/data/polarized foregrounds/fits27917_stockert11cm_180_100.bin')[0] * 1e-3
stockert11cm_header = fit.read_header('/home/omniscope/data/polarized foregrounds/fits27917_stockert11cm_180_100.bin')
stockert11cm[stockert11cm > stockert11cm_header['DATAMAX'] * 1e-3] = -1e6 * stockert11cm_header['DATAMAX']
stockert11cm[stockert11cm < stockert11cm_header['DATAMIN'] * 1e-3] = -1e6 * stockert11cm_header['DATAMAX']
stockert11cm = sv.equirectangular2heapix(stockert11cm, nside=1024, data_x=data_x, data_y=data_y)
stockert11cm[stockert11cm < 0] = np.nan
stockert11cm = {2.72: stockert11cm}
resolutions[2.72] = 21 * ARCMIN#data website uses 21, paper says 4.3
if plot_individual:
    plot_dataset(stockert11cm)

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
# motherfile[.022] = motherfile_data[:, -13]
# motherfile[.82] = motherfile_data[:, -2]
# motherfile[.010] = motherfile_data[:, -16]

# resolutions[.010] = (2.6 * 1.9)**.5 * DEGREE
# resolutions[.022] = (1.1 * 1.7)**.5 * DEGREE
resolutions[.045] = 3.6 * DEGREE
# resolutions[.82] = 1.2 * DEGREE
resolutions[2.33] = 20 * ARCMIN
if plot_individual:
    plot_dataset(motherfile)

#########################
###Haslam#########################
#########################
haslam = fit.read("/home/omniscope/data/polarized foregrounds/haslam408_dsds_Remazeilles2014.fits")['TEMPERATURE'].flatten()
haslam = {.408: haslam[hpf.nest2ring(int((len(haslam)/12)**.5), range(len(haslam)))]}
resolutions[.408] = .8 * DEGREE
if plot_individual:
    plot_dataset(haslam)

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
stockert = sv.equirectangular2heapix(stockert)

elisa = np.array([sv.equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/Elisa_POL_%s.bin"%key)[0]) for key in ['I', 'Q', 'U']])
elisa[elisa > 1000] = np.nan

drao = np.array([sv.equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/DRAO_POL_%s.bin"%key)[0]) for key in ['I', 'Q', 'U']])
drao[drao > 1000] = np.nan

reich_q = sv.equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.q.lb.fits")[0], data_x=(PI+np.arange(PI * 2 + 1e-9, 0, -PI/720))%(PI*2+1e-9))
reich_u = -sv.equirectangular2heapix(fit.read("/home/omniscope/data/polarized foregrounds/allsky.u.lb.fits")[0], data_x=(PI+np.arange(PI * 2 + 1e-9, 0, -PI/720))%(PI*2+1e-9))#reich is in IAU convention whose U is minus sign the CMB healpix convention

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
resolutions[1.42] = 0.6 * DEGREE
resolutions[1.3945] = 14.4 * ARCMIN
if plot_individual:
    plot_dataset(all_1400)

# ###########################
# ###S-PASS 9' 2.3GHz 224MHz BW
# #########################
# spass = {}
# spass[2.3] = np.array([fit.read('/home/omniscope/data/polarized foregrounds/spass_hmap_m_1111_%s.fits'%IQU)['UNKNOWN1'].flatten()[hpf.nest2ring(1024, range(hpf.nside2npix(1024)))] for IQU in ['I', 'Q', 'U']])
# spass[2.3][2] = -spass[2.3][2] # IAU convention to CMB convention
# resolutions[2.3] = 9 * ARCMIN
# if plot_individual:
#     plot_dataset(spass)

#########################
###Create new mother#########################
#########################
new_mother = {}
for dict in [
    haslam,
    motherfile,
    all_1400,
    # spass,
    stockert11cm,
    ]:
    for f, data in dict.iteritems():
        if resolutions[f] < target_fwhm * 1.1:
            if target_fwhm <= resolutions[f]:
                smoothing_fwhm = 0.
            else:
                smoothing_fwhm = (target_fwhm**2 - resolutions[f]**2)**.5
            print f, resolutions[f], smoothing_fwhm
            sys.stdout.flush()

            edge_width = np.max(min_edge_width, smoothing_fwhm)

            if data.ndim == 2:
                if I_only:
                    new_mother[f] = preprocess(data[0], mother_nside, fwhm=smoothing_fwhm, edge_width=edge_width)
                else:
                    new_mother[f] = np.array([preprocess(d, mother_nside, fwhm=smoothing_fwhm, edge_width=edge_width) for d in data])
            else:
                new_mother[f] = preprocess(data, mother_nside, fwhm=smoothing_fwhm, edge_width=edge_width)
        else:
            print "skipping", f
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


