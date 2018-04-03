import time, datetime

Timer_Start = time.time()
print('Programme Starts at: %s' % str(datetime.datetime.now()))
import ephem, sys, os, resource, warnings
# import simulate_visibilities.Bulm as Bulm
import HERA_MapMaking_VisibilitySimulation.Bulm
# import simulate_visibilities.simulate_visibilities as sv
import HERA_MapMaking_VisibilitySimulation.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import aipy as ap
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import healpy as hp
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob
import astropy
# from aipy.miriad import pol2str
from astropy.io import fits
import HERA_MapMaking_VisibilitySimulation as mmvs
from pyuvdata import UVData, UVCal, UVFITS
from HERA_MapMaking_VisibilitySimulation import UVData as UVData_HR
from HERA_MapMaking_VisibilitySimulation import UVCal as UVCal_HR
from HERA_MapMaking_VisibilitySimulation import UVFITS as UVFITS_HR
import hera_cal as hc
from hera_cal.data import DATA_PATH as data_path_heracal
# from HERA_MapMaking_VisibilitySimulation import DATA_PATH
from collections import OrderedDict as odict
from pyuvdata import utils as uvutils
import copy
import uvtools as uvt
import linsolve
from hera_cal.datacontainer import DataContainer
from astropy.time import Time
import omnical
import omnical.calibration_omni as omni
from memory_profiler import memory_usage as memuse
from collections import OrderedDict as odict
import pandas
import aipy.miriad as apm
import re
import copy
from hera_cal import utils, firstcal, cal_formats, redcal

PI = np.pi
TPI = PI * 2

try:
    str2pol = {
        'I': 1,  # Stokes Paremeters
        'Q': 2,
        'U': 3,
        'V': 4,
        'rr': -1,  # Circular Polarizations
        'll': -2,
        'rl': -3,
        'lr': -4,
        'xx': -5,  # Linear Polarizations
        'yy': -6,
        'xy': -7,
        'yx': -8,
    }
    
    pol2str = {}
    for k in str2pol: pol2str[str2pol[k]] = k
except:
    from aipy.miriad import pol2str
    from aipy.miriad import str2pol


def pixelize(sky, nside_distribution, nside_standard, nside_start, thresh, final_index, thetas, phis, sizes):
    # thetas = []
    # phis = []
    for inest in range(12 * nside_start ** 2):
        pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis,
                        sizes)


# newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
# thetas += newt.tolist()
# phis += newp.tolist()
# return np.array(thetas), np.array(phis)


def pixelize_helper(sky, nside_distribution, nside_standard, nside, inest, thresh, final_index, thetas, phis, sizes):
    # print "visiting ", nside, inest
    starti, endi = inest * nside_standard ** 2 / nside ** 2, (inest + 1) * nside_standard ** 2 / nside ** 2
    ##local mean###if nside == nside_standard or np.std(sky[starti:endi])/np.mean(sky[starti:endi]) < thresh:
    if nside == nside_standard or np.std(sky[starti:endi]) < thresh:
        nside_distribution[starti:endi] = nside
        final_index[starti:endi] = len(thetas)  # range(len(thetas), len(thetas) + endi -starti)
        # return hp.pix2ang(nside, [inest], nest=True)
        newt, newp = hp.pix2ang(nside, [inest], nest=True)
        thetas += newt.tolist()
        phis += newp.tolist()
        sizes += (np.ones_like(newt) * nside_standard ** 2 / nside ** 2).tolist()
    # sizes += (np.ones_like(newt) / nside**2).tolist()
    
    else:
        # thetas = []
        # phis = []
        for jnest in range(inest * 4, (inest + 1) * 4):
            pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh, final_index, thetas,
                            phis, sizes)


# newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh)
# thetas += newt.tolist()
# phis += newp.tolist()
# return np.array(thetas), np.array(phis)


def dot(A, B, C, nchunk=10):
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("A B C not all have 2 dims: %i %i %i" % (str(A.ndim), str(B.ndim), str(C.ndim)))
    
    chunk = len(C) / nchunk
    for i in range(nchunk):
        C[i * chunk:(i + 1) * chunk] = A[i * chunk:(i + 1) * chunk].dot(B)
    if chunk * nchunk < len(C):
        C[chunk * nchunk:] = A[chunk * nchunk:].dot(B)


def ATNIA(A, Ni, C, nchunk=20):  # C=AtNiA
    if A.ndim != 2 or C.ndim != 2 or Ni.ndim != 1:
        raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i" % (A.ndim, C.ndim, Ni.ndim))
    
    expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1]) ** 2
    print "Estimated time for A %i by %i" % (A.shape[0], A.shape[1]), expected_time, "minutes",
    sys.stdout.flush()
    
    chunk = len(C) / nchunk
    for i in range(nchunk):
        ltm = time.time()
        C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
        if expected_time >= 1.:
            print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
            sys.stdout.flush()
    if chunk * nchunk < len(C):
        C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)


def get_A(additive_A=None, A_path='', force_recompute=None, nUBL_used=None, nt_used=None, valid_npix=None, thetas=None, phis=None, used_common_ubls=None, beam_heal_equ_x=None, beam_heal_equ_y=None, lsts=None, freq=None):
    if os.path.isfile(A_path) and not force_recompute:
        print "Reading A matrix from %s" % A_path
        sys.stdout.flush()
        A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used))
    else:
        
        print "Computing A matrix..."
        sys.stdout.flush()
        A = np.empty((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used), dtype='complex128')
        timer = time.time()
        for n in range(valid_npix):
            ra = phis[n]
            dec = PI / 2 - thetas[n]
            print "\r%f%% completed, %f minutes left" % (
                100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
            sys.stdout.flush()
            
            A[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_x, tlist=lsts) / 2  # xx and yy are each half of I
            A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=lsts) / 2
        
        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A.tofile(A_path)
    
    # #put in autocorr regardless of whats saved on disk
    # for i in range(nUBL_used):
    #     for p in range(2):
    #         A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
    #         A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]
    
    A.shape = (nUBL_used * 2 * nt_used, A.shape[-1])
    if not fit_for_additive:
        A = A[:, :valid_npix]
    else:
        A[:, valid_npix:] = additive_A[:, 1:]
    # Merge A
    try:
        return np.concatenate((np.real(A), np.imag(A))).astype('float64')
    except MemoryError:
        print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
        sys.stdout.flush()
        Ashape = list(A.shape)
        Ashape[0] = Ashape[0] * 2
        np.real(A).tofile(A_path + 'tmpre')
        np.imag(A).tofile(A_path + 'tmpim')
        del (A)
        os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))
        
        os.system("rm %s" % (A_path + 'tmpim'))
        A = np.fromfile(A_path + 'tmpre', dtype='float64').reshape(Ashape)
        os.system("rm %s" % (A_path + 'tmpre'))
        print "done."
        sys.stdout.flush()
        return A.astype('float53')


def get_complex_data(real_data, nubl=None, nt=None):
    if len(real_data.flatten()) != 2 * nubl * 2 * nt:
        raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i." % (len(real_data), nubl, nt, 2 * nubl * 2 * nt))
    input_shape = real_data.shape
    real_data.shape = (2, nubl, 2, nt)
    result = real_data[0] + 1.j * real_data[1]
    real_data.shape = input_shape
    return result


def stitch_complex_data(complex_data):
    return np.concatenate((np.real(complex_data.flatten()), np.imag(complex_data.flatten()))).astype('float64')


def get_vis_normalization(data, clean_sim_data, data_shape=None):
    a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    return a.dot(b) / b.dot(b)


def sol2map(sol, valid_npix=None, npix=None, valid_pix_mask=None, final_index=None, sizes=None):
    solx = sol[:valid_npix]
    full_sol = np.zeros(npix)
    full_sol[valid_pix_mask] = solx / sizes
    return full_sol[final_index]


def sol2additive(sol, valid_npix=None, nUBL_used=None):
    return np.transpose(sol[valid_npix:].reshape(nUBL_used, 2, 2), (1, 0, 2))  # ubl by pol by re/im before transpose


def solve_phase_degen(data_xx, data_yy, model_xx, model_yy, ubls, plot=False):  # data should be time by ubl at single freq. data * phasegensolution = model
    if data_xx.shape != data_yy.shape or data_xx.shape != model_xx.shape or data_xx.shape != model_yy.shape or data_xx.shape[1] != ubls.shape[0]:
        raise ValueError("Shapes mismatch: %s %s %s %s, ubl shape %s" % (data_xx.shape, data_yy.shape, model_xx.shape, model_yy.shape, ubls.shape))
    A = np.zeros((len(ubls) * 2, 2))
    b = np.zeros(len(ubls) * 2)
    
    nrow = 0
    for p, (data, model) in enumerate(zip([data_xx, data_yy], [model_xx, model_yy])):
        for u, ubl in enumerate(ubls):
            amp_mask = (np.abs(data[:, u]) > (np.median(np.abs(data[:, u])) / 2.))
            A[nrow] = ubl[:2]
            b[nrow] = omni.medianAngle(np.angle(model[:, u] / data[:, u])[amp_mask])
            nrow += 1
    phase_cal = omni.solve_slope(np.array(A), np.array(b), 1)
    if plot:
        plt.hist((np.array(A).dot(phase_cal) - b + PI) % TPI - PI)
        plt.title('phase fitting error')
        plt.show()
    
    # sooolve
    return phase_cal


class LastUpdatedOrderedDict(odict):
    'Store items in the order the keys were last added'
    
    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        odict.__setitem__(self, key, value)


def S_casa_v_t(v, t=2015.5):
    S_0 = 2190.294  # S_casA_1GHz
    alpha = 0.725
    belta = 0.0148
    tau = 6.162 * 1.e-5
    
    a = -0.00633  # +-0.00024 year-1
    b = 0.00039  # +-0.00008 year -1
    c = 1.509 * 1.e-7  # +-0.162*1.e-7 year-1
    
    v *= 1.e-3
    
    # print(v) # from MHz to GHz
    # print(t) # in decimal year
    
    S_casa_v = S_0 * v ** (-alpha + belta * np.log(v)) * np.exp(-tau * v ** (-2.1))  # S_0: 2015.5
    
    d_speed_log_v = a + b * np.log(v) + c * v ** (-2.1)  # a,b,c: 2005.0
    
    S_casa_v_t = np.exp((t - 2015.5) * d_speed_log_v + np.log(S_casa_v))
    
    # print(d_speed_log_v)
    
    return S_casa_v_t


def S_cyga_v(v, t=2005):
    S_cyga_v = 3.835 * 1.e5 * v ** (-0.718) * np.exp(-0.342 * (21.713 / v) ** 2.1)
    
    return S_cyga_v


def UVData2AbsCalDict(datanames, pol_select=None, pop_autos=True, return_meta=False, filetype='miriad',
                      pick_data_ants=True):
    """
    turn a list of pyuvdata.UVData objects or a list of miriad or uvfits file paths
    into the datacontainer dictionary form that AbsCal requires. This format is
    keys as antennas-pair + polarization format, Ex. (1, 2, 'xx')
    and values as 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

    Parameters:
    -----------
    datanames : list of either strings of data file paths or list of UVData instances
                to concatenate into a single dictionary

    pol_select : list of polarization strings to keep

    pop_autos : boolean, if True: remove autocorrelations

    return_meta : boolean, if True: also return antenna and unique frequency and LST arrays

    filetype : string, filetype of data if datanames is a string, options=['miriad', 'uvfits']
                can be ingored if datanames contains UVData objects.

    pick_data_ants : boolean, if True and return_meta=True, return only antennas in data

    Output:
    -------
    if return_meta is True:
        (data, flags, antpos, ants, freqs, times, lsts, pols)
    else:
        (data, flags)

    data : dictionary containing baseline-pol complex visibility data
    flags : dictionary containing data flags
    antpos : dictionary containing antennas numbers as keys and position vectors
    ants : ndarray containing unique antennas
    freqs : ndarray containing frequency channels (Hz)
    times : ndarray containing julian date bins of data
    lsts : ndarray containing LST bins of data (radians)
    pols : ndarray containing list of polarization index integers
    """
    # check datanames is not a list
    if type(datanames) is not list and type(datanames) is not np.ndarray:
        if type(datanames) is str:
            # assume datanames is a file path
            uvd = UVData()
            suffix = os.path.splitext(datanames)[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(datanames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(datanames)
        else:
            # assume datanames is a UVData instance
            uvd = datanames
    else:
        # if datanames is a list, check data types of elements
        if type(datanames[0]) is str:
            # assume datanames contains file paths
            uvd = UVData()
            suffix = os.path.splitext(datanames[0])[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(datanames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(datanames)
        else:
            # assume datanames contains UVData instances
            uvd = reduce(operator.add, datanames)
    
    # load data
    d, f = firstcal.UVData_to_dict([uvd])
    
    # pop autos
    if pop_autos:
        for i, k in enumerate(d.keys()):
            if k[0] == k[1]:
                d.pop(k)
                f.pop(k)
    
    # turn into datacontainer
    data, flags = DataContainer(d), DataContainer(f)
    
    # get meta
    if return_meta:
        freqs = np.unique(uvd.freq_array)
        times = np.unique(uvd.time_array)
        lsts = np.unique(uvd.lst_array)
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
        antpos = odict(zip(ants, antpos))
        pols = uvd.polarization_array
        return data, flags, antpos, ants, freqs, times, lsts, pols
    else:
        return data, flags


def UVData2AbsCalDict_Auto(datanames, pol_select=None, pop_autos=True, return_meta=False, filetype='miriad',
                           pick_data_ants=True, svmemory=True, Time_Average=1, Frequency_Average=1, Dred=False, inplace=True, tol=5.e-4, Select_freq=False, Select_time=False):
    """
    turn a list of pyuvdata.UVData objects or a list of miriad or uvfits file paths
    into the datacontainer dictionary form that AbsCal requires. This format is
    keys as antennas-pair + polarization format, Ex. (1, 2, 'xx')
    and values as 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

    Parameters:
    -----------
    datanames : list of either strings of data file paths or list of UVData instances
                to concatenate into a single dictionary

    pol_select : list of polarization strings to keep

    pop_autos : boolean, if True: remove autocorrelations

    return_meta : boolean, if True: also return antenna and unique frequency and LST arrays

    filetype : string, filetype of data if datanames is a string, options=['miriad', 'uvfits']
                can be ingored if datanames contains UVData objects.

    pick_data_ants : boolean, if True and return_meta=True, return only antennas in data

    Output:
    -------
    if return_meta is True:
        (data, flags, antpos, ants, freqs, times, lsts, pols, autocorr, autocorr_flags)
    else:
        (data, flags, autocorr, autocorr_flags))

    data : dictionary containing baseline-pol complex visibility data
    flags : dictionary containing data flags
    antpos : dictionary containing antennas numbers as keys and position vectors
    ants : ndarray containing unique antennas
    freqs : ndarray containing frequency channels (Hz)
    times : ndarray containing julian date bins of data
    lsts : ndarray containing LST bins of data (radians)
    pols : ndarray containing list of polarization index integers
    """
    # check datanames is not a list
    if type(datanames) is not list and type(datanames) is not np.ndarray:
        if type(datanames) is str:
            # assume datanames is a file path
            # uvd = UVData()
            uvd = UVData_HR()  # Self-Contain Module form pyuvdata
            suffix = os.path.splitext(datanames)[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(datanames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(datanames, Time_Average=Time_Average, Frequency_Average=Frequency_Average, Dred=Dred, inplace=inplace, tol=tol, Select_freq=Select_freq, Select_time=Select_time)
        else:
            # assume datanames is a UVData instance
            uvd = datanames
    else:
        # if datanames is a list, check data types of elements
        if type(datanames[0]) is str:
            # assume datanames contains file paths
            # uvd = UVData()
            uvd = UVData_HR()  # Self-Contain Module form pyuvdata
            suffix = os.path.splitext(datanames[0])[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(datanames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(datanames, Time_Average=Time_Average, Frequency_Average=Frequency_Average, Dred=Dred, inplace=inplace, tol=tol, Select_freq=Select_freq, Select_time=Select_time)
        else:
            # assume datanames contains UVData instances
            uvd = reduce(operator.add, datanames)
    
    if return_meta:
        # freqs = np.unique(uvd.freq_array)
        # times = np.unique(uvd.time_array)
        # lsts = np.unique(uvd.lst_array)
        freqs = uvd.freq_array.squeeze()
        times = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
        lsts = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
        antpos = odict(zip(ants, antpos))
        pols = uvd.polarization_array
        redundancy_temp = uvd.redundancy
        redundancy = np.zeros(0, np.int)
        if len(times) != len(np.unique(uvd.time_array)):
            print ('Times Overlapping.')
        else:
            print ('No Time Overlapping.')
        if len(lsts) != len(np.unique(uvd.lst_array)):
            print ('Lsts Overlapping.')
        else:
            print ('No Lst Overlapping.')
        if len(freqs) != len(np.unique(uvd.freq_array)):
            print ('Frequencies Overlapping.')
        else:
            print ('No Frequency Overlapping.')
    
    # load data
    if not svmemory:
        d, f = firstcal.UVData_to_dict([uvd])
    else:
        d, f = UVData_to_dict_svmemory([uvd], svmemory=svmemory)
    autos = {}
    autos_flags = {}
    
    # pop autos
    if pop_autos:
        for i, k in enumerate(d.keys()):
            if k[0] == k[1]:
                autos[k] = d[k]
                autos_flags[k] = f[k]
                # redundancy = np.append(redundancy[:i], redundancy[i+1:])
                d.pop(k)
                f.pop(k)
                print('Index of Autocorr popped out: %s.' % (str(i) + ': ' + str(k)))
            else:
                redundancy = np.append(redundancy, redundancy_temp[i])
                print('Index of Baselines not popped out: %s' % (str(i) + ': ' + str(k)))
    # turn into datacontainer
    data, flags = DataContainer(d), DataContainer(f)
    autos_pro, autos_flags_pro = DataContainer(autos), DataContainer(autos_flags)
    
    # get meta
    if return_meta:
        # freqs = np.unique(uvd.freq_array)
        # times = np.unique(uvd.time_array)
        # lsts = np.unique(uvd.lst_array)
        # freqs = uvd.freq_array
        # times = uvd.time_array
        # lsts = uvd.lst_array
        # antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
        # antpos = odict(zip(ants, antpos))
        # pols = uvd.polarization_array
        # if len(times) != len(np.unique(uvd.time_array)):
        # 	print ('Times Overlapping.')
        # else:
        # 	print ('No Time Overlapping.')
        # if len(lsts) != len(np.unique(uvd.lst_array)):
        # 	print ('Lsts Overlapping.')
        # else:
        # 	print ('No Lst Overlapping.')
        # if len(freqs) != len(np.unique(uvd.freq_array)):
        # 	print ('Frequencies Overlapping.')
        # else:
        # 	print ('No Frequency Overlapping.')
        return data, flags, antpos, ants, freqs, times, lsts, pols, autos_pro, autos_flags_pro, redundancy
    else:
        return data, flags, autos_pro, autos_flags_pro, redundancy


def set_lsts_from_time_array_hourangle(times, lon='21:25:41.9', lat='-30:43:17.5'):
    """Set the lst_array based from the time_array."""
    lsts = []
    lst_array = np.zeros(len(np.unique(times)))
    # latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
    for ind, jd in enumerate(np.unique(times)):
        t = Time(jd, format='jd', location=(lon, lat))
        lst_array[np.where(np.isclose(
            jd, times, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').hourangle
    return lst_array


def set_lsts_from_time_array_radian(times, lon='21:25:41.9', lat='-30:43:17.5'):
    """Set the lst_array based from the time_array."""
    lsts = []
    lst_array = np.zeros(len(np.unique(times)))
    # latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
    for ind, jd in enumerate(np.unique(times)):
        t = Time(jd, format='jd', location=(lon, lat))
        lst_array[np.where(np.isclose(
            jd, times, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian
    return lst_array


def UVData_to_dict(uvdata_list, filetype='miriad'):
    """ Turn a list of UVData objects or filenames in to a data and flag dictionary.

        Make dictionary with blpair key first and pol second key from either a
        list of UVData objects or a list of filenames with specific file_type.

        Args:
            uvdata_list: list of UVData objects or strings of filenames.
            filetype (string, optional): type of file if uvdata_list is
                a list of filenames

        Return:
            data (dict): dictionary of data indexed by pol and antenna pairs
            flags (dict): dictionary of flags indexed by pol and antenna pairs
        """
    
    # d, f = {}, {}
    d, f = odict(), odict()
    # d, f = LastUpdatedOrderedDict(), LastUpdatedOrderedDict()
    for uv_in in uvdata_list:
        if type(uv_in) == str:
            fname = uv_in
            uv_in = UVData()
            # read in file without multiple if statements
            getattr(uv_in, 'read_' + filetype)(fname)
        
        # iterate over unique baselines
        for nbl, (i, j) in enumerate(map(uv_in.baseline_to_antnums, np.unique(uv_in.baseline_array))):
            if (i, j) not in d:
                d[i, j] = {}
                f[i, j] = {}
            for ip, pol in enumerate(uv_in.polarization_array):
                pol = pol2str[pol]
                new_data = copy.copy(uv_in.get_data((i, j, pol)))
                new_flags = copy.copy(uv_in.get_flags((i, j, pol)))
                
                if pol not in d[(i, j)]:
                    d[(i, j)][pol] = new_data
                    f[(i, j)][pol] = new_flags
                else:
                    d[(i, j)][pol] = np.concatenate(
                        [d[(i, j)][pol], new_data])
                    f[(i, j)][pol] = np.concatenate(
                        [f[(i, j)][pol], new_flags])
    return d, f


def UVData_to_dict_svmemory(uvdata_list, filetype='miriad', svmemory=True):
    """ Turn a list of UVData objects or filenames in to a data and flag dictionary.

        Make dictionary with blpair key first and pol second key from either a
        list of UVData objects or a list of filenames with specific file_type.

        Args:
            uvdata_list: list of UVData objects or strings of filenames.
            filetype (string, optional): type of file if uvdata_list is
                a list of filenames

        Return:
            data (dict): dictionary of data indexed by pol and antenna pairs
            flags (dict): dictionary of flags indexed by pol and antenna pairs
        """
    
    # d, f = {}, {}
    d, f = odict(), odict()
    # d, f = LastUpdatedOrderedDict(), LastUpdatedOrderedDict()
    print (len(uvdata_list))
    for id_uv, uv_in in enumerate(uvdata_list):
        if type(uv_in) == str:
            fname = uv_in
            uv_in = UVData()
            # read in file without multiple if statements
            getattr(uv_in, 'read_' + filetype)(fname)
        
        # iterate over unique baselines
        for nbl, (i, j) in enumerate(map(uv_in.baseline_to_antnums, uv_in.baseline_array.reshape(uv_in.Ntimes, uv_in.Nbls)[0, :])):
            print('Pair in UVData_to_dict_svmemory: %s(%s, %s)' % (nbl, i, j))
            if (i, j) not in d:
                # d[i, j] = {}
                # f[i, j] = {}
                d[i, j] = odict()
                f[i, j] = odict()
            for ip, pol in enumerate(uv_in.polarization_array):
                pol = pol2str[pol]
                new_data = copy.copy(uv_in.get_data((i, j, pol)))
                new_flags = copy.copy(uv_in.get_flags((i, j, pol)))
                
                if pol not in d[(i, j)]:
                    d[(i, j)][pol] = new_data
                    f[(i, j)][pol] = new_flags
                else:
                    d[(i, j)][pol] = np.concatenate(
                        [d[(i, j)][pol], new_data])
                    f[(i, j)][pol] = np.concatenate(
                        [f[(i, j)][pol], new_flags])
        if svmemory:
            uvdata_list[id_uv] = []
            print ('Blank uvdata: %s' % id_uv)
    
    return d, f


def Compress_Data_by_Average(data=None, dflags=None, Time_Average=1, Frequency_Average=1, data_freqs=None, data_times=None, data_lsts=None, Contain_Autocorr=True, autocorr_data_mfreq=None, DicData=False, pol=None, use_select_time=False, use_select_freq=False):
    if np.mod(data[data.keys()[0]].shape[0], Time_Average) != 0:
        if (data[data.keys()[0]].shape[0] / Time_Average) < 1.:
            # Time_Average = 1
            Time_Average = np.min((data[data.keys()[0]].shape[0], Time_Average))
    if np.mod(data[data.keys()[0]].shape[1], Frequency_Average) != 0:
        if (data[data.keys()[0]].shape[1] / Frequency_Average) < 1.:
            # Frequency_Average = 1
            Frequency_Average = np.min((data[data.keys()[0]].shape[1], Frequency_Average))
    
    remove_times = np.mod(data[data.keys()[0]].shape[0], Time_Average)
    remove_freqs = np.mod(data[data.keys()[0]].shape[1], Frequency_Average)
    if remove_times == 0:
        remove_times = -data[data.keys()[0]].shape[0]
    if remove_freqs == 0:
        remove_freqs = -data[data.keys()[0]].shape[1]
    print ('Time_Average: %s; Frequency_Average: %s.' % (Time_Average, Frequency_Average))
    print ('Remove_Times: %s; Remove_Freqs: %s.' % (remove_times, remove_freqs))
    
    # data_ff = {}
    # dflags_ff = {}
    # autocorr_data_mfreq_ff = {}
    # data_freqs_ff = {}
    # data_times_ff = {}
    # data_lsts_ff = {}
    
    # for i in range(2):
    timer = time.time()
    data_ff = LastUpdatedOrderedDict()
    dflags_ff = LastUpdatedOrderedDict()
    # autocorr_data_mfreq_ff[i] = LastUpdatedOrderedDict()
    # data_freqs_ff[i] = LastUpdatedOrderedDict()
    # data_times_ff[i] = LastUpdatedOrderedDict()
    # data_lsts_ff[i] = LastUpdatedOrderedDict()
    
    data_freqs = data_freqs[: -remove_freqs]
    data_times = data_times[: -remove_times]
    data_lsts = data_lsts[: -remove_times]
    
    data_freqs_ff = data_freqs.reshape(len(data_freqs) / Frequency_Average, Frequency_Average)[:, 0]
    data_times_ff = data_times.reshape(len(data_times) / Time_Average, Time_Average)[:, 0]
    data_lsts_ff = data_lsts.reshape(len(data_lsts) / Time_Average, Time_Average)[:, 0]
    
    for id_key, key in enumerate(data.keys()):
        
        data[key] = data[key][: -remove_times, : -remove_freqs]
        # autocorr_data_mfreq[i] = autocorr_data_mfreq[i][: -remove_times, : -remove_freqs]
        dflags[key] = dflags[key][: -remove_times, : -remove_freqs]
        # data_freqs[i] = data_freqs[i][: -remove_freqs]
        # data_times[i] = data_times[i][: -remove_times]
        # data_lsts[i] = data_lsts[i][: -remove_times]
        if id_key == 0:
            print ('rawData_Shape-%s: %s' % (key, data[key].shape))
            print ('rawDflags_Shape-%s: %s' % (key, dflags[key].shape))
            print ('rawAutocorr_Shape: (%s, %s)' % autocorr_data_mfreq.shape)
            print ('rawData_Freqs: %s' % (len(data_freqs)))
            print ('rawData_Times: %s' % (len(data_times)))
            print ('rawData_Lsts: %s' % (len(data_lsts)))
        
        data_ff[key] = np.mean(data[key].reshape(data[key].shape[0] / Time_Average, Time_Average, data[key].shape[1]), axis=1) if use_select_time else data[key].reshape(data[key].shape[0] / Time_Average, Time_Average, data[key].shape[1])[:, 0, :]
        data_ff[key] = np.mean(data_ff[key].reshape(data[key].shape[0] / Time_Average, data[key].shape[1] / Frequency_Average, Frequency_Average), axis=-1) if use_select_freq else data_ff[key].reshape(data[key].shape[0] / Time_Average, data[key].shape[1] / Frequency_Average, Frequency_Average)[:, :, 0]
        if DicData:
            data.pop(key)
        else:
            data.__delitem__(key)
        
        dflags_ff[key] = np.mean(dflags[key].reshape(dflags[key].shape[0] / Time_Average, Time_Average, dflags[key].shape[1]), axis=1) if use_select_time else dflags[key].reshape(dflags[key].shape[0] / Time_Average, Time_Average, dflags[key].shape[1])[:, 0, :]
        dflags_ff[key] = (np.mean(dflags_ff[key].reshape(dflags[key].shape[0] / Time_Average, dflags[key].shape[1] / Frequency_Average, Frequency_Average), axis=-1) > 0) if use_select_freq else (dflags_ff[key].reshape(dflags[key].shape[0] / Time_Average, dflags[key].shape[1] / Frequency_Average, Frequency_Average)[:, :, 0] > 0)
        if DicData:
            dflags.pop(key)
        else:
            dflags.__delitem__(key)
    
    print('compress_Pol_%s is done. %s seconds used.' % (pol, time.time() - timer))
    
    data = copy.deepcopy(data_ff)
    dflags = copy.deepcopy(dflags_ff)
    # autocorr_data_mfreq = copy.deepcopy(autocorr_data_mfreq_ff)
    data_freqs = copy.deepcopy(data_freqs_ff)
    data_times = copy.deepcopy(data_times_ff)
    data_lsts = copy.deepcopy(data_lsts_ff)
    
    if Contain_Autocorr:
        autocorr_data_mfreq = autocorr_data_mfreq[: -remove_times, : -remove_freqs]
        autocorr_data_mfreq_ff = np.mean(autocorr_data_mfreq.reshape(autocorr_data_mfreq.shape[0] / Time_Average, Time_Average, autocorr_data_mfreq.shape[1]), axis=1) if use_select_time else autocorr_data_mfreq.reshape(autocorr_data_mfreq.shape[0] / Time_Average, Time_Average, autocorr_data_mfreq.shape[1])[:, 0, :]
        autocorr_data_mfreq_ff = np.mean(autocorr_data_mfreq_ff.reshape(autocorr_data_mfreq.shape[0] / Time_Average, autocorr_data_mfreq.shape[1] / Frequency_Average, Frequency_Average), axis=-1) if use_select_freq else autocorr_data_mfreq_ff.reshape(autocorr_data_mfreq.shape[0] / Time_Average, autocorr_data_mfreq.shape[1] / Frequency_Average, Frequency_Average)[:, :, 0]
        autocorr_data_mfreq = copy.deepcopy(autocorr_data_mfreq_ff)
        del (autocorr_data_mfreq_ff)
        try:
            print ('Autocorr_Shape: (%s, %s)' % autocorr_data_mfreq.shape)
        except:
            print('Shape of autocorr results printing not complete.')
    
    del (data_ff)
    del (dflags_ff)
    # del (autocorr_data_mfreq_ff)
    del (data_freqs_ff)
    del (data_times_ff)
    del (data_lsts_ff)
    
    try:
        print ('Data_Shape-%s: %s' % (key, data[key].shape))
        print ('Dflags_Shape-%s: %s' % (key, dflags[key].shape))
        # print ('Autocorr_Shape: (%s, %s)' % autocorr_data_mfreq[i].shape)
        print ('Data_Freqs: %s' % (len(data_freqs)))
        print ('Data_Times: %s' % (len(data_times)))
        print ('Data_Lsts: %s' % (len(data_lsts)))
    except:
        print('Shape of results printing not complete.')
    
    if Contain_Autocorr:
        return data, dflags, autocorr_data_mfreq, data_freqs, data_times, data_lsts
    else:
        return data, dflags, data_freqs, data_times, data_lsts


def De_Redundancy(dflags=None, antpos=None, ants=None, SingleFreq=True, MultiFreq=True, data_freqs=None, Nfreqs=64, data_times=None, Ntimes=None, FreqScaleFactor=1.e6, Frequency_Select=None, Frequency_Select_Index=None, vis_data_mfreq=None, vis_data=None, tol=5.e-4):
    antloc = {}
    
    if SingleFreq:
        flist = {}
        index_freq = {}
        if Frequency_Select_Index is not None:
            for i in range(2):
                index_freq[i] = Frequency_Select_Index[i]
        elif data_freqs is not None:
            for i in range(2):
                flist[i] = np.array(data_freqs[i]) / FreqScaleFactor
                try:
                    # index_freq[i] = np.where(flist[i] == 150)[0][0]
                    #		index_freq = 512
                    index_freq[i] = np.abs(Frequency_Select - flist[i]).argmin()
                except:
                    index_freq[i] = len(flist[i]) / 2
        
        dflags_sf = {}  # single frequency
        for i in range(2):
            dflags_sf[i] = LastUpdatedOrderedDict()
            for key in dflags[i].keys():
                dflags_sf[i][key] = dflags[i][key][:, index_freq[i]]
        
        if vis_data_mfreq is not None and vis_data is None:
            vis_data = {}
            for i in range(2):
                vis_data[i] = vis_data_mfreq[i][index_freq[i], :, :]  # [pol][ freq, time, bl]
        elif vis_data is not None:
            vis_data = vis_data
        else:
            raise ValueError('No vis_data provided or calculated from vis_data_mfreq.')
    
    # ant locations
    for i in range(2):
        antloc[i] = np.array(map(lambda k: antpos[i][k], ants[i]))
    
    bls = [[], []]
    for i in range(2):
        bls[i] = odict([(x, antpos[i][x[0]] - antpos[i][x[1]]) for x in dflags[i].keys()])
        # bls[1] = odict([(y, antpos_yy[y[0]] - antpos_yy[y[1]]) for y in data_yy.keys()])
        bls = np.array(bls)
    
    bsl_coord = [[], []]
    bsl_coord[0] = np.array([bls[0][index] for index in bls[0].keys()])
    bsl_coord[1] = np.array([bls[1][index] for index in bls[1].keys()])
    # bsl_coord_x=bsl_coord_y=bsl_coord
    bsl_coord = np.array(bsl_coord)
    
    Ubl_list_raw = [[], []]
    Ubl_list = [[], []]
    # ant_pos = [[], []]
    
    Nubl_raw = np.zeros(2, dtype=int)
    times_raw = np.zeros(2, dtype=int)
    redundancy_pro = [[], []]
    redundancy_pro_mfreq = [[], []]
    bsl_coord_dred = [[], []]
    bsl_coord_dred_mfreq = [[], []]
    vis_data_dred = [[], []]
    vis_data_dred_mfreq = [[], []]
    
    for i in range(2):
        Ubl_list_raw[i] = np.array(mmvs.arrayinfo.compute_reds_total(antloc[i], tol=tol))  ## Note that a new function has been added into omnical.arrayinfo as "compute_reds_total" which include all ubls not only redundant ones.
    # ant_pos[i] = antpos[i]
    
    for i in range(2):
        for i_ubl in range(len(Ubl_list_raw[i])):
            list_bsl = []
            for i_ubl_pair in range(len(Ubl_list_raw[i][i_ubl])):
                try:
                    list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], '%s' % ['xx', 'yy'][i])))
                except:
                    try:
                        list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], '%s' % ['xx', 'yy'][1 - i])))
                    except:
                        try:
                            list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], '%s' % ['xx', 'yy'][i])))
                        except:
                            try:
                                list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], '%s' % ['xx', 'yy'][1 - i])))
                            except:
                                # print('Baseline:%s%s not in bls[%s]'%(antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], i))
                                pass
            
            if len(list_bsl) >= 1:
                Ubl_list[i].append(list_bsl)
            else:
                pass
    
    for i in range(2):
        Nubl_raw[i] = len(Ubl_list[i])
        times_raw[i] = len(data_times[i]) if data_times is not None else Ntimes
        bsl_coord_dred[i] = np.zeros((Nubl_raw[i], 3))
        bsl_coord_dred_mfreq[i] = np.zeros((Nubl_raw[i], 3))
        vis_data_dred[i] = np.zeros((times_raw[i], Nubl_raw[i]), dtype='complex128')
        vis_data_dred_mfreq[i] = np.zeros((len(data_freqs[i]), times_raw[i], Nubl_raw[i]), dtype='complex128') if data_freqs is not None else np.zeros((Nfreqs, times_raw[i], Nubl_raw[i]), dtype='complex128')
    
    if SingleFreq:
        dflags_dred = {}
        
        for i in range(2):
            dflags_dred[i] = LastUpdatedOrderedDict()
            pol = ['xx', 'yy'][i]
            
            for i_ubl in range(Nubl_raw[0]):
                vis_data_dred[i][:, i_ubl] = np.mean(vis_data[i].transpose()[Ubl_list[i][i_ubl]].transpose(), axis=1)
                bsl_coord_dred[i][i_ubl] = np.mean(bsl_coord[i][Ubl_list[i][i_ubl]], axis=0)
                dflags_dred[i][dflags_sf[i].keys()[Ubl_list[i][i_ubl][0]]] = dflags_sf[i][dflags_sf[i].keys()[Ubl_list[i][i_ubl][0]]]
                redundancy_pro[i].append(len(Ubl_list[i][i_ubl]))
    
    if MultiFreq:
        dflags_dred_mfreq = {}
        
        for i in range(2):
            dflags_dred_mfreq[i] = LastUpdatedOrderedDict()
            pol = ['xx', 'yy'][i]
            
            for i_ubl in range(Nubl_raw[0]):
                vis_data_dred_mfreq[i][:, :, i_ubl] = np.mean(vis_data_mfreq[i][:, :, Ubl_list[i][i_ubl]], axis=-1)
                bsl_coord_dred_mfreq[i][i_ubl] = np.mean(bsl_coord[i][Ubl_list[i][i_ubl]], axis=0)
                dflags_dred_mfreq[i][dflags[i].keys()[Ubl_list[i][i_ubl][0]]] = dflags[i][dflags[i].keys()[Ubl_list[i][i_ubl][0]]]
                redundancy_pro_mfreq[i].append(len(Ubl_list[i][i_ubl]))
    
    if SingleFreq and MultiFreq:
        if (la.norm(np.array(redundancy_pro[0]) - np.array(redundancy_pro_mfreq[0])) + la.norm(np.array(redundancy_pro[1]) - np.array(redundancy_pro_mfreq[1]))) != 0:
            raise ValueError('redundancy_pro doesnot match redundancy_pro_mfreq')
    elif MultiFreq:
        redundancy_pro = redundancy_pro_mfreq
    elif not SingleFreq:
        print ('No De-Redundancy Done.')
    
    if SingleFreq and MultiFreq:
        if (la.norm(bsl_coord_dred[0] - bsl_coord_dred_mfreq[0]) + la.norm(bsl_coord_dred[1] - bsl_coord_dred_mfreq[1])) != 0:
            raise ValueError('bsl_coord_dred doesnot match bsl_coord_dred_mfreq')
    elif MultiFreq:
        bsl_coord_dred = bsl_coord_dred_mfreq
    elif not SingleFreq:
        print ('No De-Redundancy Done.')
    
    if SingleFreq and MultiFreq:
        return vis_data_dred, vis_data_dred_mfreq, redundancy_pro, dflags_dred, dflags_dred_mfreq, bsl_coord_dred, Ubl_list
    elif MultiFreq:
        return vis_data_dred_mfreq, redundancy_pro, dflags_dred_mfreq, bsl_coord_dred, Ubl_list
    elif SingleFreq:
        return vis_data_dred, redundancy_pro, dflags_dred, bsl_coord_dred, Ubl_list
    else:
        return None


def get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=True, Compute_beamweight=False, A_path='', A_got=None, A_version=1.0, AllSky=True, MaskedSky=False, Synthesize_MultiFreq=True, Flist_select_index=None, Flist_select=None, flist=None, Reference_Freq_Index=None, Reference_Freq=None,
                    equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=None, thresh=2., valid_pix_thresh=1.e-4,
                    beam_weight=None, ubls=None, C=299.792458, used_common_ubls=None, nUBL_used=None, nUBL_used_mfreq=None, nt_used=None, nside_standard=None, nside_start=None, nside_beamweight=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=None):
    print('flist: %s' % str(flist))
    
    if Synthesize_MultiFreq:
        if flist is None:
            raise ValueError('No flist provided.')
        if Flist_select_index is None and Flist_select is not None:
            Flist_select_index = {}
            try:
                for i in range(2):
                    Flist_select_index[i] = np.zeros_like(Flist_select[i], dtype='int')
                    for k in range(len(Flist_select[i])):
                        Flist_select_index[i][k] = np.abs(Flist_select[i][k] - flist[i]).argmin()
            except:
                raise ValueError('Flist_select cannot come from flist.')
        elif Flist_select_index is not None:
            for i in range(2):
                Flist_select[i] = flist[i][Flist_select_index[i]]
        else:
            raise ValueError('No Flist_select or Flist_select_index provided.')
        
        # if len(Flist_select) != 2:
        # 	raise ValueError('Please Specify Flist_select for each polarization.')
        
        if Reference_Freq_Index is None:
            Reference_Freq_Index = [[], []]
            for i in range(2):
                try:
                    Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
                except:
                    Reference_Freq_Index[i] = len(flist[i]) / 2
        Reference_Freq = [[], []]
        for i in range(2):
            Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
        print ('Reference_Freq_Index: x-%s; y-%s' % (Reference_Freq_Index[0], Reference_Freq_Index[1]))
        print ('Reference_Freq: x-%s; y-%s' % (Reference_Freq[0], Reference_Freq[1]))
        print ('Flist_select_index: %s' % (str(Flist_select_index)))
        print ('Flist_select: %s' % (str(Flist_select)))
    
    else:
        # if Flist_select is None:
        try:
            if flist is None:
                Flist_select = [[Reference_Freq[0]], [Reference_Freq[1]]]
            
            else:
                if Reference_Freq_Index is not None:
                    for i in range(2):
                        try:
                            Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
                            Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
                        except:
                            Reference_Freq_Index[i] = len(flist[i]) / 2
                            Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
                else:
                    Reference_Freq_Index = [[], []]
                    for i in range(2):
                        try:
                            Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
                            Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
                        except:
                            Reference_Freq_Index[i] = len(flist[i]) / 2
                            Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
                Flist_select = [[Reference_Freq[0]], [Reference_Freq[1]]]
                Flist_select_index = {}
                try:
                    for i in range(2):
                        Flist_select_index[i] = np.zeros_like(Flist_select[i], dtype='int')
                        for k in range(len(Flist_select[i])):
                            Flist_select_index[i][k] = np.abs(Flist_select[i][k] - flist[i]).argmin()
                except:
                    raise ValueError('Flist_select cannot come from flist.')
                print ('Reference_Freq_Index: x-%s; y-%s' % (Reference_Freq_Index[0], Reference_Freq_Index[1]))
        except:
            raise ValueError('Please specify Reference_Freq for each polarization. ')
        
        print ('Reference_Freq: x-%s; y-%s' % (Reference_Freq[0], Reference_Freq[1]))
        if flist is not None:
            print ('Flist_select_index: %s' % (str(Flist_select_index)))
        print ('Flist_select: %s' % (str(Flist_select)))
    
    if len(Flist_select[0]) != len(Flist_select[1]):
        raise ValueError('Lengths of Flist_select for two pols are different.')
    
    if nUBL_used is not None and nUBL_used is not None:
        if nUBL_used != len(used_common_ubls):
            raise ValueError('len(used_common_ubls)%s != nUBL_used%s' % (len(used_common_ubls), nUBL_used))
    
    if nUBL_used_mfreq is not None:
        if nUBL_used_mfreq != len(used_common_ubls) * len(Flist_select[0]):
            print('len(used_common_ubls) * len(Flist_select[0])%s != nUBL_used_mfreq%s' % (len(used_common_ubls) * len(Flist_select[0]), nUBL_used_mfreq))
    
    if lsts is not None and nt_used is not None:
        if len(lsts) != nt_used:
            raise ValueError('number of lsts%s doesnot match nt_used%s.' % (len(lsts), nt_used))
    
    nUBL_used = len(used_common_ubls)
    nUBL_used_mfreq = len(used_common_ubls) * len(Flist_select[0])
    print('nUBL_used: %s\nnUBL_used_mfreq: %s' % (nUBL_used, nUBL_used_mfreq))
    
    if AllSky:
        A_version = A_version
        A = {}
        if equatorial_GSM_standard is None and Synthesize_MultiFreq:
            equatorial_GSM_standard = equatorial_GSM_standard_mfreq[Reference_Freq_Index[0]]  # choose x freq.
        
        for id_p, p in enumerate(['x', 'y']):
            pol = p + p
            try:
                print "%i UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(ubls[p]), np.max(np.linalg.norm(ubls[p], axis=1)) / (C / Reference_Freq[id_p]), pol)
                print "%i Used-Common-UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(used_common_ubls), np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / Reference_Freq[id_p]), pol)
            except:
                try:
                    print "%i Used-Common-UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(used_common_ubls), np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / Reference_Freq[id_p]), pol)
                except:
                    pass
            
            A[p] = np.zeros((len(Flist_select[id_p]), len(used_common_ubls), nt_used, 12 * nside_beamweight ** 2), dtype='complex128')
            
            for id_f, f in enumerate(Flist_select[id_p]):
                if not Synthesize_MultiFreq:
                    if beam_heal_equ_x is None:
                        try:
                            beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
                        except:
                            raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
                    
                    if beam_heal_equ_y is None:
                        try:
                            beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
                        except:
                            raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
                    
                    if p == 'x':
                        beam_heal_equ = beam_heal_equ_x
                    elif p == 'y':
                        beam_heal_equ = beam_heal_equ_y
                else:
                    if p == 'x':
                        beam_heal_equ = beam_heal_equ_x_mfreq[Flist_select_index[id_p][id_f]]
                    elif p == 'y':
                        beam_heal_equ = beam_heal_equ_y_mfreq[Flist_select_index[id_p][id_f]]
                
                # beam
                
                print "Computing sky weighting A matrix for pol: %s, for freq: %s" % (p, f)
                sys.stdout.flush()
                
                # A[p] = np.zeros((nt_used * len(used_common_ubls), 12 * nside_beamweight ** 2), dtype='complex128')
                
                timer = time.time()
                for i in np.arange(12 * nside_beamweight ** 2):
                    dec, ra = hpf.pix2ang(nside_beamweight, i)  # gives theta phi
                    dec = PI / 2 - dec
                    print "\r%.1f%% completed" % (100. * float(i) / (12. * nside_beamweight ** 2)),
                    sys.stdout.flush()
                    if abs(dec - lat_degree * PI / 180) <= PI / 2:
                        if Synthesize_MultiFreq:
                            A[p][id_f, :, :, i] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts)) * (equatorial_GSM_standard_mfreq[Flist_select_index[id_p][id_f], i] / equatorial_GSM_standard[i])
                        else:
                            A[p][id_f, :, :, i] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts))
                
                print "%f minutes used for pol: %s, freq: %s" % ((float(time.time() - timer) / 60.), pol, f)
                sys.stdout.flush()
            A[p] = A[p].reshape(len(Flist_select[id_p]) * len(used_common_ubls), nt_used, 12 * nside_beamweight ** 2)
            print('Shape of A[%s]: %s' % (p, str(A[p].shape)))
            A[p] = A[p].reshape(len(Flist_select[id_p]) * len(used_common_ubls) * nt_used, 12 * nside_beamweight ** 2)
            print('Shape of A[%s]: %s' % (p, str(A[p].shape)))
        
        if Compute_beamweight:
            print "Computing beam weight...",
            sys.stdout.flush()
            beam_weight = ((la.norm(A['x'], axis=0) ** 2 + la.norm(A['y'], axis=0) ** 2) ** .5)[hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
            beam_weight = beam_weight / np.mean(beam_weight)
            thetas_standard, phis_standard = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
            beam_weight = hpf.get_interp_val(beam_weight, thetas_standard, phis_standard, nest=True)  # np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
            print "done."
            sys.stdout.flush()
            return A, beam_weight
        else:
            return A
    
    elif MaskedSky:
        if equatorial_GSM_standard is None:
            equatorial_GSM_standard = equatorial_GSM_standard_mfreq[Reference_Freq_Index[0]]  # choose x freq.
        if beam_weight is None:
            if A_got is None:
                A_got = get_A_multifreq(additive_A=None, A_path=None, A_got=None, A_version=1.0, AllSky=True, MaskedSky=False, Synthesize_MultiFreq=False, flist=flist, Flist_select=None, Flist_select_index=None, Reference_Freq_Index=Reference_Freq_Index, Reference_Freq=Reference_Freq, equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
                                        used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=None, nside_start=None, nside_beamweight=nside_beamweight, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
            print "Computing beam weight...",
            sys.stdout.flush()
            beam_weight = ((la.norm(A_got['x'], axis=0) ** 2 + la.norm(A_got['y'], axis=0) ** 2) ** .5)[hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
            beam_weight = beam_weight / np.mean(beam_weight)
            thetas_standard, phis_standard = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
            beam_weight = hpf.get_interp_val(beam_weight, thetas_standard, phis_standard, nest=True)  # np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
            try:
                del (A_got)
                print('A_got has been successfully deleted.')
            except:
                print('No A_got to be deleted.')
            print "done."
            sys.stdout.flush()
        
        gsm_beamweighted = equatorial_GSM_standard * beam_weight
        
        nside_distribution = np.zeros(12 * nside_standard ** 2)
        final_index = np.zeros(12 * nside_standard ** 2, dtype=int)
        thetas, phis, sizes = [], [], []
        abs_thresh = np.mean(gsm_beamweighted) * thresh
        pixelize(gsm_beamweighted, nside_distribution, nside_standard, nside_start, abs_thresh,
                 final_index, thetas, phis, sizes)
        npix = len(thetas)
        valid_pix_mask = hpf.get_interp_val(gsm_beamweighted, thetas, phis, nest=True) > valid_pix_thresh * max(gsm_beamweighted)
        valid_npix = np.sum(valid_pix_mask)
        print '>>>>>>VALID NPIX =', valid_npix
        
        fake_solution_map = np.zeros_like(thetas)
        for i in range(len(fake_solution_map)):
            fake_solution_map[i] = np.sum(equatorial_GSM_standard[final_index == i])
        fake_solution_map = fake_solution_map[valid_pix_mask]
        
        if Synthesize_MultiFreq:
            fake_solution_map_mfreq_temp = np.zeros((len(Flist_select[0]), npix))
            fake_solution_map_mfreq = np.zeros((len(Flist_select[0]), valid_npix))
            for id_f, f in enumerate(Flist_select_index[0]):
                for i in range(npix):
                    fake_solution_map_mfreq_temp[id_f, i] = np.sum(equatorial_GSM_standard_mfreq[f, final_index == i])
                fake_solution_map_mfreq[id_f] = fake_solution_map_mfreq_temp[id_f, valid_pix_mask]
        
        try:
            del (equatorial_GSM_standard)
            # del(beam_weight)
            print('equatorial_GSM_standard and beam_weight have been successfully deleted.')
        except:
            print('No equatorial_GSM_standard or beam_weight to be deleted.')
        
        try:
            del (equatorial_GSM_standard_mfreq)
            del (fake_solution_map_mfreq_temp)
            print('equatorial_GSM_standard_mfreq and fake_solution_map_mfreq_temp have been successfully deleted.')
        except:
            print('No equatorial_GSM_standard_mfreq or fake_solution_map_mfreq_temp to be deleted.')
        
        sizes = np.array(sizes)[valid_pix_mask]
        thetas = np.array(thetas)[valid_pix_mask]
        phis = np.array(phis)[valid_pix_mask]
        try:
            np.savez(pixel_directory + 'pixel_scheme_%i_%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.npz' % (valid_npix, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none',
                                                                                                                                  bnside, nside_standard), gsm=fake_solution_map, thetas=thetas, phis=phis, sizes=sizes, nside_distribution=nside_distribution, final_index=final_index,
                     n_fullsky_pix=npix, valid_pix_mask=valid_pix_mask, thresh=thresh)
        except:
            print('Not Saving to pixel_directory.')
        
        if not fit_for_additive:
            fake_solution = np.copy(fake_solution_map)
        else:
            fake_solution = np.concatenate((fake_solution_map, np.zeros(4 * nUBL_used)))
        
        if not Compute_A:
            return beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution
        
        if os.path.isfile(A_path) and not force_recompute:
            print "Reading A matrix from %s" % A_path
            sys.stdout.flush()
            A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used * len(Flist_select[0]), 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0])))
        else:
            
            print "Computing A matrix..."
            sys.stdout.flush()
            A = np.empty((len(Flist_select[0]), nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0])), dtype='complex128')
            timer = time.time()
            for id_p, p in enumerate(['x', 'y']):
                for id_f, f in enumerate(Flist_select[id_p]):
                    if not Synthesize_MultiFreq:
                        if beam_heal_equ_x is None:
                            try:
                                beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
                            except:
                                raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
                        
                        if beam_heal_equ_y is None:
                            try:
                                beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
                            except:
                                raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
                        
                        if p == 'x':
                            beam_heal_equ = beam_heal_equ_x
                        elif p == 'y':
                            beam_heal_equ = beam_heal_equ_y
                    else:
                        if p == 'x':
                            beam_heal_equ = beam_heal_equ_x_mfreq[Flist_select_index[id_p][id_f]]
                        elif p == 'y':
                            beam_heal_equ = beam_heal_equ_y_mfreq[Flist_select_index[id_p][id_f]]
                    
                    for n in range(valid_npix):
                        ra = phis[n]
                        dec = PI / 2 - thetas[n]
                        print "\r%f%% completed, %f minutes left for %s-%s" % (
                            100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.), id_f, f),
                        sys.stdout.flush()
                        if Synthesize_MultiFreq:
                            A[id_f, :, id_p, :, n] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2) * (fake_solution_map_mfreq[id_f, n] / fake_solution_map[n])  # xx and yy are each half of I
                        else:
                            A[id_f, :, id_p, :, n] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2)  # xx and yy are each half of I
                    # A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=lsts) / 2
                    
                    print "%f minutes used for pol: %s, freq: %s" % ((float(time.time() - timer) / 60.), p, f)
                print('Shape of A[%s]: %s' % (p, str(A[:, :, id_p, :, :].shape)))
                sys.stdout.flush()
            
            A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0]))
            print('>>>>>>>>>>>>>>>>> Shape of A: %s' % (str(A.shape)))
            try:
                A.tofile(A_path)
            except:
                print('A not saved.')
        
        # #put in autocorr regardless of whats saved on disk
        # for i in range(nUBL_used):
        #     for p in range(2):
        #         A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
        #         A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]
        
        A.shape = (len(Flist_select[0]) * nUBL_used * 2 * nt_used, A.shape[-1])
        if not fit_for_additive:
            A = A[:, :valid_npix]
        else:
            A[:, valid_npix:] = additive_A[:, 1:]
        try:
            print('>>>>>>>>>>>>>>>>> Shape of A after fit_for_additive: %s' % (str(A.shape)))
        # print('>>>>>>>>>>>>>>>>> Shape of A after Real/Imag Seperation: %s' % (str(np.concatenate((np.real(A), np.imag(A))).shape)))
        except:
            print('No printing A.')
        
        # Merge A
        try:
            if Synthesize_MultiFreq:
                return np.concatenate((np.real(A), np.imag(A))).astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution  # , fake_solution_map_mfreq
            else:
                return np.concatenate((np.real(A), np.imag(A))).astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution
        except MemoryError:
            print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
            sys.stdout.flush()
            Ashape = list(A.shape)
            Ashape[0] = Ashape[0] * 2
            np.real(A).tofile(A_path + 'tmpre')
            np.imag(A).tofile(A_path + 'tmpim')
            del (A)
            os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))
            
            os.system("rm %s" % (A_path + 'tmpim'))
            A = np.fromfile(A_path + 'tmpre', dtype='float64').reshape(Ashape)
            os.system("rm %s" % (A_path + 'tmpre'))
            print "done."
            sys.stdout.flush()
            if Synthesize_MultiFreq:
                return A.astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution_map, fake_solution  # , fake_solution_map_mfreq
            else:
                return A.astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution_map, fake_solution
    
    # return A, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map


def Simulate_Visibility_mfreq(script_dir='', INSTRUMENT='', full_sim_filename_mfreq='', sim_vis_xx_filename_mfreq='', sim_vis_yy_filename_mfreq='', Force_Compute_Vis=True, Get_beam_GSM=False, Force_Compute_beam_GSM=False, Multi_freq=False, Multi_Sin_freq=False, Fake_Multi_freq=False, crosstalk_type='',
                              flist=None, freq_index=None, freq=None, equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None,
                              beam_weight=None, C=299.792458, used_common_ubls=None, nUBL_used=None, nUBL_used_mfreq=None, nt_used=None, nside_standard=None, nside_start=None, nside_beamweight=None,
                              beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=None, tlist=None, Time_Expansion_Factor=1.):
    if Force_Compute_beam_GSM or Get_beam_GSM:
        beam_heal_hor_x_mfreq = np.array([local_beam_unpol(flist[0][i])[0] for i in range(len(flist[0]))])
        beam_heal_hor_y_mfreq = np.array([local_beam_unpol(flist[1][i])[1] for i in range(len(flist[1]))])
        beam_heal_equ_x_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_x_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(len(flist[0]))])
        beam_heal_equ_y_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_y_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(len(flist[1]))])
        
        pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
        pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
        pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
        components = np.loadtxt(script_dir + '/../data/components.dat')
        scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
        w1 = si.interp1d(components[:, 0], components[:, 2])
        w2 = si.interp1d(components[:, 0], components[:, 3])
        w3 = si.interp1d(components[:, 0], components[:, 4])
        gsm_standard = {}
        for i in range(2):
            gsm_standard[i] = np.exp(scale_loglog(np.log(freq[i]))) * (w1(freq[i]) * pca1 + w2(freq[i]) * pca2 + w3(freq[i]) * pca3)
        if Multi_freq:
            gsm_standard_mfreq = {}
            for p in range(2):
                gsm_standard_mfreq[p] = np.array([np.exp(scale_loglog(np.log(flist[p][i]))) * (w1(flist[p][i]) * pca1 + w2(flist[p][i]) * pca2 + w3(flist[p][i]) * pca3) for i in range(len(flist[p]))])
        
        # rotate sky map and converts to nest
        equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
        print "Rotating GSM_standard and converts to nest...",
        
        if INSTRUMENT == 'miteor':
            DecimalYear = 2013.58  # 2013, 7, 31, 16, 47, 59, 999998)
            JulianEpoch = 2013.58
        elif 'hera47' in INSTRUMENT:
            DecimalYear = Time(tlist_JD[0], format='jd').decimalyear + (np.mean(Time(tlist_JD, format='jd').decimalyear) - Time(tlist_JD[0], format='jd').decimalyear) * Time_Expansion_Factor
            JulianEpoch = Time(tlist_JD[0], format='jd').jyear + (np.mean(Time(tlist_JD, format='jd').jyear) - Time(tlist_JD[0], format='jd').jyear) * Time_Expansion_Factor  # np.mean(Time(data_times[0], format='jd').jyear)
        print('JulianEpoch: %s' % (str(JulianEpoch)))
        
        sys.stdout.flush()
        equ_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=JulianEpoch))
        ang0, ang1 = hp.rotator.rotateDirection(equ_to_gal_matrix,
                                                hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
        equatorial_GSM_standard = {}
        for i in range(2):
            equatorial_GSM_standard[i] = hpf.get_interp_val(gsm_standard[i], ang0, ang1)
        equatorial_GSM_standard_xx = equatorial_GSM_standard[0]
        equatorial_GSM_standard_yy = equatorial_GSM_standard[1]
        del (equatorial_GSM_standard)
        if Multi_freq:
            equatorial_GSM_standard_mfreq = {}
            for p in range(2):
                equatorial_GSM_standard_mfreq[p] = np.array([hpf.get_interp_val(gsm_standard_mfreq[p][i], ang0, ang1) for i in range(len(flist[p]))])
            equatorial_GSM_standard_mfreq_xx = equatorial_GSM_standard_mfreq[0]
            equatorial_GSM_standard_mfreq_yy = equatorial_GSM_standard_mfreq[1]
            del (equatorial_GSM_standard_mfreq)
        
        print "done."
    
    # if Get_beam_GSM:
    # 	return
    
    if not Multi_freq and not Fake_Multi_freq:
        if flist is None and freq_index is None and freq is None:
            raise valueerror('no frequency can be specified.')
        elif freq is not None:
            flist = [[freq[0]], [freq[1]]]
            if flist is not None:
                freq_index = {}
                freq_index[0] = np.abs(freq[0] - flist[0]).argmin()
                freq_index[1] = np.abs(freq[1] - flist[1]).argmin()
        elif flist is not None and freq_index is not None:
            flist = [[flist[0][freq_index[0]]], [flist[1][freq_index[1]]]]
        elif flist is not None and freq_index is None:
            flist = [[flist[0][len(flist[0]) / 2]], [flist[1][len(flist[1]) / 2]]]
            freq_index = [len(flist[0]) / 2, len(flist[0]) / 2]
            print ('choose the middle of flist for each pol as default since none has been specified.')
    
    else:
        if flist is None:
            if Fake_Multi_freq and (nf_used is None or freq_index is None):
                raise ValueError('Cannot do fake-mfreq simulation without flist provided.')
            elif Fake_Multi_freq:
                flist = np.ones((2, nf_used))
            else:
                raise ValueError('Cannot do mfreq simulation without flist provided.')
        elif Multi_Sin_freq or Fake_Multi_freq:
            if freq_index is not None or freq is not None:
                if freq_index is None:
                    freq_index = {}
                    for i in range(2):
                        freq_index[i] = np.abs(freq[i] - flist[i]).argmin()
                elif freq is not None:
                    for i in range(2):
                        if freq_index[i] != np.abs(freq[i] - flist[i]).argmin():
                            print('freq not match freq_index from flist, use freq_index from flist for pol-%s.' % ['xx', 'yy'][i])
            for i in range(2):
                freq[i] = flist[i][freq_index[i]]
                print('Sinfreq from multifreq: %s-%s' % (freq_index[i], freq[i]))
    
    if len(flist[0]) != len(flist[1]):
        raise ValueError('Two pol nf_used not same: %s != %s' % (len(flist[0]), len(flist[1])))
    nf_used = len(flist[0])
    
    try:
        print('flist: %s MHz;\nnf_used: %s' % (str(flist), nf_used))
    except:
        raise ValueError('No flist information successfully processed and printed.')
    
    if used_common_ubls is not None and nUBL_used is not None:
        if len(used_common_ubls) != nUBL_used:
            raise ValueError('number of used_common_ubls%s doesnot match nUBL_used%s.' % (len(used_common_ubls), nUBL_used))
    nUBL_used = len(used_common_ubls)
    
    if lsts is not None and nt_used is not None:
        if len(lsts) != nt_used:
            raise ValueError('number of lsts%s doesnot match nt_used%s.' % (len(lsts), nt_used))
    nt_used = len(lsts)
    
    if not Multi_freq or Fake_Multi_freq:
        if beam_heal_equ_x is None and beam_heal_equ_x_mfreq is not None:
            beam_heal_equ_x = beam_heal_equ_x_mfreq[freq_index[0]]
        elif beam_heal_equ_x is None and beam_heal_equ_x_mfreq is None:
            raise ValueError('No x beam data.')
        if beam_heal_equ_y is None and beam_heal_equ_y_mfreq is not None:
            beam_heal_equ_y = beam_heal_equ_y_mfreq[freq_index[1]]
        elif beam_heal_equ_y is None and beam_heal_equ_y_mfreq is None:
            raise ValueError('No y beam data either from sinfreq or multifreq.')
        beam_heal_equ_x_mfreq = [beam_heal_equ_x]
        beam_heal_equ_y_mfreq = [beam_heal_equ_y]
    
    else:
        if beam_heal_equ_x_mfreq is None or beam_heal_equ_y_mfreq is None:
            raise ValueError('No multifreq beam data.')
    
    if not Multi_freq or Fake_Multi_freq:
        if equatorial_GSM_standard_xx is None and equatorial_GSM_standard_mfreq_xx is not None:
            equatorial_GSM_standard_xx = equatorial_GSM_standard_mfreq_xx[freq_index[0]]
        elif equatorial_GSM_standard_xx is None and equatorial_GSM_standard_mfreq_xx is None:
            raise ValueError('No equatorial_GSM_standard_xx data.')
        if equatorial_GSM_standard_yy is None and equatorial_GSM_standard_mfreq_yy is not None:
            equatorial_GSM_standard_yy = equatorial_GSM_standard_mfreq_yy[freq_index[1]]
        elif equatorial_GSM_standard_yy is None and equatorial_GSM_standard_mfreq_yy is None:
            raise ValueError('No equatorial_GSM_standard data.')
        
        equatorial_GSM_standard_mfreq = np.array([[equatorial_GSM_standard_xx], [equatorial_GSM_standard_yy]])
    
    else:
        if beam_heal_equ_x_mfreq is None or beam_heal_equ_y_mfreq is None:
            raise ValueError('No multifreq beam data.')
        equatorial_GSM_standard_mfreq = np.array([equatorial_GSM_standard_mfreq_xx, equatorial_GSM_standard_mfreq_yy])
    
    if os.path.isfile(full_sim_filename_mfreq) and not Force_Compute_Vis:
        fullsim_vis_mfreq = np.fromfile(full_sim_filename_mfreq, dtype='complex128').reshape((2, nUBL_used + 1, nt_used, nf_used))
        fullsim_vis_mfreq[0][:-1].astype('complex128').tofile(sim_vis_xx_filename_mfreq)
        fullsim_vis_mfreq[1][:-1].astype('complex128').tofile(sim_vis_yy_filename_mfreq)
    
    else:
        
        fullsim_vis_mfreq = np.zeros((2, nUBL_used + 1, nt_used, nf_used), dtype='complex128')  # since its going to accumulate along the pixels it needs to start with complex128. significant error if start with complex64
        if Fake_Multi_freq:
            print('>>>>Freq_index selected not fake before: %s' % (str(freq_index)))
            freq_index_fakemfreq = copy.deepcopy(freq_index)
            fullsim_vis, autocorr_vis = Simulate_Visibility_mfreq(full_sim_filename_mfreq='', sim_vis_xx_filename_mfreq='', sim_vis_yy_filename_mfreq='', Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_ubls,
                                                                  flist=flist, freq_index=freq_index_fakemfreq, freq=freq, equatorial_GSM_standard_xx=equatorial_GSM_standard_xx, equatorial_GSM_standard_yy=equatorial_GSM_standard_yy, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq_xx, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq_yy, beam_weight=beam_weight,
                                                                  C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
                                                                  nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
            for id_p in range(2):
                fullsim_vis_mfreq[id_p, :-1, :, freq_index[id_p]] = fullsim_vis[:, id_p, :]
                fullsim_vis_mfreq[id_p, -1, :, freq_index[id_p]] = autocorr_vis[id_p]
            # freq_index = freq_index_fakemfreq
            print('>>>>Freq_index selected not fake: %s' % (str(freq_index)))
        
        else:
            full_sim_ubls = np.concatenate((used_common_ubls, [[0, 0, 0]]), axis=0)  # tag along auto corr
            full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
            full_decs = PI / 2 - full_thetas
            full_ras = full_phis
            full_sim_mask = hpf.get_interp_val(beam_weight, full_thetas, full_phis, nest=True) > 0
            # fullsim_vis_DBG = np.zeros((2, len(used_common_ubls), nt_used, np.sum(full_sim_mask)), dtype='complex128')
            
            print "Simulating visibilities, %s, expected time %f min" % (datetime.datetime.now(), 14.6 * nf_used * (nUBL_used / 78.) * (nt_used / 193.) * (np.sum(full_sim_mask) / 1.4e5)),
            sys.stdout.flush()
            masked_equ_GSM_mfreq = equatorial_GSM_standard_mfreq[:, :, full_sim_mask]
            timer = time.time()
            for id_f, f in enumerate(flist[0]):
                for p, beam_heal_equ in enumerate([beam_heal_equ_x_mfreq[id_f], beam_heal_equ_y_mfreq[id_f]]):
                    f = flist[p][id_f]
                    for i, (ra, dec) in enumerate(zip(full_ras[full_sim_mask], full_decs[full_sim_mask])):
                        res = vs.calculate_pointsource_visibility(ra, dec, full_sim_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
                        fullsim_vis_mfreq[p, :, :, id_f] += masked_equ_GSM_mfreq[p, id_f, i] * res
            # fullsim_vis_DBG[p, ..., i] = res[:-1]
            # autocorr = ~16*la.norm, ~80*np.std, ~1.e-5*np.corrrelate
            print "simulated visibilities in %f minutes." % ((time.time() - timer) / 60.)
            try:
                fullsim_vis_mfreq.astype('complex128').tofile(full_sim_filename_mfreq)
                fullsim_vis_mfreq[0][:-1, :, :].astype('complex128').tofile(sim_vis_xx_filename_mfreq)
                fullsim_vis_mfreq[1][:-1, :, :].astype('complex128').tofile(sim_vis_yy_filename_mfreq)
            except:
                print('>>>>>>>>>>>>> Not Saved.')
    
    autocorr_vis_mfreq = np.abs(np.squeeze(fullsim_vis_mfreq[:, -1]))  # (Pol, Times, Freqs)
    fullsim_vis_mfreq = np.squeeze(fullsim_vis_mfreq[:, :-1].transpose((1, 0, 2, 3)))  # (uBL, Pol, Times, Freqs)
    
    if crosstalk_type == 'autocorr':
        autocorr_vis_mfreq_normalized = np.array([autocorr_vis[p, :, id_f] / (la.norm(autocorr_vis[p, :, id_f]) / la.norm(np.ones_like(autocorr_vis[p, :, id_f]))) for id_f in range(autocorr_vis_mfreq.shape[2]) for p in range(2)]).transpose(0, 2, 1)
    else:
        autocorr_vis_mfreq_normalized = np.ones_like(autocorr_vis_mfreq)  # ((2, nt_used, nf_used))
    
    if Multi_Sin_freq and Multi_freq:
        autocorr_vis = np.concatenate((autocorr_vis_mfreq[0:1, :, freq_index[0]], autocorr_vis_mfreq[1:2, :, freq_index[1]]), axis=0)
        autocorr_vis_normalized = np.concatenate((autocorr_vis_mfreq_normalized[0:1, :, freq_index[0]], autocorr_vis_mfreq_normalized[1:2, :, freq_index[1]]), axis=0)
        fullsim_vis = np.concatenate((fullsim_vis_mfreq[:, 0:1, :, freq_index[0]], fullsim_vis_mfreq[:, 1:2, :, freq_index[1]]), axis=1)
    
    if Multi_Sin_freq and Multi_freq:
        print('Shape of Autocorr_vis at %sMHz: %s' % (str(freq), str(autocorr_vis.shape)))
        print('Shape of Autocorr_vis_normalized at %sMHz: %s' % (str(freq), str(autocorr_vis_normalized.shape)))
        print('Shape of Fullsim_vis at %sMHz: %s' % (str(freq), str(fullsim_vis.shape)))
        print('Shape of Autocorr_vis_mfreq: %s' % (str(autocorr_vis_mfreq.shape)))
        print('Shape of Autocorr_vis_mfreq_normalized: %s' % (str(autocorr_vis_mfreq_normalized.shape)))
        print('Shape of Fullsim_vis_mfreq: %s' % (str(fullsim_vis_mfreq.shape)))
        return fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized, fullsim_vis, autocorr_vis, autocorr_vis_normalized
    
    else:
        print('Shape of Autocorr_vis_mfreq: %s' % (str(autocorr_vis_mfreq.shape)))
        print('Shape of Autocorr_vis_mfreq_normalized: %s' % (str(autocorr_vis_mfreq_normalized.shape)))
        print('Shape of Fullsim_vis_mfreq: %s' % (str(fullsim_vis_mfreq.shape)))
        return fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized


def Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq=False, Absolute_Calibration_dred=False, re_cal_times=1, Mocal_time_bin_temp=None, nt_used=None, lsts=None, Mocal_freq_bin_temp=None, flist=None, fullsim_vis_mfreq=None, vis_data_dred_mfreq=None, dflags_dred_mfreq=None, add_Autobsl=False, autocorr_vis_mfreq=None, autocorr_data_mfreq=None, bl_dred_mfreq_select=8):
    if nt_used is not None:
        if nt_used != len(lsts):
            raise ValueError('nt_used doesnot match len(lsts).')
    nt_used = len(lsts)
    
    # Mocal_time_bin_temp = 5
    mocal_time_bin = np.min([Mocal_time_bin_temp, nt_used])
    mocal_time_bin_num = nt_used / mocal_time_bin if np.mod(nt_used, mocal_time_bin) == 0 else (nt_used / mocal_time_bin + 1)
    print('Mocal_time_bin_temp: %s; mocal_time_bin: %s; mocal_time_bin_num: %s' % (Mocal_time_bin_temp, mocal_time_bin, mocal_time_bin_num))
    
    # Mocal_freq_bin_temp = 64
    mocal_freq_bin = 1 if not Absolute_Calibration_dred_mfreq else np.min([Mocal_freq_bin_temp, len(flist[0])])
    mocal_freq_bin_num = len(flist[0]) / mocal_freq_bin if np.mod(len(flist[0]), mocal_freq_bin) == 0 else (len(flist[0]) / mocal_freq_bin + 1)
    print('Mocal_freq_bin_temp: %s; mocal_freq_bin: %s; mocal_freq_bin_num: %s' % (Mocal_freq_bin_temp, mocal_freq_bin, mocal_freq_bin_num))
    
    model_dred_mfreq = {}
    data_dred_mfreq = {}
    abs_corr_data_dred_mfreq = {}
    vis_data_dred_mfreq_abscal = [[], []]
    autocorr_data_dred_mfreq_abscal = [[], []]
    vis_data_dred_abscal = [[], []]
    autocorr_data_dred_abscal = [[], []]
    interp_flags_dred_mfreq = {}
    AC_dred_mfreq = {}
    DAC_dred_mfreq = {}
    DPAC_dred_mfreq = {}
    dly_phs_corr_data_dred_mfreq = {}
    auto_select_dred_mfreq = {}
    delay_corr_data_dred_mfreq = {}
    
    try:
        cdflags_dred_mfreq = copy.deepcopy(dflags_dred_mfreq)
    except:
        pass
    
    wgts_dred_mfreq = {}
    for i in range(2):
        pol = ['xx', 'yy'][i]
        re_cal = 0
        model_dred_mfreq[i] = LastUpdatedOrderedDict()
        data_dred_mfreq[i] = LastUpdatedOrderedDict()
        
        vis_data_dred_mfreq_abscal[i] = np.zeros_like(vis_data_dred_mfreq[i], dtype='complex128')
        autocorr_data_dred_mfreq_abscal[i] = np.zeros_like(autocorr_vis_mfreq[i])
        vis_data_dred_abscal[i] = np.zeros_like(vis_data_dred_mfreq_abscal[i][index_freq[i], :, :])
        autocorr_data_dred_abscal[i] = np.zeros_like(autocorr_vis_mfreq[i][:, index_freq[i]])
        
        for id_t_bin in range(mocal_time_bin_num):
            nt_mocal_used = mocal_time_bin if (id_t_bin + 1) * mocal_time_bin <= nt_used else (nt_used - id_t_bin * mocal_time_bin)
            
            for id_f_bin in range(mocal_freq_bin_num):
                nf_mocal_used = mocal_freq_bin if (id_f_bin + 1) * mocal_freq_bin <= len(flist[0]) else (len(flist[0]) - id_f_bin * mocal_freq_bin)
                
                keys = dflags_dred_mfreq[i].keys()
                for key_index, key in enumerate(keys):
                    model_dred_mfreq[i][key] = fullsim_vis_mfreq[key_index, i, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
                    # data_dred_mfreq[i][key] = np.real(vis_data_dred_mfreq[i][:, :, key_index].transpose()) + np.abs(np.imag(vis_data_dred_mfreq[i][:, :, key_index].transpose()))*1j #[pol][freq,time,ubl_index].transpose()
                    data_dred_mfreq[i][key] = vis_data_dred_mfreq[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, key_index].transpose()  # [pol][freq,time,ubl_index].transpose()
                    cdflags_dred_mfreq[i][key] = dflags_dred_mfreq[i][key][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
                if add_Autobsl:
                    model_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = autocorr_vis_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]  # not lose generality, choose the first anntena in the first UBL for autocorrelation calibraiton.
                    data_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]  # add the autocorrelation of first antenna in the first UBL as the last line in visibility.
                    cdflags_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = np.array([[False] * autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used].shape[1]] * autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used].shape[0])
                    auto_select_dred_mfreq[i] = (keys[0][0], keys[0][0], keys[0][2])
                print(dflags_dred_mfreq[i].keys())
                print(dflags_dred_mfreq[i].keys()[0][0])
                print('(id_t_bin: %s, id_f_bin: %s) data_shape[%s][%s]: (%s) \n' % (id_t_bin, id_f_bin, ['xx', 'yy'][i], key, data_dred_mfreq[i][key].shape))
                
                wgts_dred_mfreq[i] = copy.deepcopy(cdflags_dred_mfreq[i])
                for k in wgts_dred_mfreq[i].keys():
                    if not Fake_wgts_dred_mfreq:
                        wgts_dred_mfreq[i][k] = (~wgts_dred_mfreq[i][k]).astype(np.float)
                    else:
                        wgts_dred_mfreq[i][k] = (((~wgts_dred_mfreq[i][k]).astype(np.float) + 1).astype(bool)).astype(np.float)
                
                lsts_binned = lsts[id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used]
                flist_binned = flist[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
                
                for re_cal in range(re_cal_times):  # number of times of absolute calibration
                    if re_cal == 0:
                        if not Absolute_Calibration_dred_mfreq:
                            # Skip Delay_Lincal
                            # instantiate class
                            DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                        else:
                            # instantiate class
                            try:
                                model_dred_mfreq[i], interp_flags_dred_mfreq[i] = hc.abscal.interp2d_vis(model_dred_mfreq[i], lsts_binned, flist_binned, lsts_binned, flist_binned)
                            except:
                                print('No Interp')
                            AC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                            # kernel is median filter kernel, chosen to produce time-smooth output delays for this particular dataset
                            AC_dred_mfreq[i].delay_lincal(kernel=(1, ((np.min([nf_mocal_used, 11]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True, solve_offsets=True)
                            # apply to data
                            delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].ant_dly_gain))
                            # instantiate class
                            DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                    else:
                        if not Absolute_Calibration_dred_mfreq:
                            # delay_corr_data_dred_mfreq[i] = abs_corr_data_dred_mfreq[i]
                            DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], abs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                        else:
                            # instantiate class
                            AC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], abs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                            AC_dred_mfreq[i].delay_lincal(kernel=(1, ((np.min([nf_mocal_used, 3]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True, solve_offsets=True)
                            # apply to data
                            delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].ant_dly_gain))
                            # instantiate class
                            DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                    
                    # # instantiate class
                    # DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                    # avg phase solver
                    DAC_dred_mfreq[i].phs_logcal(avg=True)
                    # apply to data
                    dly_phs_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(DAC_dred_mfreq[i].data, (DAC_dred_mfreq[i].ant_phi_gain))
                    # instantiate class
                    DPAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], dly_phs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
                    # run amp linsolve
                    DPAC_dred_mfreq[i].abs_amp_logcal()
                    # run phs linsolve
                    DPAC_dred_mfreq[i].TT_phs_logcal(zero_psi=False, four_pol=False)
                    # apply to data
                    abs_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(DPAC_dred_mfreq[i].data,
                                                                        (DPAC_dred_mfreq[i].abs_psi_gain, DPAC_dred_mfreq[i].TT_Phi_gain, DPAC_dred_mfreq[i].abs_eta_gain), gain_convention='multiply')
                
                for key_id, key in enumerate(dflags_dred_mfreq[i].keys()):
                    vis_data_dred_mfreq_abscal[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, key_id] = abs_corr_data_dred_mfreq[i][key].transpose()
                # vis_data_dred_mfreq_abscal[i][:, :, key_id] = np.real(abs_corr_data_dred_mfreq[i][key].transpose()) + np.abs(np.imag(abs_corr_data_dred_mfreq[i][key].transpose()))*1j
                if add_Autobsl:
                    autocorr_data_dred_mfreq_abscal[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used] = abs_corr_data_dred_mfreq[i][auto_select_dred_mfreq[i]]
                else:
                    autocorr_data_dred_mfreq_abscal[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used] = autocorr_vis_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
        
        vis_data_dred_abscal[i] = vis_data_dred_mfreq_abscal[i][index_freq[i], :, :]
        if add_Autobsl:
            autocorr_data_dred_abscal[i] = autocorr_data_dred_mfreq_abscal[i][:, index_freq[i]]
        else:
            autocorr_data_dred_abscal[i] = autocorr_vis_mfreq[i][:, index_freq[i]]
    
    try:
        bl_dred_mfreq = [dflags_dred_mfreq[0].keys()[bl_dred_mfreq_select], dflags_dred_mfreq[1].keys()[bl_dred_mfreq_select]]  # [(25, 37, 'xx'), (25, 37, 'yy')]
        fig3 = {}
        axes3 = {}
        fig3_data = {}
        axes3_data = {}
        fig3_data_abscorr = {}
        axes3_data_abscorr = {}
        for i in range(2):  # add another redundant 'for loop' for testing plotting.
            pol = ['xx', 'yy'][i]
            try:
                plt.figure(80000000 + 10 * i)
                fig3[i], axes3[i] = plt.subplots(2, 1, figsize=(12, 8))
                plt.sca(axes3[i][0])
                uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_select, i, :, :], mode='log', mx=6, drng=4)
                plt.colorbar()
                plt.title(pol + ' model AMP {}'.format(bl_dred_mfreq[i]))
                plt.sca(axes3[i][1])
                uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_select, i, :, :], mode='phs', mx=np.pi, drng=2 * np.pi)
                plt.colorbar()
                plt.title(pol + ' model PHS {}'.format(bl_dred_mfreq[i]))
                plt.show(block=False)
                plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_model-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                # plt.cla()
                
                plt.figure(90000000 + 10 * i)
                fig3_data[i], axes3_data[i] = plt.subplots(2, 1, figsize=(12, 8))
                plt.sca(axes3_data[i][0])
                uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_select].transpose(), mode='log', mx=1.5, drng=5)
                plt.colorbar()
                plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq[i]))
                plt.sca(axes3_data[i][1])
                uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
                plt.colorbar()
                plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq[i]))
                plt.show(block=False)
                plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                # plt.cla()
                
                ####################### after ABS Calibration #########################
                
                plt.figure(8000000 + 10 * i)
                fig3_data_abscorr[i], axes3_data_abscorr[i] = plt.subplots(2, 1, figsize=(12, 8))
                plt.sca(axes3_data_abscorr[i][0])
                uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_select].transpose(), mode='log', mx=6, drng=4)
                plt.colorbar()
                plt.title(pol + ' abs_caled data AMP {}'.format(bl_dred_mfreq[i]))
                plt.sca(axes3_data_abscorr[i][1])
                uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
                plt.colorbar()
                plt.title(pol + ' abs_caled data PHS {}'.format(bl_dred_mfreq[i]))
                plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_data-caled-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                plt.show(block=False)
            # plt.cla()
            except:
                print('Error when Plotting Mocal Results.')
    except:
        print('No Plotting for Model_Calibration Results.')
    
    return vis_data_dred_mfreq_abscal, autocorr_data_dred_mfreq_abscal, vis_data_dred_abscal, autocorr_data_dred_abscal, mocal_time_bin, mocal_freq_bin


def PointSource_Calibration(data_var_xx_filename_pscal='', data_var_yy_filename_pscal='', PointSource_AbsCal=False, PointSource_AbsCal_SingleFreq=False, Pt_vis=False, From_AbsCal=False, comply_ps2mod_autocorr=False, southern_points=None, phase_degen_niter_max=50,
                            index_freq=None, freq=None, flist=None, lsts=None, tlist=None, vis_data_dred_mfreq=None, vis_data_dred_mfreq_abscal=None, autocorr_data_mfreq=None, autocorr_data_dred_mfreq_abscal=None, equatorial_GSM_standard_mfreq=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None,
                            Integration_Time=None, Frequency_Bin=None, used_redundancy=None, nt=None, nUBL=None, ubls=None, bl_dred_mfreq_pscal_select=8, dflags_dred_mfreq=None, INSTRUMENT=None, used_common_ubls=None, nUBL_used=None, nt_used=None, bnside=None, nside_standard=None):
    for source in southern_points.keys():
        southern_points[source]['body'] = ephem.FixedBody()
        southern_points[source]['body']._ra = southern_points[source]['ra']
        southern_points[source]['body']._dec = southern_points[source]['dec']
    
    flux_func = {}
    # flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
    # flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])
    flux_func['cas'] = si.interp1d(flist[0], np.array([S_casa_v_t(flist[0][i], DecimalYear) for i in range(len(flist[0]))]))
    flux_func['cyg'] = si.interp1d(flist[0], np.array([S_cyga_v(flist[0][i], DecimalYear) for i in range(len(flist[0]))]))
    
    full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
    
    if Pt_vis:
        flux_raw_gsm_ps = {}
        flux_gsm_ps = {}
        flux_raw_dis_gsm_ps = {}
        flux_dis_gsm_ps = {}
        pix_index_gsm_ps = {}
        pix_raw_index_gsm_ps = {}
        pix_max_index_gsm_ps = {}
        pt_sources = southern_points.keys()
        for source in pt_sources:
            flux_raw_gsm_ps[source] = 0
            flux_gsm_ps[source] = 0
            flux_raw_dis_gsm_ps[source] = []
            flux_dis_gsm_ps[source] = []
            pix_raw_index_gsm_ps[source] = []
            pix_index_gsm_ps[source] = []
            # pix_max_index_gsm_ps[source] = []
            for i in range(len(equatorial_GSM_standard)):
                if la.norm(np.array([full_phis[i] - southern_points[source]['body']._ra,
                                     (PI / 2 - full_thetas[i]) - southern_points[source]['body']._dec])) <= 0.1:
                    flux_raw_gsm_ps[source] += equatorial_GSM_standard[i]
                    flux_raw_dis_gsm_ps[source].append(equatorial_GSM_standard[i])
                    pix_raw_index_gsm_ps[source].append(i)
            
            pix_max_index_gsm_ps[source] = pix_raw_index_gsm_ps[source][flux_raw_dis_gsm_ps[source].index(np.array(flux_raw_dis_gsm_ps[source]).max())]
            for j in range(len(flux_raw_dis_gsm_ps[source])):
                if flux_raw_dis_gsm_ps[source][j] >= 0.4 * equatorial_GSM_standard[pix_max_index_gsm_ps[source]]:
                    flux_gsm_ps[source] += equatorial_GSM_standard[pix_raw_index_gsm_ps[source][j]]
                    flux_dis_gsm_ps[source].append(equatorial_GSM_standard[pix_raw_index_gsm_ps[source][j]])
                    pix_index_gsm_ps[source].append(pix_raw_index_gsm_ps[source][j])
            
            print('total flux of %s' % source, flux_gsm_ps[source])
            print('total raw flux of %s' % source, flux_raw_gsm_ps[source])
            print('maximum pix flux of %s' % source, equatorial_GSM_standard[pix_max_index_gsm_ps[source]])
            print('pix-index with maximum flux of %s' % source, pix_max_index_gsm_ps[source])
            print('raw-pix-indexes of %s' % source, pix_raw_index_gsm_ps[source])
            print('pix-indexes of %s' % source, pix_index_gsm_ps[source])
            print('\n')
        
        # pt_sources = ['cyg', 'cas']
        pt_sources = southern_points.keys()
        pt_vis = np.zeros((len(pt_sources), 2, nUBL_used, nt_used), dtype='complex128')
        if INSTRUMENT == 'miteor':
            print "Simulating cyg casvisibilities, %s, expected time %.1f min" % (datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (2. / 1.4e5)),
            sys.stdout.flush()
            timer = time.time()
            for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
                for i, source in enumerate(pt_sources):
                    ra = southern_points[source]['body']._ra
                    dec = southern_points[source]['body']._dec
                    # 			pt_vis[i, p] = jansky2kelvin * flux_func[source](freq) * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
                    pt_vis[i, p] = flux_gsm_ps[source] * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
        elif 'hera47' in INSTRUMENT:
            print "Simulating cyg casvisibilities, %s, expected time %.1f min" % (datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (2. / 1.4e5)),
            sys.stdout.flush()
            timer = time.time()
            for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
                for i, source in enumerate(pt_sources):
                    ra = southern_points[source]['body']._ra
                    dec = southern_points[source]['body']._dec
                    # 			pt_vis[i, p] = jansky2kelvin * flux_func[source](freq) * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
                    pt_vis[i, p] = flux_gsm_ps[source] * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
    
    vis_freq = {}
    
    autocorr_data_dred_mfreq_pscal = {}
    vis_data_dred_mfreq_pscal = {}
    
    if PointSource_AbsCal_SingleFreq:
        pscal_freqstart = index_freq[0]
        pscal_freqend = index_freq[0] + 1
    else:
        pscal_freqstart = 0
        pscal_freqend = np.min([len(flist[0]), len(flist[1])])
    
    for j, p in enumerate(['x', 'y']):
        pol = p + p
        vis_data_dred_mfreq_pscal[j] = np.zeros_like(vis_data_dred_mfreq[j])
        autocorr_data_dred_mfreq_pscal[j] = np.zeros_like(autocorr_data_mfreq[j])
    
    for id_f in range(pscal_freqstart, pscal_freqend, 1):
        vis_freq[0] = flist[0][id_f]
        vis_freq[1] = flist[1][id_f]
        # cal_lst_range = np.array([5, 6]) / TPI * 24.
        # 		cal_lst_range = np.array([tlist[15], tlist[-15]])
        cal_lst_range = np.array([tlist[len(tlist) / 3], tlist[-len(tlist) / 3]])
        calibrate_ubl_length = 2600 / np.mean([vis_freq[0], vis_freq[1]])  # 10.67
        # cal_time_mask = tmask	 #(tlist>cal_lst_range[0]) & (tlist<cal_lst_range[1])#a True/False mask on all good data to get good data in cal time range
        cal_time_mask = (tlist >= cal_lst_range[0]) & (tlist <= cal_lst_range[1])
        # cal_ubl_mask = np.linalg.norm(ubls[p], axis=1) >= calibrate_ubl_length
        
        print('%i times used' % len(lsts[cal_time_mask]))
        
        flux_raw_gsm_ps = {}
        flux_gsm_ps = {}
        flux_raw_dis_gsm_ps = {}
        flux_dis_gsm_ps = {}
        pix_index_gsm_ps = {}
        pix_raw_index_gsm_ps = {}
        pix_max_index_gsm_ps = {}
        pt_sources = southern_points.keys()
        for source in pt_sources:
            flux_raw_gsm_ps[source] = 0
            flux_gsm_ps[source] = 0
            flux_raw_dis_gsm_ps[source] = []
            flux_dis_gsm_ps[source] = []
            pix_raw_index_gsm_ps[source] = []
            pix_index_gsm_ps[source] = []
            # pix_max_index_gsm_ps[source] = []
            for i in range(len(equatorial_GSM_standard_mfreq[id_f])):
                if la.norm(np.array([full_phis[i] - southern_points[source]['body']._ra,
                                     (PI / 2 - full_thetas[i]) - southern_points[source]['body']._dec])) <= 0.1:
                    flux_raw_gsm_ps[source] += equatorial_GSM_standard_mfreq[id_f, i]
                    flux_raw_dis_gsm_ps[source].append(equatorial_GSM_standard_mfreq[id_f, i])
                    pix_raw_index_gsm_ps[source].append(i)
            
            pix_max_index_gsm_ps[source] = pix_raw_index_gsm_ps[source][flux_raw_dis_gsm_ps[source].index(np.array(flux_raw_dis_gsm_ps[source]).max())]
            for j in range(len(flux_raw_dis_gsm_ps[source])):
                if flux_raw_dis_gsm_ps[source][j] >= 0.5 * equatorial_GSM_standard_mfreq[id_f, pix_max_index_gsm_ps[source]]:
                    flux_gsm_ps[source] += equatorial_GSM_standard_mfreq[id_f, pix_raw_index_gsm_ps[source][j]]
                    flux_dis_gsm_ps[source].append(equatorial_GSM_standard_mfreq[id_f, pix_raw_index_gsm_ps[source][j]])
                    pix_index_gsm_ps[source].append(pix_raw_index_gsm_ps[source][j])
            
            print('total flux of %s' % source, flux_gsm_ps[source])
            print('total raw flux of %s' % source, flux_raw_gsm_ps[source])
            print('maximum pix flux of %s' % source, equatorial_GSM_standard_mfreq[id_f, pix_max_index_gsm_ps[source]])
            print('pix-index with maximum flux of %s' % source, pix_max_index_gsm_ps[source])
            print('raw-pix-indexes of %s' % source, pix_raw_index_gsm_ps[source])
            print('pix-indexes of %s' % source, pix_index_gsm_ps[source])
            print('\n')
        
        Ni = {}
        cubls = copy.deepcopy(ubls)
        ubl_sort = {}
        noise_data_pscal = {}
        N_data_pscal = {}
        vis_data_dred_pscal = {}
        
        From_AbsCal = False
        
        for i, p in enumerate(['x', 'y']):
            pol = p + p
            cal_ubl_mask = np.linalg.norm(ubls[p], axis=1) >= calibrate_ubl_length
            # get Ni (1/variance) and data
            # var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, vartag)
            # noise_data_pscal['y'] = np.array([(np.random.normal(0,autocorr_data[1][t_index]/(Integration_Time*Frequency_Bin)**0.5,nUBL_used) ) for t_index in range(len(autocorr_data[1]))],dtype='float64').flatten()
            
            if From_AbsCal:
                vis_data_dred_pscal[i] = vis_data_dred_mfreq_abscal[i][id_f][np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
                noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_abscal[i][t_index, id_f] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(autocorr_data_dred_mfreq_abscal[0].shape[0])], dtype='float64').flatten()  # Absolute Calibrated
            else:
                vis_data_dred_pscal[i] = vis_data_dred_mfreq[i][id_f][np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
                noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_mfreq[i][t_index, id_f] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(autocorr_data_mfreq[0].shape[0])], dtype='float64').flatten()  # Absolute Calibrated
            
            N_data_pscal[p] = noise_data_pscal[p] * noise_data_pscal[p]
            # N_data_pscal[p] = N_data[p]
            # N_data_pscal['y'] = noise_data_pscal['y'] * noise_data_pscal['y']
            Ni[p] = 1. / N_data_pscal[p].reshape((nt, nUBL))[np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
            ubls[p] = ubls[p][cal_ubl_mask]
            ubl_sort[p] = np.argsort(la.norm(ubls[p], axis=1))
            
            print "%i UBLs to include" % len(ubls[p])
        
        del (noise_data_pscal)
        
        print "Computing UNpolarized point sources matrix..."
        sys.stdout.flush()
        # cal_sources = ['cyg', 'cas']
        cal_sources = southern_points.keys()
        Apol = np.empty((np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask), len(cal_sources)), dtype='complex128')
        timer = time.time()
        for n, source in enumerate(cal_sources):
            ra = southern_points[source]['body']._ra
            dec = southern_points[source]['body']._dec
            
            Apol[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq[0], beam_heal_equ=beam_heal_equ_x_mfreq[id_f], tlist=lsts[cal_time_mask])
            Apol[:, 1, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq[1], beam_heal_equ=beam_heal_equ_y_mfreq[id_f], tlist=lsts[cal_time_mask])
        
        Apol = np.conjugate(Apol).reshape((np.sum(cal_ubl_mask), 2 * np.sum(cal_time_mask), len(cal_sources)))
        Ni = np.transpose([Ni['x'], Ni['y']], (1, 0, 2))
        
        realA = np.zeros((2 * Apol.shape[0] * Apol.shape[1], 1 + 2 * np.sum(cal_ubl_mask) * 2), dtype='float64')
        # 		realA[:, 0] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2])))), axis=0).dot([jansky2kelvin_mfreq[0][id_f] * flux_func[source](vis_freq[0]) for source in cal_sources])
        realA[:, 0] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2])))), axis=0).dot([flux_gsm_ps[source] for source in cal_sources])
        vis_scale = la.norm(realA[:, 0]) / len(realA) ** .5
        for coli, ncol in enumerate(range(1, realA.shape[1])):
            realA[coli * np.sum(cal_time_mask): (coli + 1) * np.sum(cal_time_mask), ncol] = vis_scale
        
        realNi = np.concatenate((Ni.flatten() * 2, Ni.flatten() * 2))
        realAtNiAinv = np.linalg.pinv(np.einsum('ji,j,jk->ik', realA, realNi, realA))
        
        b = np.transpose([vis_data_dred_pscal[0], vis_data_dred_pscal[1]], (1, 0, 2))
        phase_degen_niter = 0
        phase_degen2 = {'x': np.zeros(2), 'y': np.zeros(2)}
        phase_degen_iterative_x = np.zeros(2)
        phase_degen_iterative_y = np.zeros(2)
        
        def tocomplex(realdata):
            reshapedata = realdata.reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
            return reshapedata[0] + reshapedata[1] * 1.j
        
        # phase_degen_niter_max = 100
        while (phase_degen_niter < phase_degen_niter_max and max(np.linalg.norm(phase_degen_iterative_x), np.linalg.norm(phase_degen_iterative_y)) > 1e-5) or phase_degen_niter == 0:
            phase_degen_niter += 1
            b[:, 0] = b[:, 0] * np.exp(1.j * ubls['x'][:, :2].dot(phase_degen_iterative_x))[:, None]
            b[:, -1] = b[:, -1] * np.exp(1.j * ubls['y'][:, :2].dot(phase_degen_iterative_y))[:, None]
            realb = np.concatenate((np.real(b.flatten()), np.imag(b.flatten())))
            
            psol = realAtNiAinv.dot(np.transpose(realA).dot(realNi * realb))
            realb_fit = realA.dot(psol)
            perror = ((realb_fit - realb) * (realNi ** .5)).reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
            
            realbfit_noadditive = realA[:, 0] * psol[0]
            realbfit_additive = realb_fit - realbfit_noadditive
            realb_noadditive = realb - realbfit_additive
            bfit_noadditive = tocomplex(realbfit_noadditive)
            b_noadditive = tocomplex(realb_noadditive)
            if phase_degen_niter == phase_degen_niter_max:
                phase_degen_iterative_x = solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'])  # , [3, 3, 1e3])
                phase_degen_iterative_y = solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'])  # , [3, 3, 1e3])
            
            else:
                phase_degen_iterative_x = solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'])
                phase_degen_iterative_y = solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'])
            phase_degen2['x'] += phase_degen_iterative_x
            phase_degen2['y'] += phase_degen_iterative_y
            print phase_degen_niter, phase_degen2['x'], phase_degen2['y'], np.linalg.norm(perror)
        
        renorm = 1 / (2 * psol[0])
        
        print (renorm, vis_freq[0], phase_degen2['x'], vis_freq[1], phase_degen2['y'])
        
        # freqs[fi] = vis_freq
        
        ################################# apply to data and var and output unpolarized version ####################################
        # data_var_xx_filename_pscal = script_dir + '/../Output/%s_%s_p2_u%i_t%i_nside%i_bnside%i_var_data_xx_pscal.simvis' % (INSTRUMENT, freq, nUBL, nt, nside_standard, bnside)
        # data_var_yy_filename_pscal = script_dir + '/../Output/%s_%s_p2_u%i_t%i_nside%i_bnside%i_var_data_yy_pscal.simvis' % (INSTRUMENT, freq, nUBL, nt, nside_standard, bnside)
        
        ######### recover ubls and ubl_sort ##########
        ubls = cubls
        # ubl_sort = cubl_sort
        
        if Keep_Red:
            nUBL = len(bsl_coord_x)
            for p in ['x', 'y']:
                # ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
                ubls[p] = globals()['bsl_coord_' + p]
            common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
        
        else:
            nUBL = len(bsl_coord_dred[0])
            nUBL_yy = len(bsl_coord_dred[1])
            for i in range(2):
                p = ['x', 'y'][i]
                ubls[p] = bsl_coord_dred[i]
            common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
        
        # get data and var and apply change
        
        for j, p in enumerate(['x', 'y']):
            pol = p + p
            
            if From_AbsCal:
                vis_data_dred_mfreq_pscal[j][id_f] = vis_data_dred_mfreq_abscal[j][id_f] * np.exp(1.j * ubls[p][:, :2].dot(phase_degen2[p])) * renorm
                if comply_ps2mod_autocorr:
                    autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_vis_mfreq[j][:, id_f]
                else:
                    autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_data_dred_mfreq_abscal[j][:, id_f] * np.abs(renorm)  # Absolute Calibrated
            else:
                vis_data_dred_mfreq_pscal[j][id_f] = vis_data_dred_mfreq[j][id_f] * np.exp(1.j * ubls[p][:, :2].dot(phase_degen2[p])) * renorm
                if comply_ps2mod_autocorr:
                    autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_vis_mfreq[j][:, id_f]
                else:
                    autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_data_mfreq[j][:, id_f] * np.abs(renorm)  # Absolute Calibrated
    
    noise_data_pscal = {}
    N_data_pscal = {}
    vis_data_dred_pscal = {}
    for i, p in enumerate(['x', 'y']):
        pol = p + p
        vis_data_dred_pscal[i] = vis_data_dred_mfreq_pscal[i][index_freq[i]]
        noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_pscal[i][t_index, index_freq[i]] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[i]) ** 0.5) for t_index in range(len(autocorr_data_dred_mfreq_pscal[i]))], dtype='float64').flatten()  # Absolute Calibrated
        
        N_data_pscal[p] = noise_data_pscal[p] * noise_data_pscal[p]
        N_data_pscal[p] = N_data_pscal[p].reshape((nt, nUBL))
        
        if not os.path.isfile(globals()['data_var_' + pol + '_filename_pscal']):
            try:
                N_data_pscal[p].astype('float64').tofile(globals()['data_var_' + pol + '_filename_pscal'])
            except:
                print('N_data_pscal not saved.')
        else:
            print('N_data_pscal already exists on disc.')
    # (new_var * 100.).astype('float32').tofile(op_var100_filename)
    del (noise_data_pscal)
    
    try:
        bl_dred_mfreq_pscal = [dflags_dred_mfreq[0].keys()[bl_dred_mfreq_pscal_select], dflags_dred_mfreq[1].keys()[bl_dred_mfreq_pscal_select]]  # [(25, 37, 'xx'), (25, 37, 'yy')]
        fig4 = {}
        axes4 = {}
        fig4_data = {}
        axes4_data = {}
        fig4_data_abscorr = {}
        axes4_data_abscorr = {}
        for i in range(2):  # add another redundant 'for loop' for testing plotting.
            pol = ['xx', 'yy'][i]
            try:
                plt.figure(80000000 + 10 * i)
                fig4[i], axes4[i] = plt.subplots(2, 1, figsize=(12, 8))
                plt.sca(axes4[i][0])
                uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_pscal_select, i, :, :], mode='log', mx=6, drng=4)
                plt.colorbar()
                plt.title(pol + ' model AMP {}'.format(bl_dred_mfreq_pscal[i]))
                plt.sca(axes4[i][1])
                uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_pscal_select, i, :, :], mode='phs', mx=np.pi, drng=2 * np.pi)
                plt.colorbar()
                plt.title(pol + ' model PHS {}'.format(bl_dred_mfreq_pscal[i]))
                plt.show(block=False)
                plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_model-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                # plt.cla()
                
                if From_AbsCal:
                    plt.figure(90000000 + 10 * i)
                    fig4_data[i], axes4_data[i] = plt.subplots(2, 1, figsize=(12, 8))
                    plt.sca(axes4_data[i][0])
                    uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=1.5, drng=5)
                    plt.colorbar()
                    plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq_pscal[i]))
                    plt.sca(axes4_data[i][1])
                    uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
                    plt.colorbar()
                    plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq_pscal[i]))
                    plt.show(block=False)
                    plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                
                else:
                    plt.figure(90000000 + 10 * i)
                    fig4_data[i], axes4_data[i] = plt.subplots(2, 1, figsize=(12, 8))
                    plt.sca(axes4_data[i][0])
                    uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=1.5, drng=5)
                    plt.colorbar()
                    plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq_pscal[i]))
                    plt.sca(axes4_data[i][1])
                    uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
                    plt.colorbar()
                    plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq_pscal[i]))
                    plt.show(block=False)
                    plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                # plt.cla()
                
                ####################### after ABS Calibration #########################
                
                plt.figure(8000000 + 10 * i)
                fig4_data_abscorr[i], axes4_data_abscorr[i] = plt.subplots(2, 1, figsize=(12, 8))
                plt.sca(axes4_data_abscorr[i][0])
                uvt.plot.waterfall(vis_data_dred_mfreq_pscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=6, drng=4)
                plt.colorbar()
                plt.title(pol + ' abs_caled data AMP {}'.format(bl_dred_mfreq_pscal[i]))
                plt.sca(axes4_data_abscorr[i][1])
                uvt.plot.waterfall(vis_data_dred_mfreq_pscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
                plt.colorbar()
                plt.title(pol + ' abs_caled data PHS {}'.format(bl_dred_mfreq_pscal[i]))
                plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-caled-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
                plt.show(block=False)
            # plt.cla()
            except:
                print('Error when Plotting Pscal Results')
    except:
        print('No Plotting for Pscal Results.')
    
    if Pt_vis:
        return vis_data_dred_mfreq_pscal, autocorr_data_dred_mfreq_pscal, vis_data_dred_pscal, pt_vis, pt_sources
    else:
        return vis_data_dred_mfreq_pscal, autocorr_data_dred_mfreq_pscal, vis_data_dred_pscal


def Pre_Calibration(pre_calibrate=False, pre_ampcal=False, pre_phscal=False, pre_addcal=False, comply_ps2mod_autocorr=False, Use_PsAbsCal=False, Use_AbsCal=False, Use_Fullsim_Noise=False, Precal_time_bin_temp=None, nt_used=None, nUBL_used=None, data_shape=None, cal_times=1, niter_max=50, antpairs=None, ubl_index=None,
                    autocorr_vis_normalized=None, fullsim_vis=None, data=None, Ni=None, pt_vis=None, pt_sources=None, used_common_ubls=None, freq=None, lsts=None, lst_offset=None, INSTRUMENT=None, Absolute_Calibration_dred_mfreq=False, mocal_time_bin=None, mocal_freq_bin=None, bnside=None, nside_standard=None):
    if nt_used is not None:
        if nt_used != len(lsts):
            raise ValueError('nt_used doesnot match len(lsts).')
    nt_used = len(lsts)
    
    if nUBL_used is not None:
        if nUBL_used != len(used_common_ubls):
            raise ValueError('nUBL_used doesnot match len(used_common_ubls).')
    nUBL_used = len(used_common_ubls)
    
    #####1. antenna based calibration#######
    precal_time_bin = np.min([Precal_time_bin_temp, nt_used])
    precal_time_bin_num = (data_shape['xx'][1] / precal_time_bin) if np.mod(data_shape['xx'][1], precal_time_bin) == 0 else (data_shape['xx'][1] / precal_time_bin) + 1
    print ("Precal_time_bin: %s; \nPrecal_time_bin_num: %s" % (precal_time_bin, precal_time_bin_num))
    raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    
    try:
        if antpairs is not None:
            used_antpairs = antpairs[abs(ubl_index['x']) - 1]
            n_usedants = np.unique(used_antpairs)
    except:
        pass
    
    #####2. re-phasing and crosstalk#######
    additive_A = np.zeros((nUBL_used, 2, nt_used, 1 + 4 * nUBL_used)).astype('complex128')
    
    # put in autocorr regardless of whats saved on disk
    for p in range(2):
        additive_A[:, p, :, 0] = fullsim_vis[:, p]
        for i in range(nUBL_used):
            additive_A[i, p, :, 1 + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]  # [id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
            additive_A[i, p, :, 1 + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]  # [id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
    additive_A.shape = (nUBL_used * 2 * nt_used, 1 + 4 * nUBL_used)
    
    additive_term = np.zeros_like(data)
    additive_term_incr = np.zeros_like(data)
    
    for id_t_bin in range(precal_time_bin_num):
        nt_precal_used = precal_time_bin if ((id_t_bin + 1) * precal_time_bin) <= data_shape['xx'][1] else (data_shape['xx'][1] - id_t_bin * precal_time_bin)
        print ('Nt_precal_used: %s' % nt_precal_used)
        
        additive_A_tbin = additive_A.reshape(nUBL_used, 2, nt_used, 1 + 4 * nUBL_used)[:, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used), :].reshape(nUBL_used * 2 * nt_precal_used, 1 + 4 * nUBL_used)
        
        for cal_index in range(cal_times):
            
            if pre_calibrate:
                # import omnical.calibration_omni as omni
                # raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
                # raw_Ni = np.copy(Ni)
                
                real_additive_A = np.concatenate((np.real(additive_A_tbin), np.imag(additive_A_tbin)), axis=0).astype('complex128')
                if pre_ampcal:  # if pre_ampcal, allow xx and yy to fit amp seperately
                    n_prefit_amp = 2
                    real_additive_A.shape = (2 * nUBL_used, 2, nt_precal_used, 1 + 4 * nUBL_used)
                    real_additive_A_expand = np.zeros((2 * nUBL_used, 2, nt_precal_used, n_prefit_amp + 4 * nUBL_used), dtype='complex128')
                    for i in range(n_prefit_amp):
                        real_additive_A_expand[:, i, :, i] = real_additive_A[:, i, :, 0]
                    real_additive_A_expand[..., n_prefit_amp:] = real_additive_A[..., 1:]
                    real_additive_A = real_additive_A_expand
                    real_additive_A.shape = (2 * nUBL_used * 2 * nt_precal_used, n_prefit_amp + 4 * nUBL_used)
                else:
                    n_prefit_amp = 1
                
                additive_AtNiA = np.empty((n_prefit_amp + 4 * nUBL_used, n_prefit_amp + 4 * nUBL_used), dtype='complex128')
                if pre_addcal:
                    ATNIA(real_additive_A, Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), additive_AtNiA)
                    additive_AtNiAi = sla.inv(additive_AtNiA)
                else:
                    real_additive_A[..., n_prefit_amp:] = 0.
                    ATNIA(real_additive_A, Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), additive_AtNiA)
                    additive_AtNiAi = sla.pinv(additive_AtNiA)
                
                niter = 0
                rephases = np.zeros((2, 2))
                # additive_term = np.zeros_like(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten())
                # additive_term_incr = np.zeros_like(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten())
                while (niter == 0 or la.norm(rephases) > .001 or la.norm(additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) / la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) > .001) and niter < niter_max:
                    niter += 1
                    
                    if pre_phscal:
                        cdata = get_complex_data(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used)
                        for p, pol in enumerate(['xx', 'yy']):
                            # rephase = omni.solve_phase_degen_fast(cdata[:, p].transpose(), cdata[:, p].transpose(), fullsim_vis[:, p].transpose(), fullsim_vis[:, p].transpose(), used_common_ubls)
                            rephase = solve_phase_degen(cdata[:, p].transpose(), cdata[:, p].transpose(), fullsim_vis[:, p, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].transpose(), fullsim_vis[:, p, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].transpose(), used_common_ubls)
                            rephases[p] = rephase
                            if p == 0:
                                print 'pre process rephase', pol, rephase,
                            else:
                                print pol, rephase
                            cdata[:, p] *= np.exp(1.j * used_common_ubls[:, :2].dot(rephase))[:, None]
                        data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(cdata).reshape(2, data_shape['xx'][0], 2, nt_precal_used).astype('complex128')
                    
                    additive_sol = additive_AtNiAi.dot(np.transpose(real_additive_A).dot(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten() * Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()))
                    print ('>>>>>>>>>>>>>additive fitting amp', additive_sol[:n_prefit_amp])
                    # additive_term_incr_tbin = additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]))[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()
                    additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = real_additive_A[:, n_prefit_amp:].dot(additive_sol[n_prefit_amp:]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
                    data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] -= additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
                    additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] += additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
                    try:
                        print ("additive fraction", la.norm(additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) / la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()))
                    except:
                        print('No additive fraction printed.')
                
                # cadd = get_complex_data(additive_term)
                
                if pre_ampcal:
                    data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) / additive_sol[:n_prefit_amp, None]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
                    if comply_ps2mod_autocorr or Use_AbsCal or Use_Fullsim_Noise and not Use_PsAbsCal:
                        pass
                    elif not (Use_PsAbsCal and comply_ps2mod_autocorr):
                        Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) * additive_sol[:n_prefit_amp, None] ** 2).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
                    additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) / additive_sol[:n_prefit_amp, None]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
                    
                    print('Additive_sol: %s' % additive_sol[:n_prefit_amp])
    
    cadd = get_complex_data(additive_term, nubl=nUBL_used, nt=nt_used)
    
    try:
        print 'saving data to', os.path.dirname(data_filename) + '/' + INSTRUMENT + tag + datatag + vartag + '_gsmcal_n%i_bn%i_nubl%s_nt%s-mtbin%s-mfbin%s-tbin%s.npz' % (nside_standard, bnside, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none')
        np.savez(os.path.dirname(data_filename) + '/' + INSTRUMENT + tag + datatag + vartag + '_gsmcal_n%i_bn%i_%s_%s-mtbin%s-mfbin%s-tbin%s.npz' % (nside_standard, bnside, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'),
                 data=data,
                 simdata=stitch_complex_data(fullsim_vis),
                 psdata=[stitch_complex_data(vis) for vis in pt_vis],
                 pt_sources=pt_sources,
                 ubls=used_common_ubls,
                 tlist=lsts,
                 Ni=Ni,
                 freq=freq)
    except:
        print('Error when Saving Calibrated Results Package.')
    
    try:
        if plot_data_error:
            # plt.clf()
            
            cdata = get_complex_data(data, nubl=nUBL_used, nt=nt_used)
            crdata = get_complex_data(raw_data, nubl=nUBL_used, nt=nt_used)  # / (additive_sol[0] * (pre_ampcal) + (not pre_ampcal))
            cNi = get_complex_data(Ni, nubl=nUBL_used, nt=nt_used)
            
            fun = np.abs
            srt = sorted((lsts - lst_offset) % 24. + lst_offset)
            asrt = np.argsort((lsts - lst_offset) % 24. + lst_offset)
            pncol = min(int(60. / (srt[-1] - srt[0])), 12) if nt_used > 1 else (len(ubl_sort['x']) / 2)
            us = ubl_sort['x'][::len(ubl_sort['x']) / pncol] if len(ubl_sort['x']) / pncol >= 1 else ubl_sort['x']
            figure = {}
            for p in range(2):
                for nu, u in enumerate(us):
                    plt.figure(5000 + 100 * p + nu)
                    # plt.subplot(5, (len(us) + 4) / 5, nu + 1)
                    figure[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
                    figure[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
                    figure[3], = plt.plot(srt, fun(crdata[u, p][asrt]), '+', label='raw_data')
                    figure[4], = plt.plot(srt, fun(cNi[u, p][asrt]) ** -.5, label='Ni')
                    if pre_calibrate:
                        figure[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
                        data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p]))),
                    else:
                        data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
                    plt.yscale('log')
                    plt.title("%s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
                    plt.ylim([-1.05 * data_range, 1.05 * data_range])
                    if pre_calibrate:
                        plt.legend(handles=[figure[1], figure[2], figure[3], figure[4], figure[5]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise', 'additive'], loc=0)
                    else:
                        plt.legend(handles=[figure[1], figure[2], figure[3], figure[4]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise'], loc=0)
                    plt.savefig(
                        script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Abs_Full_vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard))
                    plt.show(block=False)
            
            fun = np.angle
            for p in range(2):
                for nu, u in enumerate(us):
                    plt.figure(50000 + 100 * p + nu)
                    # plt.subplot(5, (len(us) + 4) / 5, nu + 1)
                    figure[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
                    figure[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
                    figure[3], = plt.plot(srt, fun(crdata[u, p][asrt]), '+', label='raw_data')
                    figure[4], = plt.plot(srt, fun(cNi[u, p][asrt]) ** -.5, label='Ni')
                    if pre_calibrate:
                        figure[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
                        data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p]))),
                    else:
                        data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
                    # plt.yscale('log')
                    plt.title("%s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
                    plt.ylim([-1.05 * data_range, 1.05 * data_range])
                    if pre_calibrate:
                        plt.legend(handles=[figure[1], figure[2], figure[3], figure[4], figure[5]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise', 'additive'], loc=0)
                    else:
                        plt.legend(handles=[figure[1], figure[2], figure[3], figure[4]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise'], loc=0)
                    plt.savefig(
                        script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Angle_Full_vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard))
                    plt.show(block=False)
        # plt.gcf().clear()
        # plt.clf()
        # plt.close()
    except:
        print('Error when Plotting Calibrated Results.')
    
    if pre_calibrate:
        return data, Ni, additive_A, additive_term, additive_sol, precal_time_bin
    else:
        return additive_A, additive_term, precal_time_bin
