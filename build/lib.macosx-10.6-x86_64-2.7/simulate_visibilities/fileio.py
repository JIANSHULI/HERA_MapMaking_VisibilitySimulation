'''XXX DOCSTRING'''
# XXX lots of imports... are all necessary?  can code be separated into files with smaller dependency lists?
# XXX this file has gotten huge. need to break into smaller files
# XXX clean house on commented code?
# XXX obey python style conventions
import math, random, traceback, ephem, string, commands, shutil, resource, threading, time
import multiprocessing as mp
from time import ctime
import aipy as ap
import struct
import numpy as np
import os, sys
import _omnical as _O
from info import RedundantInfo
import warnings
from array import array
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import scipy as sp
    import scipy.sparse as sps
    import scipy.linalg as la
    import scipy.signal as ss
    import scipy.ndimage.filters as sfil
    from scipy import interpolate
    try:
        from numpy import nanmedian as nanmedian
    except:
        print "WARNING: using scipy's nanmedian function with is much slower than numpy.nanmedian. Consider numpy 1.9+."
        from scipy.stats import nanmedian

__version__ = '4.0.4'

julDelta = 2415020.# =julian date - pyephem's Observer date
PI = np.pi
TPI = 2 * np.pi

# XXX this probably belongs in a different file; it's not omnical, it's fileio.
def importuvs(uvfilenames, wantpols, totalVisibilityId = None, nTotalAntenna = None, lat = None, lon = None, timingTolerance = math.pi/12/3600/100, init_mem = 4.e9, verbose = False):
    '''tolerance of timing in radians in lst. init_mem is the initial memory it allocates for reading uv files. return lst in sidereal hour'''

    METHODNAME = "*importuvs*"

    ###############sanitize inputs################################
    uvfilenames = [os.path.expanduser(uvfilename) for uvfilename in uvfilenames]
    for uvfilename in uvfilenames:
        if not (os.path.isdir(uvfilename) and os.path.isfile(uvfilename + '/visdata')):
            raise IOError("UV file %s does not exit or is not a valid MIRIAD UV file."%uvfilename)

    ############################################################
    sun = ephem.Sun()
    #julDelta = 2415020
    ####get some info from the first uvfile####################
    uv=ap.miriad.UV(uvfilenames[0])
    nfreq = uv.nchan;
    if nTotalAntenna is None:
        nant = uv['nants'] # 'nants' should be the number of dual-pol antennas. PSA32 has a bug in double counting
    else:
        nant = nTotalAntenna

    if totalVisibilityId is None:
        totalVisibilityId = np.concatenate([[[i,j] for i in range(j + 1)] for j in range(nant)])
    elif nant * (nant + 1) / 2 < len(totalVisibilityId):
        raise Exception("FATAL ERROR: Total number of antenna %d implies %d baselines whereas the length of totalVisibilityId is %d."%(nant, nant * (nant + 1) / 2, len(totalVisibilityId)))
    startfreq = uv['sfreq']
    dfreq = uv['sdf']

    sa = ephem.Observer()
    if lon is None:
        sa.lon = uv['longitu']
    else:
        sa.lon = lon
    if lat is None:
        sa.lat = uv['latitud']
    else:
        sa.lat = lat

    del(uv)

    #######compute bl1dmatrix####each entry is 1 indexed with minus meaning conjugate
    bl1dmatrix = np.zeros((nant, nant), dtype = 'int32')
    for a1a2, bl in zip(totalVisibilityId, range(len(totalVisibilityId))):
        a1, a2 = a1a2
        bl1dmatrix[a1, a2] = bl + 1
        bl1dmatrix[a2, a1] = - (bl + 1)
    ####prepare processing
    deftime = int(init_mem / 8. / nfreq / len(wantpols)/ len(totalVisibilityId))#use 4GB of memory by default.
    if verbose:
        print "Declaring initial array shape (%i, %i, %i, %i)..."%(deftime, len(wantpols), len(totalVisibilityId), nfreq),
    sys.stdout.flush()
    try:
        data = np.zeros((deftime, len(wantpols), len(totalVisibilityId), nfreq), dtype = 'complex64')
        flags = np.zeros((deftime, len(wantpols), len(totalVisibilityId), nfreq), dtype = 'bool')
    except MemoryError:
        raise Exception("Failed to allocate %.2fGB of memory. Set init_mem keyword in Bytes for importuvs() to decrease initial memory allocation."%(init_mem/1.074e9))
    if verbose:
        print "Done."
    sys.stdout.flush()
    #sunpos = np.zeros((deftime, 2))
    t = []#julian date
    timing = []#local time string
    lst = []#in units of sidereal hour

    ###start processing
    datapulled = False
    for uvfile in uvfilenames:
        uv = ap.miriad.UV(uvfile)
        if len(timing) > 0:
            print FILENAME + METHODNAME + "MSG:",  timing[-1]#uv.nchan
        #print FILENAME + " MSG:",  uv['nants']

        for p, pol in enumerate(wantpols.keys()):
            uv.rewind()
            uv.select('clear', 0, 0)
            uv.select('polarization', wantpols[pol], 0, include=True)
            pol_exist = False
            current_t = None
            if p == 0:#need time extracting shananigans
                for preamble, rawd, flag in uv.all(raw=True):
                    pol_exist = True

                    if len(t) < 1 or t[-1] != preamble[1]:#first bl of a timeslice
                        t += [preamble[1]]
                        sa.date = preamble[1] - julDelta
                        #sun.compute(sa)
                        timing += [sa.date.__str__()]
                        if abs((uv['lst'] - float(sa.sidereal_time()) + math.pi)%(2*math.pi) - math.pi) >= timingTolerance:
                            raise Exception("Error: uv['lst'] is %f radians whereas time computed by ephem is %f radians, the difference is larger than tolerance of %f."%(uv['lst'], float(sa.sidereal_time()), timingTolerance))
                        else:
                            lst += [(float(sa.sidereal_time()) * 24./2./math.pi)]
                        if len(t) > len(data):
                            print FILENAME + METHODNAME + "MSG:",  "expanding number of time slices from", len(data), "to", len(data) + deftime
                            data = np.concatenate((data, np.zeros((deftime, data.shape[1], data.shape[2], data.shape[3]), dtype = 'complex64')))
                            flags = np.concatenate((flags, np.zeros((deftime, flags.shape[1], flags.shape[2], flags.shape[3]), dtype = 'bool')))
                            #sunpos = np.concatenate((sunpos, np.zeros((deftime, 2))))
                            #sunpos[len(t) - 1] = np.asarray([[sun.alt, sun.az]])
                    current_t = len(t) - 1

                    a1, a2 = preamble[2]
                    bl = bl1dmatrix[a1, a2]#bl is 1 indexed with minus meaning conjugate
                    datapulled = True
                    #print info[p]['subsetbl'][info[p]['crossindex'][bl]],
                    data[current_t, p, abs(bl) - 1] = (np.real(rawd) + 1.j * np.sign(bl) * np.imag(rawd))
                    flags[current_t, p, abs(bl) - 1] = flag
            else:
                for preamble, rawd, flag in uv.all(raw=True):
                    pol_exist = True
                    if current_t is None or t[current_t] != preamble[1]:
                        try:
                            current_t = t.index(preamble[1])
                        except ValueError:
                            raise ValueError("Julian date %f for %s does not exist in %s for file %s."%(preamble[1], pol, wantpols.keys()[0], uvfile))

                    a1, a2 = preamble[2]
                    bl = bl1dmatrix[a1, a2]#bl is 1 indexed with minus meaning conjugate
                    datapulled = True
                    #print info[p]['subsetbl'][info[p]['crossindex'][bl]],
                    data[current_t, p, abs(bl) - 1] = (np.real(rawd) + 1.j * np.sign(bl) * np.imag(rawd))
                    flags[current_t, p, abs(bl) - 1] = flag
            if not pol_exist:
                raise IOError("Polarization %s does not exist in uv file %s."%(pol, uvfile))
        #currentpol = 0
        #for preamble, rawd in uv.all():
            #if len(t) < 1 or t[-1] != preamble[1]:#first bl of a timeslice
                #t += [preamble[1]]
                #sa.date = preamble[1] - julDelta
                ##sun.compute(sa)
                #timing += [sa.date.__str__()]
                #if abs((uv['lst'] - float(sa.sidereal_time()) + math.pi)%(2*math.pi) - math.pi) >= timingTolerance:
                    #raise Exception("Error: uv['lst'] is %f radians whereas time computed by ephem is %f radians, the difference is larger than tolerance of %f."%(uv['lst'], float(sa.sidereal_time()), timingTolerance))
                #else:
                    #lst += [(float(sa.sidereal_time()) * 24./2./math.pi)]
                #if len(t) > len(data):
                    #print FILENAME + METHODNAME + " MSG:",  "expanding number of time slices from", len(data), "to", len(data) + deftime
                    #data = np.concatenate((data, np.zeros((deftime, len(wantpols), nant * (nant + 1) / 2, nfreq), dtype = 'complex64')))
                    ##sunpos = np.concatenate((sunpos, np.zeros((deftime, 2))))
                    ##sunpos[len(t) - 1] = np.asarray([[sun.alt, sun.az]])
            #for p, pol in zip(range(len(wantpols)), wantpols.keys()):
                #if wantpols[pol] == uv['pol']:#//todo: use select()
                    #a1, a2 = preamble[2]
                    #bl = bl1dmatrix[a1, a2]#bl is 1 indexed with minus meaning conjugate
                    #datapulled = True
                    ##print info[p]['subsetbl'][info[p]['crossindex'][bl]],
                    #data[len(t) - 1, p, abs(bl) - 1] = (np.real(rawd.data) + 1.j * np.sign(bl) * np.imag(rawd.data)).astype('complex64')
        del(uv)
        if not datapulled:
            raise IOError("FATAL ERROR: no data pulled from " + uvfile + ", check polarization information! Exiting.")
    reorder = (1, 0, 3, 2)
    return np.transpose(data[:len(t)],reorder), t, timing, lst, np.transpose(flags[:len(t)],reorder)

def pick_slice_uvs(uvfilenames, pol_str_or_num, t_index_lst_jd, findex, totalVisibilityId = None, nTotalAntenna = None, timingTolerance = 100, verbose = False):
    '''tolerance of timing in radians in lst.'''
    METHODNAME = "*pick_slice_uvs*"

    ###############sanitize inputs################################
    uvfilenames = [os.path.expanduser(uvfilename) for uvfilename in uvfilenames]
    for uvfilename in uvfilenames:
        if not (os.path.isdir(uvfilename) and os.path.isfile(uvfilename + '/visdata')):
            raise IOError("UV file %s does not exit or is not a valid MIRIAD UV file."%uvfilename)

    try:
        try:
            pnum = int(pol_str_or_num)
            pol = ap.miriad.pol2str[pnum]
        except ValueError:
            pol = pol_str_or_num
            pnum = ap.miriad.str2pol[pol]
    except KeyError:
        raise ValueError("Invalid polarization option %s. Need to be a string like 'xx', 'xy' or a valid MIRIAD pol number like -5 or -6."%pnum)
    ############################################################
    sun = ephem.Sun()

    ####get some info from the first uvfile####################
    uv=ap.miriad.UV(uvfilenames[0])
    nfreq = uv.nchan;
    if nTotalAntenna is None:
        nant = uv['nants'] # 'nants' should be the number of dual-pol antennas. PSA32 has a bug in double counting
    else:
        nant = nTotalAntenna

    if totalVisibilityId is None:
        totalVisibilityId = np.concatenate([[[i,j] for i in range(j + 1)] for j in range(nant)])
    elif nant * (nant + 1) / 2 < len(totalVisibilityId):
        raise Exception("FATAL ERROR: Total number of antenna %d implies %d baselines whereas the length of totalVisibilityId is %d."%(nant, nant * (nant + 1) / 2, len(totalVisibilityId)))
    startfreq = uv['sfreq']
    dfreq = uv['sdf']

    sa = ephem.Observer()
    sa.lon = uv['longitu']
    sa.lat = uv['latitud']
    del(uv)

    if findex >= nfreq:
        raise IOError("UV file indicates that it has %i frequency channels, so the input findex %i is invalid."%(nfreq, findex))
    #######compute bl1dmatrix####each entry is 1 indexed with minus meaning conjugate
    bl1dmatrix = np.zeros((nant, nant), dtype = 'int32')
    for a1a2, bl in zip(totalVisibilityId, range(len(totalVisibilityId))):
        a1, a2 = a1a2
        bl1dmatrix[a1, a2] = bl + 1
        bl1dmatrix[a2, a1] = - (bl + 1)
    ####prepare processing
    data = np.zeros(len(totalVisibilityId), dtype = 'complex64')
    #sunpos = np.zeros((deftime, 2))
    t = []#julian date
    timing = []#local time string
    lst = []#in units of sidereal hour

    ###start processing
    datapulled = False
    for uvfile in uvfilenames:
        uv = ap.miriad.UV(uvfile)
        if len(timing) > 0:
            print FILENAME + METHODNAME + "MSG:",  timing[-1]#uv.nchan
        #print FILENAME + " MSG:",  uv['nants']

        uv.rewind()
        uv.select('clear', 0, 0)
        uv.select('polarization', pnum, 0, include=True)
        pol_exist = False
        current_t = None
        for preamble, rawd in uv.all():
            pol_exist = True

            if len(t) < 1 or t[-1] != preamble[1]:#first bl of a timeslice
                if datapulled:
                    break
                t += [preamble[1]]
                sa.date = preamble[1] - julDelta
                #sun.compute(sa)
                timing += [sa.date.__str__()]
                if abs((uv['lst'] - float(sa.sidereal_time()) + math.pi)%(2*math.pi) - math.pi) >= timingTolerance:
                    raise Exception("Error: uv['lst'] is %f radians whereas time computed by ephem is %f radians, the difference is larger than tolerance of %f."%(uv['lst'], float(sa.sidereal_time()), timingTolerance))
                else:
                    lst += [(float(sa.sidereal_time()) * 24./2./math.pi)]
                    #sunpos = np.concatenate((sunpos, np.zeros((deftime, 2))))
                    #sunpos[len(t) - 1] = np.asarray([[sun.alt, sun.az]])

                if (type(t_index_lst_jd) is type(0.)) and (t_index_lst_jd == t[-1] or t_index_lst_jd == lst[-1]):
                    datapulled = True
                elif type(t_index_lst_jd) is type(0) and t_index_lst_jd == len(t)-1:
                    datapulled = True

            if datapulled:
                a1, a2 = preamble[2]
                bl = bl1dmatrix[a1, a2]#bl is 1 indexed with minus meaning conjugate
                data[abs(bl) - 1] = (np.real(rawd.data[findex]) + 1.j * np.sign(bl) * np.imag(rawd.data[findex])).astype('complex64')

            if not pol_exist:
                raise IOError("Polarization %s does not exist in uv file %s."%(pol, uvfile))

        del(uv)
    if not datapulled:
        raise IOError("FATAL ERROR: no data pulled. Total of %i time slices read from UV files. Please check polarization information."%len(t))
    return data

# XXX also doesn't belong in this file.  it's fileio
def exportuv(uv_path, data, flags, pols, jds, inttime, sfreq, sdf, latitude, longitude, totalVisibilityId = None, comment='none', overwrite = False):
    '''flags true when bad, lat lon radians, frequency GHz, jd days, inttime seconds, pols in -5~-8 miriad convention'''
    uv_path = os.path.expanduser(uv_path)
    if os.path.isdir(uv_path):
        if overwrite:
            try:
                shutil.rmtree(uv_path)
            except:
                raise IOError('Output path %s exists and overwrite failed.'%uv_path)
        else:
            raise IOError('Output path %s exists. Use --overwrite if you wish to overwrite.'%uv_path)

    if data.ndim != 4:
        raise TypeError('Data has %i dimensions, not 4 dimensions (pol, time, freq, baseline).'%(data.ndim))
    if totalVisibilityId is None:
        nant = int(np.floor((data.shape[-1] * 2)**.5))
        if data.shape[-1] != nant * (nant + 1) / 2:
            raise TypeError('totalVisibilityId is not supplied and baseline dimension in data is %i which cannot be automatically translated into nAntenna.'%data.shape[-1])
        totalVisibilityId = np.concatenate([[[i,j] for i in range(j + 1)] for j in range(nant)])
    if (data.shape[1] != len(jds)):
        raise TypeError('data and jds have different time lengths %i %i.'%(data.shape[1], len(jds)))
    if (data.shape[0] != len(pols)):
        raise TypeError('data and pols have different pol lengths %i %i.'%(data.shape[0], len(pols)))
    if (data.shape[3] != len(totalVisibilityId)):
        raise TypeError('data and totalVisibilityId have different baseline counts %i %i.'%(data.shape[3], len(totalVisibilityId)))
    if (data.shape != flags.shape):
        raise TypeError('data and flags have different dimensions %s %s.'%(data.shape, flags.shape))
    for pnum in pols:
        if type(pnum) != type(1):
            raise TypeError('pols must be in miriad numbers like -5 for xx. A %s is passed in.'%pnum)


    uv = ap.miriad.UV(uv_path, 'new')
    uv['history'] = 'Made this file from scratch using omnical.calibration_omni.exportuv() on %s. Initial comments: %s.\n'%(time.asctime(time.localtime(time.time())), comment)

    uv.add_var('pol', 'i')
    uv.add_var('inttime', 'r')
    uv.add_var('latitud', 'd')
    uv.add_var('longitu', 'd')
    uv.add_var('lst', 'd')
    uv.add_var('nants', 'i')
    uv.add_var('nchan', 'i')
    uv.add_var('npol', 'i')
    uv.add_var('sdf', 'd')#GHZ
    uv.add_var('sfreq', 'd')#GHZ


    uv['nchan'] = data.shape[2]
    uv['npol'] = data.shape[0]
    uv['nants'] = np.max(np.array(totalVisibilityId)) + 1
    uv['latitud'] = latitude
    uv['longitu'] = longitude
    uv['sdf'] = sdf
    uv['sfreq'] = sfreq
    uv['inttime'] = inttime
    sa = ephem.Observer()
    sa.lat = latitude
    sa.lon = longitude

    for p, pnum in enumerate(pols):
        uv['pol'] = pnum
        for t, jd in enumerate(jds):
            sa.date = jd - julDelta
            uv['lst'] = sa.sidereal_time()
            for bl, antpair in enumerate(totalVisibilityId):
                uvw = np.array([antpair[0]-antpair[1]-1,0,0], dtype=np.double)

                preamble = (uvw, jd, (antpair[0],antpair[1]))
                if antpair[0] == antpair[1]:
                    maskdata = np.ma.masked_array(np.conjugate(data[p, t, :, bl]), mask = flags[p, t, :, bl], dtype='complex64')
                else:
                    maskdata = np.ma.masked_array(data[p, t, :, bl], mask = flags[p, t, :, bl], dtype='complex64')
                uv.write(preamble, maskdata)
    del(uv)
    return

# XXX fileio
def get_uv_pols(uvi):
    '''XXX DOCSTRING'''
    input_is_str = (type(uvi) == type('a'))
    if input_is_str:
        uvi = ap.miriad.UV(uvi)

    uvpols = []
    npols = uvi['npol']
    uvi.rewind()
    uvi.select('clear', 0, 0)
    for preamble, data in uvi.all():
        if uvi['pol'] not in uvpols:
            uvpols = uvpols + [uvi['pol']]
            if len(uvpols) == npols:
                break
    uvi.rewind()
    uvi.select('clear', 0, 0)

    if input_is_str:
        del(uvi)
    return [ap.miriad.pol2str[p] for p in uvpols]

# XXX fileio
def apply_omnigain_uvs(uvfilenames, omnigains, totalVisibilityId, info, oppath, ano, adds={}, flags=None, nTotalAntenna = None, overwrite = False, comment = '', verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "*apply_omnigain_uvs*"

    for key in omnigains.keys():
        if key not in ['x','y']:
            raise KeyError("Unsupported key for omnigains: %s. Only 'x' and 'y' are supported."%key)

    ttotal = len(omnigains[omnigains.keys()[0]])
    ftotal = int(omnigains[omnigains.keys()[0]][0,0,3])

    newflag = np.zeros((ttotal, ftotal), dtype=bool)
    if flags is not None:
        for pol in flags.keys():
            if flags[pol].shape != (ttotal, ftotal):
                raise TypeError("flags file on %s has shape %s which does not agree with omnigains shape %s."%(pol, flags[pol].shape, (ttotal, ftotal)))
            newflag = newflag|flags[pol]

    if adds != {}:
        if (ttotal != len(adds[adds.keys()[0]]) or ftotal != len(adds[adds.keys()[0]][0]) or len(totalVisibilityId) != len(adds[adds.keys()[0]][0,0])):
            raise Exception("Error: additives have different nTime or nFrequency or number of baseline!")
    if info.keys() != omnigains.keys():
        raise Exception("Error: info and calparfilenames cannot have different sets of polarizations!")

    ####get some info from the first uvfile
    uv=ap.miriad.UV(uvfilenames[0])
    uvpols = get_uv_pols(uv)
    uvpols = [uvpol for uvpol in uvpols if uvpol in ['xx', 'yy', 'xy', 'yx']]
    nfreq = uv.nchan;
    if nfreq != ftotal:
        raise Exception("Error: uv file %s and omnigains have different nFrequency!"%uvfilenames[0])
    if nTotalAntenna is None:
        nant = uv['nants'] # 'nants' should be the number of dual-pol antennas. PSA32 has a bug in double counting
    else:
        nant = nTotalAntenna

    if nant * (nant + 1) / 2 < len(totalVisibilityId):
        raise Exception("FATAL ERROR: Total number of antenna %d implies %d baselines whereas the length of totalVisibilityId is %d."%(nant, nant * (nant + 1) / 2, len(totalVisibilityId)))
    startfreq = uv['sfreq']
    dfreq = uv['sdf']
    del(uv)

    #######compute bl1dmatrix####each entry is 1 indexed with minus meaning conjugate, the bl here is the number in totalVisibilityId
    bl1dmatrix = np.zeros((nant, nant), dtype = 'int32')

    for a1a2, bl in zip(totalVisibilityId, range(len(totalVisibilityId))):
        a1, a2 = a1a2
        bl1dmatrix[a1, a2] = bl + 1
        bl1dmatrix[a2, a1] = - (bl + 1)
    ####load calpar from omnigain
    calpars = {}#bad antenna included
    badants = {}
    for key in ['x', 'y']:
        calpars[key] = np.ones((ttotal, nant, nfreq),dtype='complex64')
        badants[key] = np.zeros(nant, dtype='bool')
        if key in omnigains.keys():
            badants[key] = np.ones(nant, dtype='bool')
            badants[key][info[key]['subsetant']] = False
            calpars[key][:,info[key]['subsetant'],:] = omnigains[key][:,:,4::2] + 1.j * omnigains[key][:,:,5::2]



    #########start processing#######################
    t = []
    timing = []
    #datapulled = False
    for uvfile in uvfilenames:
        uvi = ap.miriad.UV(uvfile)

        if oppath is None:
            oppath = os.path.abspath(os.path.dirname(os.path.dirname(uvfile + '/'))) + '/'
        opuvname = oppath + os.path.basename(os.path.dirname(uvfile+'/')) + ano + 'O'
        if verbose:
            print FILENAME + METHODNAME + "MSG: Creating %s"%opuvname
        if os.path.isdir(opuvname):
            if overwrite:
                shutil.rmtree(opuvname)
            else:
                raise IOError("%s already exists. Use overwrite option to overwrite."%opuvname)
        uvo = ap.miriad.UV(opuvname, status='new')
        uvo.init_from_uv(uvi)
        historystr = "Applied OMNICAL on %s: "%time.asctime(time.localtime(time.time()))
        uvo['history'] += historystr + comment + "\n"

        for p, pol in enumerate(uvpols):
            uvi.rewind()
            uvi.select('clear', 0, 0)
            uvi.select('polarization', ap.miriad.str2pol[pol], 0, include=True)
            current_t = None
            if p == 0:#need time extracting shananigans
                if len(timing) > 0:
                    if verbose:
                        print FILENAME + METHODNAME + "MSG:", uvfile + ' after', timing[-1]#uv.nchan
                        sys.stdout.flush()
                for preamble, data, flag in uvi.all(raw=True):
                    uvo.copyvr(uvi)
                    if len(t) < 1 or t[-1] != preamble[1]:#first bl of a timeslice
                        t += [preamble[1]]

                        if len(t) > ttotal:
                            raise Exception(FILENAME + METHODNAME + " MSG: FATAL ERROR: omnigain input array has length", omnigains[0].shape, "but the total length is exceeded when processing " + uvfile + " Aborted!")
                    a1, a2 = preamble[2]
                    if pol in adds.keys():
                        bl = bl1dmatrix[a1, a2]
                        if bl > 0:
                            additive = adds[pol][len(t) - 1, :, bl - 1]
                        elif bl < 0:
                            additive = adds[pol][len(t) - 1, :, - bl - 1].conjugate()
                        else:
                            additive = 0
                            flag[:] = True
                    else:
                        additive = 0
                    #print data.shape, additive.shape, calpars[pol][len(t) - 1, a1].shape
                    uvo.write(preamble, (data-additive)/calpars[pol[0]][len(t) - 1, a1].conjugate()/calpars[pol[1]][len(t) - 1, a2], badants[pol[0]][a1]|badants[pol[1]][a2]|flag|newflag[len(t) - 1])
            else:
                for preamble, data, flag in uvi.all(raw=True):
                    if current_t is None or t[current_t] != preamble[1]:
                        try:
                            current_t = t.index(preamble[1])
                        except ValueError:
                            raise ValueError("Julian date %f for %s does not exist in %s for file %s."%(preamble[1], pol, wantpols.keys()[0], uvfile))
                    uvo.copyvr(uvi)
                    a1, a2 = preamble[2]
                    if pol in adds.keys():
                        bl = bl1dmatrix[a1, a2]
                        if bl > 0:
                            additive = adds[pol][current_t, :, bl - 1]
                        elif bl < 0:
                            additive = adds[pol][current_t, :, - bl - 1].conjugate()
                        else:
                            additive = 0
                            flag[:] = True
                    else:
                        additive = 0
                    #print data.shape, additive.shape, calpars[pol][len(t) - 1, a1].shape
                    uvo.write(preamble, (data-additive)/calpars[pol[0]][current_t, a1].conjugate()/calpars[pol[1]][current_t, a2], badants[pol[0]][a1]|badants[pol[1]][a2]|flag|newflag[current_t])

        del(uvo)
        del(uvi)

    return


# XXX fileio
def apply_omnical_uvs(uvfilenames, calparfilenames, totalVisibilityId, info, wantpols, oppath, ano, additivefilenames = None, nTotalAntenna = None, comment = '', overwrite= False):
    '''XXX DOCSTRING'''
    METHODNAME = "*apply_omnical_uvs*"
    if len(additivefilenames) != len(calparfilenames) and additivefilenames is not None:
        raise Exception("Error: additivefilenames and calparfilenames have different lengths!")
    if len(info) != len(calparfilenames):
        raise Exception("Error: info and calparfilenames have different lengths!")
    if additivefilenames is None:
        additivefilenames = ["iDontThinkYouHaveAFileCalledThis" for _ in calparfilenames]

    ####get some info from the first uvfile
    uv=ap.miriad.UV(uvfilenames[0])
    nfreq = uv.nchan;
    if nTotalAntenna is None:
        nant = uv['nants'] # 'nants' should be the number of dual-pol antennas. PSA32 has a bug in double counting
    else:
        nant = nTotalAntenna

    if nant * (nant + 1) / 2 < len(totalVisibilityId):
        raise Exception("FATAL ERROR: Total number of antenna %d implies %d baselines whereas the length of totalVisibilityId is %d."%(nant, nant * (nant + 1) / 2, len(totalVisibilityId)))
    startfreq = uv['sfreq']
    dfreq = uv['sdf']
    del(uv)

    #######compute bl1dmatrix####each entry is 1 indexed with minus meaning conjugate, the bl here is not number in totalVisibilityId, but in info['subsetbl'], so it's different from bl1dmatrix in import_uvs method. it also has 2 pols
    bl1dmatrix = [np.zeros((nant, nant), dtype = 'int32') for p in range(len(info))]

    for a1a2, bl in zip(totalVisibilityId, range(len(totalVisibilityId))):
        a1, a2 = a1a2
        for p in range(len(info)):
            for sbl, bl2 in zip(range(len(info[p]['subsetbl'])), info[p]['subsetbl']):
                if bl == bl2:
                    bl1dmatrix[p][a1, a2] = sbl + 1
                    bl1dmatrix[p][a2, a1] = - (sbl + 1)
                    break
    ####load calpar and check dimensions, massage calpar from txfx(3+2a+2u) to t*goodabl*f
    calpars = []#bad antenna included
    adds = []#badubl not included
    for p in range(len(wantpols)):
        calpar = np.fromfile(calparfilenames[p], dtype='float32')
        if len(calpar)%(nfreq *( 3 + 2 * (info[p]['nAntenna'] + info[p]['nUBL']))) != 0:
            print FILENAME + METHODNAME + " MSG:",  "FATAL ERROR: calpar input array " + calparfilenames[p] + " has length", calpar.shape, "which is not divisible by ", nfreq, 3 + 2 * (info[p]['nAntenna'] + info[p]['nUBL']), "Aborted!"
            return
        ttotal = len(calpar)/(nfreq *( 3 + 2 * (info[p]['nAntenna'] + info[p]['nUBL'])))
        calpar = calpar.reshape((ttotal, nfreq, ( 3 + 2 * (info[p]['nAntenna'] + info[p]['nUBL']))))
        calpars.append(1 + np.zeros((ttotal, nant, nfreq),dtype='complex64'))
        calpars[p][:,info[p]['subsetant'],:] = ((10**calpar[:,:,3:3+info[p]['nAntenna']])*np.exp(1.j * calpar[:,:,3+info[p]['nAntenna']:3+2*info[p]['nAntenna']] * math.pi / 180)).transpose((0,2,1))

        if os.path.isfile(additivefilenames[p]):
            adds.append(np.fromfile(additivefilenames[p], dtype='complex64').reshape((ttotal, nfreq, len(info[p]['subsetbl']))))
        else:
            adds.append(np.zeros((ttotal, nfreq, len(info[p]['subsetbl']))))

    #########start processing#######################
    t = []
    timing = []
    #datapulled = False
    for uvfile in uvfilenames:
        uvi = ap.miriad.UV(uvfile)
        if len(timing) > 0:
            print FILENAME + METHODNAME + "MSG:", uvfile + ' after', timing[-1]#uv.nchan

        if oppath is None:
            oppath = os.path.abspath(os.path.dirname(os.path.dirname(uvfile + '/'))) + '/'
        opuvname = oppath + os.path.basename(os.path.dirname(uvfile+'/')) + ano + 'O'
        print FILENAME + METHODNAME + "MSG: Creating %s"%opuvname
        if overwrite and os.path.isdir(opuvname):
            shutil.rmtree(opuvname)
        uvo = ap.miriad.UV(opuvname, status='new')
        uvo.init_from_uv(uvi)
        historystr = "Applied OMNICAL %s: "%time.asctime(time.localtime(time.time()))
        #for cpfn, adfn in zip(calparfilenames, additivefilenames):
            #historystr += os.path.abspath(cpfn) + ' ' + os.path.abspath(adfn) + ' '
        uvo['history'] += historystr + comment + "\n"
        for preamble, data, flag in uvi.all(raw=True):
            uvo.copyvr(uvi)
            if len(t) < 1 or t[-1] != preamble[1]:#first bl of a timeslice
                t += [preamble[1]]

                if len(t) > ttotal:
                    print FILENAME + METHODNAME + " MSG: FATAL ERROR: calpar input array " + calparfilenames[p] + " has length", calpar.shape, "but the total length is exceeded when processing " + uvfile + " Aborted!"
                    return
            polwanted = False
            for p, pol in zip(range(len(wantpols)), wantpols.keys()):
                if wantpols[pol] == uvi['pol']:
                    a1, a2 = preamble[2]
                    bl = bl1dmatrix[p][a1][a2]
                    if bl > 0:
                        additive = adds[p][len(t) - 1, :, bl - 1]
                    elif bl < 0:
                        additive = adds[p][len(t) - 1, :, - bl - 1].conjugate()
                    else:
                        additive = 0
                        flag[:] = True
                    #print data.shape, additive.shape, calpars[p][len(t) - 1, a1].shape
                    uvo.write(preamble, (data-additive)/calpars[p][len(t) - 1, a1].conjugate()/calpars[p][len(t) - 1, a2], flag)
                    polwanted = True
                    break
            if not polwanted:
                uvo.write(preamble, data, flag)

        del(uvo)
        del(uvi)
        #if not datapulled:
            #print FILENAME + METHODNAME + " MSG:",  "FATAL ERROR: no data pulled from " + uvfile + ", check polarization information! Exiting."
            #exit(1)
    return

