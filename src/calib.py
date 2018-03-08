'''XXX DOCSTRING'''
# XXX lots of imports... are all necessary?  can code be separated into files with smaller dependency lists?
# XXX clean house on commented code?
# XXX obey python style conventions
import math, ephem, time
import multiprocessing as mp
import struct
import numpy as np
import os, sys
import _omnical as _O
from copy import deepcopy
import info
from arrayinfo import ArrayInfo, ArrayInfoLegacy
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import scipy as sp
    import scipy.linalg as la
    import scipy.ndimage.filters as sfil
    try:
        from numpy import nanmedian as nanmedian
    except:
        print "WARNING: using scipy's nanmedian function with is much slower than numpy.nanmedian. Consider numpy 1.9+."
        from scipy.stats import nanmedian

__version__ = '5.0.2'

julDelta = 2415020.# =julian date - pyephem's Observer date
PI = np.pi

class RedundantInfo(info.RedundantInfo):
    def __init__(self, filename=None):
        info.RedundantInfo.__init__(self, filename)

    def calpar_size(self, nant, nubl, has_chi2ant=True):
        """
        Quickly compute the size of the calpar array.
        """
        return 3 + 2*(nant+nubl) + bool(has_chi2ant)*nant

    def order_data(self, dd):
        '''Create a data array ordered for use in _omnical.redcal.  'dd' is
        a dict whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
        the conjugation convention of the provided data.  'dd' values are 2D arrays
        of (time,freq) data.''' # XXX does time/freq ordering matter.  should data be 2D instead?
        return np.array([dd[bl] if dd.has_key(bl) else dd[bl[::-1]].conj()
            for bl in self.bl_order()]).transpose((1,2,0))

    def pack_calpar(self, calpar, gains=None, vis=None):
        '''Pack gain solutions and/or model visibilities for baseline types into a 'calpar' array that follows
        the internal data order used by omnical.  This function facilitates wrapping _omnical.redcal to
        abstract away internal data ordering when providing initial guesses for antenna gains and
        visibilities for ubls.
        self: a RedundantInfo object
        calpar: the appropriately (time,freq,3+2*(nant+nubl)+nant) shaped array into which gains/vis will be copied
        gains: dictionary of antenna number: gain solution (vs time,freq)
        vis: dictionary of baseline: model visibility (vs time,freq).  baseline is cross-indexed to the appropriate
        baseline, so only one representative of each ubl type should be provided.'''
        assert(calpar.shape[-1] == self.calpar_size(self.nAntenna, len(self.ublcount)))
        nant = self.nAntenna

        if gains is not None:
            for i,ai in enumerate(self.subsetant):
                if not gains.has_key(ai): continue
                amp = np.log10(np.abs(gains[ai])); amp.shape = ((1,) + amp.shape)[-2:]
                # XXX does phs need to be conjugated b/c omnical & aipy don't have same conj convention?
                phs = np.angle(gains[ai]); phs.shape = ((1,) + phs.shape)[-2:] 
                calpar[...,3+i], calpar[...,3+nant+i] = amp, phs
        if vis is not None:
            for (ai,aj),v in vis.iteritems():
                i,j = self.ant_index(ai), self.ant_index(aj)
                n = self.bl1dmatrix[i,j] # index of this bl in self.bl2d
                u = self.bltoubl[n] # index of ubl that this bl corresponds to
                if tuple(self.bl2d[n]) != (i,j): # check if bl reversed wrt ubl
                    assert(tuple(self.bl2d[n]) == (j,i))
                    v = v.conj()
                # XXX possible that frombuffer might do this a bit more efficiently
                calpar[...,3+2*nant+2*u] = v.real # interleave real/imag
                calpar[...,3+2*nant+2*u+1] = v.imag
        return calpar

    def unpack_calpar(self, calpar, res=None):
        '''Parse the calpar result from omnical (complementary to pack_calpar).
        Result is parsed into meta, gains, and vis dicts which are returned.  meta has keys 'iter' and
        'chisq', gains as keys of antenna number, and vis has unique baseline solutions, indexed by a
        representative baseline.'''
        meta, gains, vis = {}, {}, {}
        meta['iter'],meta['chisq'] = calpar[...,0], calpar[...,2]
        chisq_per_ant = calpar[...,self.calpar_size(self.nAntenna, len(self.ublcount), False):]
        for i,ai in enumerate(self.subsetant):
            gains[ai] = 10**calpar[...,3+i] * np.exp(1j*calpar[...,3+self.nAntenna+i])
            meta['chisq%d' % (ai)] = chisq_per_ant[...,i]
        for u in xrange(len(self.ublcount)):
            # XXX possible that frombuffer might do this a bit more efficiently
            v = calpar[...,3+2*self.nAntenna+2*u] + 1j*calpar[...,3+2*self.nAntenna+2*u+1]
            n = self.ublindex[np.sum(self.ublcount[:u])]
            i,j = self.bl2d[n]
            vis[(self.subsetant[i],self.subsetant[j])] = v
        if res is not None:
            res = dict(zip(map(tuple,self.subsetant[self.bl2d]), res.transpose([2,0,1])))
            meta['res'] = res
        return meta, gains, vis

# XXX maybe omnical should only solve one time at a time, so that prev sol can be used as starting point for next time

# calpar inside of _omnical is a flat array of calibration solutions as follows:
# calpar[...,0] = # iterations
# calpar[...,1] = additiveout**2 # XXX don't understand point of this.  maybe how much of chisq is additiveout?
# calpar[...,2] = chisq # XXX maybe break this down by antenna someday
# calpar[...,3:3+nant] = log10(abs(g_i)) # g_i is gain of ant i, where i is internal indexing order in subsetant
# calpar[...,3+nant:3+2*nant] = phase(g_a)
# calpar[...,3+2*nant:3+2*nant+2*nubl] = real,imag component of avg solution for each ubl
# Any other dimensions of the array (which precede the final axis)
# will be over time and frequency, as in the input data array
# XXX is the omnical baseline convention conjugated wrt aipy?  I think so

# def calpar_size(nant, nubl, has_chi2ant=True):
#     """
#     Quickly compute the size of the calpar array.
#     """
#     return 3 + 2*(nant+nubl) + bool(has_chi2ant)*nant
# 
# def pack_calpar(info, calpar, gains=None, vis=None):
#     '''Pack gain solutions and/or model visibilities for baseline types into a 'calpar' array that follows
#     the internal data order used by omnical.  This function facilitates wrapping _omnical.redcal to
#     abstract away internal data ordering when providing initial guesses for antenna gains and
#     visibilities for ubls.
#     info: a RedundantInfo object
#     calpar: the appropriately (time,freq,3+2*(nant+nubl)+nant) shaped array into which gains/vis will be copied
#     gains: dictionary of antenna number: gain solution (vs time,freq)
#     vis: dictionary of baseline: model visibility (vs time,freq).  baseline is cross-indexed to the appropriate
# aseline, so only one representative of each ubl type should be provided.'''
#     assert(calpar.shape[-1] == calpar_size(info.nAntenna, len(info.ublcount)))
#     nant = info.nAntenna
#     if gains is not None:
#         for i,ai in enumerate(info.subsetant):
#             if not gains.has_key(ai): continue
#             amp = np.log10(np.abs(gains[ai])); amp.shape = ((1,) + amp.shape)[-2:]
#             # XXX does phs need to be conjugated b/c omnical & aipy don't have same conj convention?
#             phs = np.angle(gains[ai]); phs.shape = ((1,) + phs.shape)[-2:]
#             calpar[...,3+i], calpar[...,3+nant+i] = amp, phs
#     if vis is not None:
#         for (ai,aj),v in vis.iteritems():
#             i,j = info.ant_index(ai), info.ant_index(aj)
#             n = info.bl1dmatrix[i,j] # index of this bl in info.bl2d
#             u = info.bltoubl[n] # index of ubl that this bl corresponds to
#             if tuple(info.bl2d[n]) != (i,j): # check if bl reversed wrt ubl
#                 assert(tuple(info.bl2d[n]) == (j,i))
#                 v = v.conj()
#             # XXX possible that frombuffer might do this a bit more efficiently
#             calpar[...,3+2*nant+2*u] = v.real # interleave real/imag
#             calpar[...,3+2*nant+2*u+1] = v.imag
#     return calpar
# 
# def unpack_calpar(info, calpar):
#     '''Parse the calpar result from omnical (complementary to pack_calpar).
#     Result is parsed into meta, gains, and vis dicts which are returned.  meta has keys 'iter' and
#     'chisq', gains as keys of antenna number, and vis has unique baseline solutions, indexed by a
#     representative baseline.'''
#     meta, gains, vis = {}, {}, {}
#     meta['iter'],meta['chisq'] = calpar[...,0], calpar[...,2]
#     chisq_per_ant = calpar[...,calpar_size(info.nAntenna, len(info.ublcount), False):]
#     for i,ai in enumerate(info.subsetant):
#         gains[ai] = 10**calpar[...,3+i] * np.exp(1j*calpar[...,3+info.nAntenna+i])
#         meta['chisq%d' % (ai)] = chisq_per_ant[...,i]
#     for u in xrange(len(info.ublcount)):
#         # XXX possible that frombuffer might do this a bit more efficiently
#         v = calpar[...,3+2*info.nAntenna+2*u] + 1j*calpar[...,3+2*info.nAntenna+2*u+1]
#         n = info.ublindex[np.sum(info.ublcount[:u])]
#         i,j = info.bl2d[n]
#         vis[(info.subsetant[i],info.subsetant[j])] = v
#     return meta, gains, vis

def redcal(data, info, xtalk=None, gains=None, vis=None,
        removedegen=False, uselogcal=False, uselincal=False, maxiter=50, conv=1e-3, stepsize=.3, trust_period=1, **kwargs):
    '''Perform redundant calibration, parsing results into meta, gains, and vis dicts which are returned.  This
    function wraps _omnical.redcal to abstract away internal data ordering.  'data' is a dict of measured visibilities,
    indexed by baseline.  Initial guesses for xtalk, antenna gains,
    and unique baselines may be passed in through xtalk, gains, and vis dictionaries, respectively.'''
    data = info.order_data(data) # put data into
    calpar = np.zeros((data.shape[0],data.shape[1], info.calpar_size(info.nAntenna, len(info.ublcount))), dtype=np.float32)
    info.pack_calpar(calpar, gains=gains, vis=vis, **kwargs)
    if xtalk is None: xtalk = np.zeros_like(data) # crosstalk (aka "additivein/out") will be overwritten
    else: xtalk = info.order_data(xtalk)
    res = _O.redcal(data, calpar, info, xtalk,
        removedegen=int(removedegen), uselogcal=int(uselogcal), uselincal=int(uselincal),
        maxiter=int(maxiter), conv=float(conv), stepsize=float(stepsize),
        computeUBLFit=int(vis is None), trust_period=int(trust_period))
    meta, gains, vis = info.unpack_calpar(calpar, res=res, **kwargs)
    return meta, gains, vis

# TODO: wrap _O._redcal to return calpar parsed up sensibly
# considerations: _redcal starts with calpar as a starting place.  Do we need the ability to go
# from parsed solutions back to calpar?  if so, might want an object that holds calpar and parses accordingly


def logcal(data, info, gains=None, xtalk=None, maxiter=50, conv=1e-3, stepsize=.3, trust_period=1):
    '''Perform logcal. Calls redcal() function with logcal=True.'''

    m, g, v = redcal(data, info, gains=gains, uselogcal=True, xtalk=xtalk,
                     conv=conv, stepsize=stepsize,
                     trust_period=trust_period, maxiter=maxiter)

    return m, g, v


def lincal(data, info, gains=None, vis=None, xtalk=None, maxiter=50, conv=1e-3,
           stepsize=.3, trust_period=1):
    '''Perform lincal. Calls redcal() function with lincal=True.'''

    m, g, v = redcal(data, info, gains=gains, vis=vis, uselincal=True, xtalk=xtalk,
                     conv=conv, stepsize=stepsize,
                     trust_period=trust_period, maxiter=maxiter)

    return m, g, v


def removedegen(data, info, gains, vis, **kwargs):
    '''Run removedegen by calling redcal with removedegen=True'''
    # XXX make data an optional parameter into redcal

    m, g, v = redcal(data, info, gains=gains, vis=vis, removedegen=True, **kwargs)

    return m, g, v






# XXX if calpar is parsed into a sensible format, then apply_calpar functions should not be necessary.
def apply_calpar(data, calpar, visibilityID):
    '''apply complex calpar for all antennas onto all bls, calpar's dimension will be assumed to mean: 1D: constant over time and freq; 2D: constant over time; 3D: change over time and freq'''
    assert(calpar.shape[-1] > np.amax(visibilityID) and data.shape[-1] == len(visibilityID))
    if len(calpar.shape) == 3 and len(data.shape) == 3 and calpar.shape[:2] == data.shape[:2]:
        return data/(np.conjugate(calpar[:,:,visibilityID[:,0].astype(int)]) * calpar[:,:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 2 and (len(data.shape) == 3 or len(data.shape) == 2) and calpar.shape[0] == data.shape[-2]:
        return data/(np.conjugate(calpar[:,visibilityID[:,0].astype(int)]) * calpar[:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 1 and len(data.shape) <= 3:
        return data/(np.conjugate(calpar[visibilityID[:,0].astype(int)]) * calpar[visibilityID[:,1].astype(int)])
    else:
        raise Exception("Dimension mismatch! I don't know how to interpret data dimension of " + str(data.shape) + " and calpar dimension of " + str(calpar.shape) + ".")

def apply_calpar2(data, calpar, calpar2, visibilityID):
    '''apply complex calpar for all antennas onto all bls, calpar's dimension will be assumed to mean: 1D: constant over time and freq; 2D: constant over time; 3D: change over time and freq'''
    METHODNAME = "*apply_calpar2*"
    assert(calpar.shape[-1] > np.amax(visibilityID) and data.shape[-1] == len(visibilityID) and calpar.shape == calpar2.shape)
    if len(calpar.shape) == 3 and len(data.shape) == 3 and calpar.shape[:2] == data.shape[:2]:
        return data/(np.conjugate(calpar[:,:,visibilityID[:,0].astype(int)]) * calpar2[:,:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 2 and (len(data.shape) == 3 or len(data.shape) == 2) and calpar.shape[0] == data.shape[-2]:
        return data/(np.conjugate(calpar[:,visibilityID[:,0].astype(int)]) * calpar2[:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 1 and len(data.shape) <= 3:
        return data/(np.conjugate(calpar[visibilityID[:,0].astype(int)]) * calpar2[visibilityID[:,1].astype(int)])
    else:
        raise Exception("Dimension mismatch! I don't know how to interpret data dimension of " + str(data.shape) + " and calpar dimension of " + str(calpar.shape) + ".")

# XXX utility function, should be separate file XXX appears to be unused
#def stdmatrix(length, polydegree):
#    '''to find out the error in fitting y by a polynomial poly(x), one compute error vector by (I-A.(At.A)^-1 At).y, where Aij = i^j. This function returns (I-A.(At.A)^-1 At)'''
#    A = np.array([[i**j for j in range(polydegree + 1)] for i in range(length)], dtype='int')
#    At = A.transpose()
#    return np.identity(length) - A.dot(la.pinv(At.dot(A), cond = 10**(-6)).dot(At))

# XXX appears to be unused
#def omnical2omnigain(omnicalPath, utctimePath, info, outputPath = None):
#    '''outputPath should be a path without extensions like .omnigain which will be appended'''
#    if outputPath is None:
#        outputPath = omnicalPath.replace('.omnical', '')
#
#    #info = redundantCalibrator.info
#
#    if not os.path.isfile(utctimePath):
#        raise Exception("File %s does not exist!"%utctimePath)
#    with open(utctimePath) as f:
#        utctimes = f.readlines()
#    calpars = np.fromfile(omnicalPath, dtype='float32')
#
#    nT = len(utctimes)
#    nF = len(calpars) / nT / (3 + 2 * info['nAntenna'] + 2 * info['nUBL'])
#    #if nF != redundantCalibrator.nFrequency:
#        #raise Exception('Error: time and frequency count implied in the infput files (%d %d) does not agree with those speficied in redundantCalibrator (%d %d). Exiting!'%(nT, nF, redundantCalibrator.nTime, redundantCalibrator.nFrequency))
#    calpars = calpars.reshape((nT, nF, (3 + 2 * info['nAntenna'] + 2 * info['nUBL'])))
#
#    jd = np.zeros((len(utctimes), 2), dtype='float32')#Julian dat is the only double in this whole thing so im storing it in two chunks as float
#    sa = ephem.Observer()
#    for utctime, t in zip(utctimes, range(len(utctimes))):
#        sa.date = utctime
#        jd[t, :] = struct.unpack('ff', struct.pack('d', sa.date + julDelta))
#
#    opchisq = np.zeros((nT, 2 + 1 + 2 * nF), dtype = 'float32')
#    opomnigain = np.zeros((nT, info['nAntenna'], 2 + 1 + 1 + 2 * nF), dtype = 'float32')
#    opomnifit = np.zeros((nT, info['nUBL'], 2 + 3 + 1 + 2 * nF), dtype = 'float32')
#
#    opchisq[:, :2] = jd
#    opchisq[:, 2] = float(nF)
#    #opchisq[:, 3::2] = calpars[:, :, 0]#number of lincal iters
#    opchisq[:, 3:] = calpars[:, :, 2]#chisq which is sum of squares of errors in each visbility
#
#    opomnigain[:, :, :2] = jd[:, None]
#    opomnigain[:, :, 2] = np.array(info['subsetant']).astype('float32')
#    opomnigain[:, :, 3] = float(nF)
#    gains = (10**calpars[:, :, 3:(3 + info['nAntenna'])] * np.exp(1.j * math.pi * calpars[:, :, (3 + info['nAntenna']):(3 + 2 * info['nAntenna'])] / 180)).transpose((0,2,1))
#    opomnigain[:, :, 4::2] = np.real(gains)
#    opomnigain[:, :, 5::2] = np.imag(gains)
#
#    opomnifit[:, :, :2] = jd[:, None]
#    opomnifit[:, :, 2:5] = np.array(info['ubl']).astype('float32')
#    opomnifit[:, :, 5] = float(nF)
#    opomnifit[:, :, 6::2] = calpars[:, :, 3 + 2 * info['nAntenna']::2].transpose((0,2,1))
#    opomnifit[:, :, 7::2] = calpars[:, :, 3 + 2 * info['nAntenna'] + 1::2].transpose((0,2,1))
#
#
#    opchisq.tofile(outputPath + '.omnichisq')
#    opomnigain.tofile(outputPath + '.omnigain')
#    opomnifit.tofile(outputPath + '.omnifit')

def _redcal(data, rawCalpar, Info, additivein, additive_out, removedegen=0, uselogcal=1, maxiter=50, conv=1e-3, stepsize=.3, computeUBLFit=1, trust_period=1):
    # XXX with merge of _omnical.redcal and _omnical.redcal2, is this function still necessary?
    '''same as _O.redcal, but does not return additiveout. Rather it puts additiveout into an inputted container.  this is the target of RedundantCalibrator._redcal_multithread'''
    np_rawCalpar = np.frombuffer(rawCalpar, dtype=np.float32)
    np_rawCalpar.shape=(data.shape[0], data.shape[1], len(rawCalpar) / data.shape[0] / data.shape[1])
    np_additive_out = np.frombuffer(additive_out, dtype=np.complex64)
    np_additive_out.shape = data.shape
    # XXX Why is np_additive_out used here instead of additive_out?
    _O.redcal(data, np_rawCalpar, Info, additivein, np_additive_out, removedegen=removedegen, uselogcal=uselogcal, maxiter=int(maxiter), conv=float(conv), stepsize=float(stepsize), computeUBLFit = int(computeUBLFit), trust_period = int(trust_period))

    #np_additive_out = _O.redcal(data, np_rawCalpar, Info, additivein, removedegen=removedegen, uselogcal=uselogcal, maxiter=int(maxiter), conv=float(conv), stepsize=float(stepsize), computeUBLFit = int(computeUBLFit), trust_period = int(trust_period))
    #additive_out[:len(additive_out)/2] = np.real(np_additive_out.flatten())
    #additive_out[len(additive_out)/2:] = np.imag(np_additive_out.flatten())


#  ___        _              _          _    ___      _ _ _             _
# | _ \___ __| |_  _ _ _  __| |__ _ _ _| |_ / __|__ _| (_) |__ _ _ __ _| |_ ___ _ _
# |   / -_) _` | || | ' \/ _` / _` | ' \  _| (__/ _` | | | '_ \ '_/ _` |  _/ _ \ '_|
# |_|_\___\__,_|\_,_|_||_\__,_\__,_|_||_\__|\___\__,_|_|_|_.__/_| \__,_|\__\___/_|

class RedundantCalibrator:
    # XXX is this class necessary, or could it be replaced by a sensible wrapper around redcal?
    '''This class is the main tool for performing redundant calibration on data sets.
    For a given redundant configuration, say 32 antennas with 3 bad antennas, the
    user should create one instance of Redundant calibrator and reuse it for all data
    collected from that array. In general, upon creating an instance, the user need
    to create the info field of the instance by either computing it or reading it
    from a text file.'''
    def __init__(self, nTotalAnt, info=None):
        self.arrayinfo = ArrayInfoLegacy(nTotalAnt) # XXX if works, clean up
        self.Info = None
        self.removeDegeneracy = True
        self.removeAdditive = False
        self.removeAdditivePeriod = -1
        self.convergePercent = 0.01 #convergence criterion in relative change of chi^2. By default it stops when reaches 0.01, namely 1% decrease in chi^2. # XXX maybe just an arg to lincal w/ default?
        self.maxIteration = 50 #max number of iterations in lincal # XXX maybe just an arg to lincal w/ default?
        self.stepSize = 0.3 #step size for lincal. (0, 1]. < 0.4 recommended. # XXX maybe just an arg to lincal w/ default?
        self.computeUBLFit = True
        self.trust_period = 1 #How many time slices does lincal start from logcal result rather than the previous time slice's lincal result. default 1 means always start from logcal. if 10, it means lincal start from logcal results (or g = 1's) every 10 time slices
        self.nTime = 0
        self.nFrequency = 0
        self.utctime = None
        self.rawCalpar = None
        self.omnichisq = None
        self.omnigain = None
        self.omnifit = None
        if info is not None: # XXX what is the point of leaving info == None?
            if type(info) == str: self.read_redundantinfo(info)
            else: self.Info = info
    def calpar_size(self, nant, nubl, has_chi2ant=True):
        """
        Quickly compute the size of the calpar array.
        """
        return 3 + 2*(nant+nubl) + bool(has_chi2ant)*nant 
    def read_redundantinfo(self, filename, txtmode=False, verbose=False):
        '''redundantinfo is necessary for running redundant calibration. The text file
        should contain 29 lines each describes one item in the info.'''
        self.Info = RedundantInfo(filename=filename, txtmode=txtmode, verbose=verbose)
        self.totalVisibilityId = self.Info.totalVisibilityId # XXX might this raise an exception?
        self._gen_totalVisibilityId_dic()
    def write_redundantinfo(self, filename, overwrite=False, verbose=False):
        #self.Info.tofile(filename, overwrite=overwrite, verbose=verbose)
        self.Info.to_npz(filename)
    #def read_arrayinfo(self, arrayinfopath, verbose = False):
    #    return self.arrayinfo.read_arrayinfo(arrayinfopath,verbose=verbose) # XXX if works, clean up
    def _redcal(self, data, additivein, nthread=None, verbose=False, uselogcal=0, uselincal=0):
        '''for best performance, try setting nthread to larger than number of cores.'''
        #assert(data.ndim == 3 and data.shape[-1] == len(self.arrayinfo.totalVisibilityId))
        #assert(data.shape == additivein.shape) # XXX is this taken care of in wrapper?
        self.nTime, self.nFrequency = data.shape[:2]
        nUBL = len(self.Info.ublcount)
        if uselogcal or self.rawCalpar is None:
            self.rawCalpar = np.zeros((self.nTime, self.nFrequency, self.calpar_size(self.Info.nAntenna, nUBL)), dtype=np.float32)
        assert(self.rawCalpar.shape == (self.nTime,self.nFrequency, self.calpar_size(self.Info.nAntenna, nUBL)))
        if nthread is None: nthread = min(mp.cpu_count() - 1, self.nFrequency)
        if nthread >= 2: return self._redcal_multithread(data, additivein, 0, nthread, verbose=verbose)
        return _O.redcal(data, self.rawCalpar, self.Info,
            additivein, removedegen=int(self.removeDegeneracy), uselogcal=uselogcal, uselincal=uselincal,
            maxiter=int(self.maxIteration), conv=float(self.convergePercent), stepsize=float(self.stepSize),
            computeUBLFit=int(self.computeUBLFit), trust_period=self.trust_period)
    def lincal(self, data, additivein, nthread=None, verbose=False):
        '''XXX DOCSTRING'''
        return self._redcal(data, additivein, nthread=nthread, verbose=verbose, uselogcal=0, uselincal=1)
    def logcal(self, data, additivein, nthread=None, verbose=False):
        '''XXX DOCSTRING'''
        return self._redcal(data, additivein, nthread=nthread, verbose=verbose, uselogcal=1, uselincal=0)
    def _redcal_multithread(self, data, additivein, uselogcal, nthread, verbose = False):
        '''XXX DOCSTRING'''
        nthread = min(nthread, self.nFrequency)
        additiveouts = {}
        np_additiveouts = {}
        rawCalpar = {}
        np_rawCalpar = {}
        threads = {}
        fchunk = {}
        chunk = int(self.nFrequency) / int(nthread)
        excess = int(self.nFrequency) % int(nthread)
        kwarg = {"removedegen": int(self.removeDegeneracy), "uselogcal": uselogcal, "maxiter": int(self.maxIteration), "conv": float(self.convergePercent), "stepsize": float(self.stepSize), "computeUBLFit": int(self.computeUBLFit), "trust_period": self.trust_period}
        for i in range(nthread):
            if excess == 0: fchunk[i] = (i * chunk, min((1 + i) * chunk, self.nFrequency),)
            elif i < excess: fchunk[i] = (i * (chunk+1), min((1 + i) * (chunk+1), self.nFrequency),)
            else: fchunk[i] = (fchunk[i-1][1], min(fchunk[i-1][1] + chunk, self.nFrequency),)
            rawCalpar[i] = mp.RawArray('f', self.nTime * (fchunk[i][1] - fchunk[i][0]) * (self.rawCalpar.shape[2]))
            np_rawCalpar[i] = np.frombuffer(rawCalpar[i], dtype='float32')
            np_rawCalpar[i].shape = (self.rawCalpar.shape[0], fchunk[i][1]-fchunk[i][0], self.rawCalpar.shape[2])
            np_rawCalpar[i][:] = self.rawCalpar[:, fchunk[i][0]:fchunk[i][1]]
            additiveouts[i] = mp.RawArray('f', self.nTime * (fchunk[i][1] - fchunk[i][0]) * len(self.Info.subsetbl) * 2)#factor of 2 for re/im
            np_additiveouts[i] = np.frombuffer(additiveouts[i], dtype='complex64')
            np_additiveouts[i].shape = (data.shape[0], fchunk[i][1]-fchunk[i][0], len(self.Info.subsetbl))
            threads[i] = mp.Process(target = _redcal, args = (data[:,fchunk[i][0]:fchunk[i][1],:], rawCalpar[i], self.Info, additivein[:,fchunk[i][0]:fchunk[i][1],:], additiveouts[i]), kwargs=kwarg)
        if verbose:
            print "Starting %s Process"%cal_name[uselogcal],
            sys.stdout.flush()
        for i in range(nthread):
            if verbose:
                print "#%i"%i,
                sys.stdout.flush()
            threads[i].start()
        if verbose: print "Finished Process",
        for i in range(nthread):
            threads[i].join()
            if verbose: print "#%i"%i,
        if verbose:
            print ""
            sys.stdout.flush()
        self.rawCalpar = np.concatenate([np_rawCalpar[i] for i in range(nthread)],axis=1)
        return np.concatenate([np_additiveouts[i] for i in range(nthread)],axis=1)
    def get_calibrated_data(self, data, additivein = None):
        '''XXX DOCSTRING'''
        if data.ndim != 3 or data.shape != (self.nTime, self.nFrequency, len(self.totalVisibilityId)):
            raise ValueError("Data shape error: it must be a 3D numpy array of dimensions time * frequency * bl(%i, %i, %i)"%(self.nTime, self.nFrequency, len(self.totalVisibilityId)))
        if additivein is not None and data.shape != additivein.shape:
            raise ValueError("Data shape error: data and additivein have different shapes.")
        if data.shape[:2] != self.rawCalpar.shape[:2]:
            raise ValueError("Data shape error: data and self.rawCalpar have different first two dimensions.")
        calpar = np.ones((len(self.rawCalpar), len(self.rawCalpar[0]), self.nTotalAnt), dtype='complex64')
        calpar[:,:,self.Info.subsetant] = (10**(self.rawCalpar[:, :, 3: (3 + self.Info.nAntenna)])) * np.exp(1.j * self.rawCalpar[:, :, (3 + self.Info.nAntenna): (3 + 2 * self.Info.nAntenna)])
        if additivein is None: return apply_calpar(data, calpar, self.totalVisibilityId)
        else: return apply_calpar(data - additivein, calpar, self.totalVisibilityId)
    def get_modeled_data(self):
        '''XXX DOCSTRING'''
        if self.rawCalpar is None:
            raise ValueError("self.rawCalpar doesn't exist. Please calibrate first using logcal() or lincal().")
        if len(self.totalVisibilityId) <= np.max(self.Info.subsetbl):
            raise ValueError("self.totalVisibilityId of length %i is shorter than max index in subsetbl %i. Probably you are using an outdated version of redundantinfo."%(len(self.totalVisibilityId), np.max(self.Info.subsetbl)))
        mdata = np.zeros((self.rawCalpar.shape[0], self.rawCalpar.shape[1], len(self.totalVisibilityId)), dtype='complex64')
        mdata[..., self.Info.subsetbl[self.Info.crossindex]] = (self.rawCalpar[..., 3 + 2 * (self.Info.nAntenna)::2] + 1.j * self.rawCalpar[..., 4 + 2 * (self.Info.nAntenna)::2])[..., self.Info.bltoubl]
        #mdata[..., self.Info.subsetbl[self.Info.crossindex]] = np.abs(mdata[..., self.Info.subsetbl[self.Info.crossindex]]) * np.exp(self.Info.reversed * 1.j * np.angle(mdata[..., self.Info.subsetbl[self.Info.crossindex]])) * 10.**(self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,0]] + self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,1]]) * np.exp(-1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,0]] + 1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,1]])
        mdata[..., self.Info.subsetbl[self.Info.crossindex]] = np.abs(mdata[..., self.Info.subsetbl[self.Info.crossindex]]) * np.exp(1.j * np.angle(mdata[..., self.Info.subsetbl[self.Info.crossindex]])) * 10.**(self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,0]] + self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,1]]) * np.exp(-1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,0]] + 1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,1]])
        return mdata
    def get_omnichisq(self):
        '''XXX DOCSTRING'''
        if self.utctimes is None or self.rawCalpar is None:
            raise Exception("Error: either self.utctimes or self.rawCalpar does not exist.")
        if len(self.utctimes) != len(self.rawCalpar):
            raise Exception("Error: length of self.utctimes is not equal to self.rawCalpar. One of them is wrong.")
        jd = np.zeros((len(self.utctimes), 2), dtype='float32')#Julian dat is the only double in this whole thing so im storing it in two chunks as float
        sa = ephem.Observer()
        for utctime, t in zip(self.utctimes, range(len(self.utctimes))):
            sa.date = utctime
            jd[t, :] = struct.unpack('ff', struct.pack('d', sa.date + julDelta))
        omnichisq = np.zeros((self.nTime, 2 + 1 + self.nFrequency), dtype = 'float32')
        omnichisq[:, :2] = jd
        omnichisq[:, 2] = float(self.nFrequency)
        omnichisq[:, 3:] = self.rawCalpar[:, :, 2]#chisq which is sum of squares of errors in each visbility
        return omnichisq
    def get_omnigain(self):
        '''XXX DOCSTRING'''
        if self.utctimes is None or self.rawCalpar is None:
            raise Exception("Error: either self.utctimes or self.rawCalpar does not exist.")
        if len(self.utctimes) != len(self.rawCalpar):
            raise Exception("Error: length of self.utctimes is not equal to self.rawCalpar. One of them is wrong.")
        jd = np.zeros((len(self.utctimes), 2), dtype='float32')#Julian dat is the only double in this whole thing so im storing it in two chunks as float
        sa = ephem.Observer()
        for utctime, t in zip(self.utctimes, range(len(self.utctimes))):
            sa.date = utctime
            jd[t, :] = struct.unpack('ff', struct.pack('d', sa.date + julDelta))
        omnigain = np.zeros((self.nTime, self.Info.nAntenna, 2 + 1 + 1 + 2 * self.nFrequency), dtype = 'float32')
        omnigain[:, :, :2] = jd[:, None]
        omnigain[:, :, 2] = np.array(self.Info.subsetant).astype('float32')
        omnigain[:, :, 3] = float(self.nFrequency)
        gains = (10**self.rawCalpar[:, :, 3:(3 + self.Info.nAntenna)] * np.exp(1.j * self.rawCalpar[:, :, (3 + self.Info.nAntenna):(3 + 2 * self.Info.nAntenna)])).transpose((0,2,1))
        omnigain[:, :, 4::2] = np.real(gains)
        omnigain[:, :, 5::2] = np.imag(gains)
        return omnigain
    def get_omnifit(self):
        '''XXX DOCSTRING'''
        if self.utctimes is None or self.rawCalpar is None:
            raise Exception("Error: either self.utctimes or self.rawCalpar does not exist.")
        if len(self.utctimes) != len(self.rawCalpar):
            raise Exception("Error: length of self.utctimes is not equal to self.rawCalpar. One of them is wrong.")
        jd = np.zeros((len(self.utctimes), 2), dtype='float32')#Julian dat is the only double in this whole thing so im storing it in two chunks as float
        sa = ephem.Observer()
        for utctime, t in zip(self.utctimes, range(len(self.utctimes))):
            sa.date = utctime
            jd[t, :] = struct.unpack('ff', struct.pack('d', sa.date + julDelta))
        omnifit = np.zeros((self.nTime, len(self.Info.ublcount), 2 + 3 + 1 + 2 * self.nFrequency), dtype = 'float32')
        omnifit[:, :, :2] = jd[:, None]
        omnifit[:, :, 2:5] = np.array(self.Info.ubl).astype('float32')
        omnifit[:, :, 5] = float(self.nFrequency)
        omnifit[:, :, 6::2] = self.rawCalpar[:, :, 3 + 2 * self.Info.nAntenna::2].transpose((0,2,1))
        omnifit[:, :, 7::2] = self.rawCalpar[:, :, 3 + 2 * self.Info.nAntenna + 1::2].transpose((0,2,1))
        return omnifit
    def diagnose(self, data = None, additiveout = None, flag = None, verbose = True, healthbar = 2, ubl_healthbar = 50, warn_low_redun = False, ouput_txt = False):
        # XXX what does this function do?
        '''XXX DOCSTRING'''
        errstate = np.geterr()
        np.seterr(invalid = 'ignore')
        nUBL = len(self.Info.ublcount)
        if self.rawCalpar is None:
            raise Exception("No calibration has been performed since rawCalpar does not exist.")
        if flag is None: flag = np.zeros(self.rawCalpar.shape[:2], dtype='bool')
        elif flag.shape != self.rawCalpar.shape[:2]:
            raise TypeError('flag and self.rawCalpar have different shapes %s %s.'%(flag.shape, self.rawCalpar.shape[:2]))
        checks = 1
        bad_count = np.zeros((3,self.Info.nAntenna), dtype='int')
        bad_ubl_count = np.zeros(len(self.Info.ublcount), dtype='int')
        median_level = nanmedian(nanmedian(self.rawCalpar[:,:,3:3+self.Info.nAntenna], axis= 0), axis= 1)
        bad_count[0] = np.array([(np.abs(self.rawCalpar[:,:,3+a] - median_level) >= .15)[~flag].sum() for a in range(self.Info.nAntenna)])**2
        if data is not None and data.shape[:2] == self.rawCalpar.shape[:2]:
            checks += 1
            subsetbl = self.Info.subsetbl
            crossindex = self.Info.crossindex
            ncross = len(self.Info.crossindex)
            bl1dmatrix = self.Info.bl1dmatrix
            ant_level = np.array([np.median(np.abs(data[:, :, [subsetbl[crossindex[bl]] for bl in bl1dmatrix[a] if (bl < ncross and bl >= 0)]]), axis = -1) for a in range(self.Info.nAntenna)])
            median_level = nanmedian(ant_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_count[1] = np.array([(np.abs(ant_level[a] - median_level)/median_level >= .667)[~flag].sum() for a in range(self.Info.nAntenna)])**2
        if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2]:
            checks += 1
            subsetbl = self.Info.subsetbl
            crossindex = self.Info.crossindex
            ncross = len(self.Info.crossindex)
            bl1dmatrix = self.Info.bl1dmatrix
            ant_level = np.array([np.median(np.abs(additiveout[:, :, [crossindex[bl] for bl in bl1dmatrix[a] if bl < ncross]]), axis = 2) for a in range(self.Info.nAntenna)])
            median_level = np.median(ant_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_count[2] = np.array([(np.abs(ant_level[a] - median_level)/median_level >= .667)[~flag].sum() for a in range(self.Info.nAntenna)])**2
            ublindex = [np.array(index).astype('int')[:,2] for index in self.Info.ublindex]
            ubl_level = np.array([np.median(np.abs(additiveout[:, :, [crossindex[bl] for bl in ublindex[u]]]), axis = 2) for u in range(nUBL)])
            median_level = np.median(ubl_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_ubl_count += np.array([((ubl_level[u] - median_level)/median_level >= .667)[~flag].sum() for u in range(nUBL)])**2
        np.seterr(invalid = errstate['invalid'])
        bad_count = (np.mean(bad_count,axis=0)/float(np.sum(~flag))**2 * 100).astype('int')
        bad_ubl_count = (bad_ubl_count/float(self.nTime * self.nFrequency)**2 * 100).astype('int')
        if verbose:
            print "DETECTED BAD ANTENNA ABOVE HEALTH THRESHOLD %i: "%healthbar
            for a in range(len(bad_count)):
                if bad_count[a] > healthbar:
                    print "antenna #%i, vector = %s, badness = %i"%(self.Info.subsetant[a], self.Info.antloc[a], bad_count[a])
            if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2] and ubl_healthbar != 100:
                print "DETECTED BAD BASELINE TYPE ABOVE HEALTH THRESHOLD %i: "%ubl_healthbar
                for a in range(len(bad_ubl_count)):
                    if bad_ubl_count[a] > ubl_healthbar and (self.Info.ublcount[a] > 5 or (warn_low_redun)):
                        print "index #%i, vector = %s, redundancy = %i, badness = %i"%(a, self.Info.ubl[a], self.Info.ublcount[a], bad_ubl_count[a])
        if not ouput_txt: return bad_count, bad_ubl_count
        else:
            txt = ''
            txt += "DETECTED BAD ANTENNA ABOVE HEALTH THRESHOLD %i: \n"%healthbar
            for a in range(len(bad_count)):
                if bad_count[a] > healthbar:
                    txt += "antenna #%i, vector = %s, badness = %i\n"%(self.Info.subsetant[a], self.Info.antloc[a], bad_count[a])
            if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2] and ubl_healthbar != 100:
                txt += "DETECTED BAD BASELINE TYPE ABOVE HEALTH THRESHOLD %i: \n"%ubl_healthbar
                for a in range(len(bad_ubl_count)):
                    if bad_ubl_count[a] > ubl_healthbar and (self.Info.ublcount[a] > 5 or (warn_low_redun)):
                        txt += "index #%i, vector = %s, redundancy = %i, badness = %i\n"%(a, self.Info.ubl[a], self.Info.ublcount[a], bad_ubl_count[a])
            return txt
    def flag(self, mode='12', twindow=5, fwindow=5, nsigma=4, _dbg_plotter=None, _niter=3):
        '''return true if flagged False if good and unflagged'''
        # XXX what does this function do?
        if self.rawCalpar is None or (self.rawCalpar[:,:,2] == 0).all():
            raise Exception("flag cannot be run before lincal.")
        chisq = np.copy(self.rawCalpar[:,:,2])
        nan_flag = np.isnan(np.sum(self.rawCalpar,axis=-1))|np.isinf(np.sum(self.rawCalpar,axis=-1))
        #chisq flag: spike_flag
        spike_flag = np.zeros_like(nan_flag)
        if '1' in mode:
            median_level = nanmedian(nanmedian(chisq))
            thresh = nsigma * (2. / (len(self.subsetbl) - self.nAntenna - len(self.ublcount)+ 2))**.5 # relative sigma is sqrt(2/k)
            for i in range(_niter):
                chisq[nan_flag|spike_flag] = 1e6 * median_level
                if twindow >= self.nTime:
                    filtered_tdir = np.ones(self.nTime)#will rescale anyways * np.min(np.median(chisq, axis = 1))
                else:
                    filtered_tdir = sfil.minimum_filter(np.median(chisq, axis = 1), size = twindow, mode='reflect')
                if fwindow >= self.nFrequency:
                    filtered_fdir = np.ones(self.nFrequency)#will rescale anyways * np.min(np.median(chisq, axis = 0))
                else:
                    filtered_fdir = sfil.minimum_filter(np.median(chisq, axis = 0), size = fwindow, mode='reflect')
                smoothed_chisq = np.outer(filtered_tdir, filtered_fdir)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",category=RuntimeWarning)
                    smoothed_chisq = smoothed_chisq * np.median(chisq[~(nan_flag|spike_flag)] / smoothed_chisq[~(nan_flag|spike_flag)])
                    del_chisq = chisq - smoothed_chisq
                    del_chisq[(nan_flag|spike_flag)] = np.nan
                    estimate_chisq_sigma = np.nanstd(del_chisq,axis=0)
                    estimate_chisq_sigma[np.isnan(estimate_chisq_sigma)] = 0
                spike_flag = spike_flag | (np.abs(chisq - smoothed_chisq) >= np.minimum(smoothed_chisq * thresh, estimate_chisq_sigma * nsigma)) | (chisq == 0)
            if _dbg_plotter is not None:
                _dbg_plotter.imshow(np.abs(chisq - smoothed_chisq)/smoothed_chisq, vmin=-thresh, vmax=thresh, interpolation='none')
        #bl fit flag
        if '2' in mode:
            nubl = 10
            short_ubl_index = np.argsort(np.linalg.norm(self.ubl, axis=1))[:min(nubl, len(self.ublcount))]
            shortest_ubl_vis = self.rawCalpar[:,:,3+2*self.nAntenna+2*short_ubl_index] + 1.j * self.rawCalpar[:,:,3+2*self.nAntenna+2*short_ubl_index+1]
            change_rate = np.median(np.abs(shortest_ubl_vis[:-1] - shortest_ubl_vis[1:]), axis = 2)
            nan_mask2 = np.isnan(change_rate)|np.isinf(change_rate)
            change_rate[nan_mask2] = 0
            if twindow >= self.nTime:
                filtered_tdir = np.ones(self.nTime - 1)#will rescale anyways * np.min(np.median(chisq, axis = 1))
            else:
                filtered_tdir = sfil.minimum_filter(np.median(change_rate, axis = 1), size = twindow, mode='reflect')
            if fwindow >= self.nFrequency:
                filtered_fdir = np.ones(self.nFrequency)#will rescale anyways * np.min(np.median(chisq, axis = 0))
            else:
                filtered_fdir = sfil.minimum_filter(np.median(change_rate, axis = 0), size = fwindow, mode='reflect')
            smoothed_change_rate = np.outer(filtered_tdir, filtered_fdir)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                smoothed_change_rate = smoothed_change_rate * np.median(change_rate[~nan_mask2] / smoothed_change_rate[~nan_mask2])
            ubl_flag_short = (change_rate > 2 * smoothed_change_rate) | nan_mask2
            ubl_flag = np.zeros_like(spike_flag)
            ubl_flag[:-1] = ubl_flag_short
            ubl_flag[1:] = ubl_flag[1:] | ubl_flag_short
        else: ubl_flag = np.zeros_like(nan_flag)
        return_flag = (nan_flag|spike_flag|ubl_flag)
        return return_flag
    def compute_redundantinfo(self, arrayinfoPath=None, tol=1e-6):
        self.Info = self.arrayinfo.compute_redundantinfo(arrayinfoPath=arrayinfoPath, tol=tol)
    def get_ublindex(self,antpair): # XXX should this be part of info?
        '''need to do compute_redundantinfo first for this function to work (needs 'bl1dmatrix')
        input the antenna pair(as a list of two numbers), return the corresponding ubl index'''
        #check if the input is a list, tuple, np.array of two numbers
        if not (type(antpair) == list or type(antpair) == np.ndarray or type(antpair) == tuple):
            raise Exception("input needs to be a list of two numbers")
        elif len(np.array(antpair)) != 2:
            raise Exception("input needs to be a list of two numbers")
        elif type(antpair[0]) == str or type(antpair[0]) == np.string_:
            raise Exception("input needs to be number not string")
        try: _ = self.Info.bl1dmatrix
        except: raise Exception("needs Info.bl1dmatrix")
        crossblindex=self.Info.bl1dmatrix[antpair[0]][antpair[1]]
        if antpair[0]==antpair[1]: return "auto correlation"
        elif crossblindex == 99999: return "bad ubl"
        return self.Info.bltoubl[crossblindex]
    #def get_reversed(self,antpair): # XXX should this be part of info?
    #    '''need to do compute_redundantinfo first
    #    input the antenna pair, return -1 if it is a reversed bl and 1 if it is not reversed'''
    #    #check if the input is a list, tuple, np.array of two numbers
    #    if not (type(antpair) == list or type(antpair) == np.ndarray or type(antpair) == tuple):
    #        raise Exception("input needs to be a list of two numbers")
    #    elif len(np.array(antpair)) != 2:
    #        raise Exception("input needs to be a list of two numbers")
    #    elif type(antpair[0]) == str or type(antpair[0]) == np.string_:
    #        raise Exception("input needs to be number not string")
    #    #check if self.info['bl1dmatrix'] exists
    #    try: _ = self.Info.bl1dmatrix
    #    except: raise Exception("needs Info.bl1dmatrix")
    #    crossblindex=self.Info.bl1dmatrix[antpair[0]][antpair[1]]
    #    if antpair[0] == antpair[1]: return 1
    #    if crossblindex == 99999: return 'badbaseline'
    #    return self.Info.reversed[crossblindex]

##########################Sub-class#############################
# XXX application to PAPER should be in another file
class RedundantCalibrator_PAPER(RedundantCalibrator):
    '''XXX DOCSTRING'''
    def __init__(self, aa):
        nTotalAnt = len(aa)
        RedundantCalibrator.__init__(self, nTotalAnt)
        self.aa = aa
        self.antennaLocationAtom = np.zeros((self.nTotalAnt,3), dtype='int32')
        for i in range(len(self.aa.ant_layout)):
            for j in range(len(self.aa.ant_layout[0])):
                self.antennaLocationAtom[self.aa.ant_layout[i][j]] = np.array([i, j, 0])

        self.preciseAntennaLocation = .299792458 * np.array([ant.pos for ant in self.aa])
        self._goodAntenna = self.aa.ant_layout.flatten()
        self._goodAntenna.sort()
        # XXX need to clean this up, now that it's in ArrayInfo
        self.badAntenna = []
        self.badUBLpair = []
        for i in range(nTotalAnt):
            if i not in self._goodAntenna:
                self.badAntenna.append(i)
        self.antennaLocationAtom = self.antennaLocationAtom - np.mean(self.antennaLocationAtom[self._goodAntenna], axis = 0).astype('int32')

        ##self.antennaLocation = np.copy(self.antennaLocationAtom) #* [4, 30, 0]
        ####fit for idealized antloc
        A = np.array([list(a) + [1] for a in self.antennaLocationAtom[self._goodAntenna]])
        self.antennaLocation = np.zeros_like(self.antennaLocationAtom).astype('float64')
        self.antennaLocation[self._goodAntenna] = self.antennaLocationAtom[self._goodAntenna].dot(la.pinv(A.transpose().dot(A)).dot(A.transpose().dot(self.preciseAntennaLocation[self._goodAntenna]))[:3])##The overall constant is so large that it screws all the matrix inversion up. so im not including the over all 1e8 level shift
        self.antennaLocation[self._goodAntenna, ::2] = self.antennaLocation[self._goodAntenna, ::2].dot(np.array([[np.cos(PI/2+aa.lat), np.sin(PI/2+aa.lat)],[-np.sin(PI/2+aa.lat), np.cos(PI/2+aa.lat)]]).transpose())###rotate into local coordinates

class RedundantCalibrator_X5(RedundantCalibrator):
    def __init__(self, antennaLocation):
        nant = len(antennaLocation)
        RedundantCalibrator.__init__(self, nant)
        self.antennaLocation = antennaLocation
        self.totalVisibilityId = np.concatenate([[[i,j] for j in range(i, nant)] for i in range(nant)])
        self._gen_totalVisibilityId_dic()

        self.badAntenna = range(16) + range(56,60) + [16,19,50]

# XXX omnical should not be in charge of storing these quantities to disk.  this should be up to the user
def load_omnichisq(path):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path): raise IOError("Path %s does not exist."%path)
    omnichisq = np.fromfile(path, dtype = 'float32')
    NF = int(omnichisq[2])
    omnichisq.shape = (len(omnichisq) / (NF + 3), (NF + 3))
    return omnichisq

def load_omnigain(path, info=None):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path): raise IOError("Path %s does not exist."%path)
    if info is None: info = path.replace('.omnigain', '.binfo')
    if type(info) == type('a'): info = read_redundantinfo(info)
    omnigain = np.fromfile(path, dtype = 'float32')
    omnigain.shape = (omnigain.shape[0] / (info['nAntenna']) / (2 + 1 + 1 + 2 * int(omnigain[3])), info['nAntenna'], 2 + 1 + 1 + 2 * int(omnigain[3]))
    return omnigain

def load_omnifit(path, info=None):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path): raise IOError("Path %s does not exist."%path)
    if info is None: info = path.replace('.omnifit', '.binfo')
    if type(info) == type('a'): info = read_redundantinfo(info)
    nUBL = len(info.ublcount)
    omnifit = np.fromfile(path, dtype = 'float32')
    omnifit.shape = (omnifit.shape[0] / nUBL / (2 + 3 + 1 + 2 * int(omnifit[5])), nUBL, 2 + 3 + 1 + 2 * int(omnifit[3]))
    return omnifit

def get_omnitime(omnistuff):
    '''XXX DOCSTRING'''
    if len(omnistuff.shape) == 2:
        return np.array([struct.unpack('d', struct.pack('ff', *(pair.tolist())))[0] for pair in omnistuff[:, :2]])
    elif len(omnistuff.shape) == 3:
        return np.array([struct.unpack('d', struct.pack('ff', *(pair.tolist())))[0] for pair in omnistuff[:, 0, :2]])
    else:
        raise ValueError('get_omnitime does not know how to deal with array of shape %s.'%omnistuff.shape)

# XXX utility function belongs in another file
def omniview(data_in, info, plotrange = None, oppath = None, suppress = False, title = '', plot_single_ubl = False, plot_3 = False, plot_1 = -1):
    '''plot_3: only plot the 3 most redundant ones. plot_1: counting start from 0 the most redundant bl'''
    import matplotlib.pyplot as plt
    data = np.array(data_in)
    try:#in case info is Info class
        info = info.get_info()
    except:
        pass
    nUBL = len(info.ublcount)
    if plot_3 and nUBL < 3:
        plot_3 = False

    colors=[]
    colorgrid = int(math.ceil((nUBL/12.+1)**.34))
    for red in range(colorgrid):
        for green in range(colorgrid):
            for blue in range(colorgrid):
                #print red, green, blue
                colors += [(np.array([red, green, blue]).astype('float')/(colorgrid - 1)).tolist()]
    #colors.remove([0,0,0])
    colors.remove([1,1,1])

    if plot_3:
        select_ubl_index = np.argsort(info['ublcount'])[-3:]
    elif plot_1 >= 0:
        select_ubl_index = np.argsort(info['ublcount'])[::-1][plot_1:plot_1+1]


    if len(data.shape) == 1 or data.shape[0] == 1:
        ds = [data[info['subsetbl']][info['crossindex']]]
        fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True)
        axes = [axes]
    else:
        ds = data[:, info['subsetbl'][info['crossindex']]]
        fig, axes = plt.subplots(nrows=1, ncols=len(ds), sharey=True, sharex=True)

    outputdata = []
    for i in range(len(ds)):
        outputdata = outputdata + [[]]
        d = ds[i]
        ax = axes[i]
        if plotrange is None:
            plotrange = 1.2*np.nanmax(np.abs(d))

        ubl = 0
        for marker in ["o", "v", "^", "<", ">", "8", "s", "p", "h", (6,1,0), (8,1,0), "d"]:
            for color in colors:
                #print info['ublindex'][ubl][:,2]
                #print marker, color
                if (plot_single_ubl or len(info['ublindex'][ubl]) > 1) and (not (plot_3 or plot_1 >= 0) or ubl in select_ubl_index):
                    if plot_3 or plot_1 >= 0:
                        color = [[1,0,0],[0,1,0],[0,0,1]][select_ubl_index.tolist().index(ubl)]
                    #ax.scatter(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]),np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')])*info['reversed'][np.array(info['ublindex'][ubl][:,2]).astype('int')], marker=marker, color=color)
                    ax.scatter(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]),np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]), marker=marker, color=color)
                    #outputdata[i] = outputdata[i] + [(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]) + 1.j * np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')])*info['reversed'][np.array(info['ublindex'][ubl][:,2]).astype('int')], marker, color, info['ubl'][ubl])]
                    outputdata[i] = outputdata[i] + [(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]) + 1.j * np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]), marker, color, info['ubl'][ubl])]

                ubl += 1
                if ubl == nUBL:
                    #if i == 1:
                        #ax.text(-(len(ds)-1 + 0.7)*plotrange, -0.7*plotrange, "#Ant:%i\n#UBL:%i"%(info['nAntenna'],info['nUBL']),bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
                    ax.set_title(title + "\nGood Antenna count: %i\nUBL count: %i"%(info['nAntenna'],nUBL))
                    ax.grid(True)
                    ax.set(adjustable='datalim', aspect=1)
                    ax.set_xlabel('Real')
                    ax.set_ylabel('Imag')
                    break
            if ubl == nUBL:
                break
    plt.axis([-plotrange, plotrange, -plotrange, plotrange])
    if oppath is not None:
        plt.savefig(oppath, bbox_inches='tight')
    if not suppress:
        plt.show()
    else:
        plt.close()
    return outputdata

# XXX utility function belongs in another file
def lin_depend(v1, v2, tol = 0):
    '''whether v1 and v2 are linearly dependent'''
    if len(v1) != len(v2): raise Exception("Length mismatch %i vs %i."%(len(v1), len(v2)))
    if la.norm(v1) == 0: return True
    return la.norm(np.dot(v1, v2)/np.dot(v1, v1) * v1 - v2) <= tol

# XXX utility function belongs in another file
def _f(rawcal_ubl=[], verbose=False):
    '''run this function twice in a row and its christmas'''
    if verbose and rawcal_ubl != []: print "Starting ubl:", rawcal_ubl
    if rawcal_ubl == []: rawcal_ubl += [2,3]
    if verbose: print "ubl:", rawcal_ubl

def find_solution_path(info, input_rawcal_ubl=[], tol = 0.0, verbose=False):
    '''return (intialantenna, solution_path) for raw calibration. solution path
    contains a list of [(a0, a1, crossubl), a] = [(ublindex entry), (which ant is
    solved, 0 or 1)]. When raw calibrating, initialize calpar to have [0] at
    initial antenna, then simply iterate along the solution_path, use crossubl and
    a0 or a1 specified by a to solve for the other a1 or a0 and append it to
    calpar. Afterwards, use mean angle on calpars'''
    ###select 2 ubl for calibration
    rawcal_ubl = list(input_rawcal_ubl)
    if verbose and rawcal_ubl != []: print "Starting ubl:", rawcal_ubl
    if rawcal_ubl == []:
        ublcnt_tmp = info['ublcount'].astype('float')
        rawcal_ubl += [np.argmax(ublcnt_tmp)]
        if verbose:
            print "Picking %s with redundancy %i as first ubl"%(info['ubl'][rawcal_ubl[-1]], ublcnt_tmp[rawcal_ubl[-1]])
        ublcnt_tmp[rawcal_ubl[-1]] = np.nan
        rawcal_ubl += [np.nanargmax(ublcnt_tmp)]
        if verbose:
            print "Picking %s with redundancy %i as second ubl"%(info['ubl'][rawcal_ubl[-1]], ublcnt_tmp[rawcal_ubl[-1]])
        ublcnt_tmp[rawcal_ubl[-1]] = np.nan
        #while np.allclose(info['ubl'][rawcal_ubl[0]]/(la.norm(info['ubl'][rawcal_ubl[0]])/la.norm(info['ubl'][rawcal_ubl[1]])), info['ubl'][rawcal_ubl[1]]) or np.allclose(info['ubl'][rawcal_ubl[0]]/(la.norm(info['ubl'][rawcal_ubl[0]])/la.norm(info['ubl'][rawcal_ubl[1]])), -info['ubl'][rawcal_ubl[1]]):
        while lin_depend(info['ubl'][rawcal_ubl[0]], info['ubl'][rawcal_ubl[1]], tol=tol):
            if verbose:
                print info['ubl'][rawcal_ubl[0]], "and", info['ubl'][rawcal_ubl[1]], "are linearly dependent."
            try:
                rawcal_ubl[1] = np.nanargmax(ublcnt_tmp)
                if verbose:
                    print "Picking %s with redundancy %i as second ubl"%(info['ubl'][rawcal_ubl[-1]], ublcnt_tmp[rawcal_ubl[-1]])
            # XXX this not good
            except: raise Exception("Cannot find two unique bls that are linearly independent!")
            ublcnt_tmp[rawcal_ubl[-1]] = np.nan
    if verbose: print "ubl:", info['ubl'][rawcal_ubl[0]], info['ubl'][rawcal_ubl[1]]
    if info['ublcount'][rawcal_ubl[0]] + info['ublcount'][rawcal_ubl[1]] <= info['nAntenna'] + 2:
        raise Exception('Array not redundant enough! Two most redundant bls %s and %s have %i and %i bls, which is not larger than 2 + %i'%(info['ubl'][rawcal_ubl[0]],info['ubl'][rawcal_ubl[1]], info['ublcount'][rawcal_ubl[0]],info['ublcount'][rawcal_ubl[1]], info['nAntenna']))

    ublindex = np.concatenate((np.array(info['ublindex'][rawcal_ubl[0]]).astype('int'), np.array(info['ublindex'][rawcal_ubl[1]]).astype('int')))#merge ublindex since we set both ubl phase to 0

    ###The overarching goal is to find a solution path (a sequence of unique bls to solve) that can get multiple solutions to multiple antennas using just two sets of ubls
    solution_path = []

    antcnt = np.bincount(np.array(ublindex)[:,:2].flatten(), minlength = info['nAntenna'])#how many times each antenna appear in the two sets of ubls. at most 4. not useful if < 2
    unsolved_ant = []
    for a in range(len(antcnt)):
        if antcnt[a] == 0:
            unsolved_ant.append(a)
    if verbose: print "antcnt", antcnt, "Antennas", np.array(info['subsetant'])[unsolved_ant], "not directly solvable."


    ###Status string for ubl: NoUse: none of the two ants have been solved; Solvable: at least one of the ants have solutions; Done: used to generate one antennacalpar
    ublstatus = ["NoUse" for i in ublindex]

    ###antenna calpars, a list for each antenna
    calpar = np.array([[]] * info['nAntenna']).tolist()
    ###select initial antenna
    initialant = int(np.argmax(antcnt))
    if verbose: print "initialant", np.array(info['subsetant'])[initialant]
    calpar[initialant].append(0)
    for i in range(len(ublstatus)):
        if initialant in ublindex[i, 0:2]:
            ublstatus[i] = "Solvable"

    ###start looping
    solvecnt = 10#number of solved bls in each loop, 10 is an arbitrary starting point
    if verbose: print "new ant solved",
    while solvecnt > 0:
        solvecnt = 0
        for i in range(len(ublstatus)):
            if ublstatus[i] == "Solvable":
                solvecnt += 1
                if calpar[ublindex[i, 0]] != []:#if the first antenna is solved
                    #print ublindex[i], ublindex[i, 1], len(calpar[ublindex[i, 1]]), calpar[ublindex[i, 1]],
                    calpar[ublindex[i, 1]].append(0)#just append a dummy
                    ublstatus[i] = "Done"
                    solution_path.append([ublindex[i], 0])
                    #print len(calpar[ublindex[i, 1]]), calpar[ublindex[i, 1]]
                    if len(calpar[ublindex[i, 1]]) == 1:
                        if verbose: print np.array(info['subsetant'])[ublindex[i, 1]],
                        for j in range(len(ublstatus)):
                            if (ublindex[i, 1] in ublindex[j, 0:2]) and ublstatus[j] == "NoUse":
                                ublstatus[j] = "Solvable"
                else:
                    #print ublindex[i], ublindex[i, 0], len(calpar[ublindex[i, 0]]), calpar[ublindex[i, 0]],
                    calpar[ublindex[i, 0]].append(0)#just append a dummy
                    ublstatus[i] = "Done"
                    solution_path.append([ublindex[i], 1])
                    #print len(calpar[ublindex[i, 0]]), calpar[ublindex[i, 0]]
                    if len(calpar[ublindex[i, 0]]) == 1:
                        if verbose: print np.array(info['subsetant'])[ublindex[i, 0]],
                        for j in range(len(ublstatus)):
                            if (ublindex[i, 0] in ublindex[j, 0:2]) and ublstatus[j] == "NoUse":
                                ublstatus[j] = "Solvable"
    if verbose:
        print ""
        if len(solution_path) != len(ublindex):
            print "Solution path has %i entries where as total candidates in ublindex have %i. The following bls form their isolated isaland:"%(len(solution_path), len(ublindex))
            unsolved_ubl = []
            for i in range(len(ublstatus)):
                if ublstatus[i] != "Done":
                    print np.array(info['subsetant'])[ublindex[i][0]], np.array(info['subsetant'])[ublindex[i][1]]
                    unsolved_ubl.append(ublindex[i])
            unsolved_ubl = np.array(unsolved_ubl)[:,:2].flatten()
            for a in range(info['nAntenna']):
                if a in unsolved_ubl:
                    unsolved_ant.append(a)
    ant_solved = 10
    additional_solution_path = []
    while len(unsolved_ant) > 0 and ant_solved > 0:#find a ubl that can solve these individual antennas not involved in the chosen 2 ubls. Use while because the first ant in the unsolved_ant may not be solvable on the first pass

        ant_solved = 0
        for a in unsolved_ant:
            if verbose: print "trying to solve for ", np.array(info['subsetant'])[a],
            ublcnt_tmp = info['ublcount'].astype('float')
            third_ubl_good = False
            tried_all_ubl = False
            while (not third_ubl_good) and (not tried_all_ubl):
                try:
                    third_ubl = np.nanargmax(ublcnt_tmp)
                    ublcnt_tmp[third_ubl] = np.nan
                except:
                    tried_all_ubl = True
                    break
                if verbose: print "trying ubl ", third_ubl,
                third_ubl_good = False #assume false and start checking if this ubl 1) has this antenna 2) has another bls whose two ants are both solved
                if (len(info['ublindex'][third_ubl]) < 2) or (a not in info['ublindex'][third_ubl]):
                    continue
                third_ubl_good1 = False
                third_ubl_good2 = False
                for a1, a2, bl in info['ublindex'][third_ubl]:
                    if (a1 not in unsolved_ant) and (a2 not in unsolved_ant):
                        third_ubl_good1 = True
                        if third_ubl_good2:
                            break
                    if ((a == a1) and (a2 not in unsolved_ant)) or ((a == a2) and (a1 not in unsolved_ant)):
                        third_ubl_good2 = True
                        if third_ubl_good1:
                            break
                third_ubl_good = (third_ubl_good1 and third_ubl_good2)
            if third_ubl_good:#figure out how to use this third ubl to solve this a
                if verbose:
                    print "picked ubl", info['ubl'][third_ubl], "to solve for ant", np.array(info['subsetant'])[a]
                get_ubl_fit = []#a recipe for how to get the ublfit and solvefor the unsolved antenna
                for a1, a2, bl in info['ublindex'][third_ubl].astype('int'):
                    if (a1 not in unsolved_ant) and (a2 not in unsolved_ant):
                        #get_ubl_fit.append([a1, a2, bl, info['reversed'][bl]])
                        get_ubl_fit.append([a1, a2, bl, 1])
                for a1, a2, bl in info['ublindex'][third_ubl].astype('int'):
                    if (a1 not in unsolved_ant) and (a2 == a):
                        #get_ubl_fit.append([a1, a2, bl, info['reversed'][bl], 0])
                        get_ubl_fit.append([a1, a2, bl, 1, 0])
                        break
                    if (a2 not in unsolved_ant) and (a1 == a):
                        #get_ubl_fit.append([a1, a2, bl, info['reversed'][bl], 1])
                        get_ubl_fit.append([a1, a2, bl, 1, 1])
                        break
                additional_solution_path.append(get_ubl_fit)
                ant_solved += 1
                unsolved_ant.remove(a)

    #remove the effect of enforcing the two bls to be 0, rather, set the first two linearly independent antennas w.r.t initant to be 0
    # find two antennas:
    a1 = initialant
    a2 = initialant
    for index in info['ublindex'][rawcal_ubl[0]]:
        if index[0] == initialant:
            a1 = int(index[1])
            break
        elif index[1] == initialant:
            a1 = int(index[0])
            break
    bl1 = np.array(info['antloc'][a1]) - info['antloc'][initialant]
    for index in info['ublindex'][rawcal_ubl[1]]:
        if index[0] == initialant:
            a2 = int(index[1])
            break
        elif index[1] == initialant:
            a2 = int(index[0])
            break
    bl2 = np.array(info['antloc'][a2]) - info['antloc'][initialant]
    #A = np.array([bl1[:2], bl2[:2]])
    #remove_Matrix = (np.array(info['antloc'])- info['antloc'][initialant])[:,:2].dot(la.pinv(A.transpose().dot(A)).dot(A.transpose()))
    A = np.array([bl1, bl2])
    remove_Matrix = (np.array(info['antloc'])- info['antloc'][initialant]).dot(la.pinv(A.transpose().dot(A)).dot(A.transpose()))
    degeneracy_remove = [a1, a2, remove_Matrix]
    if verbose: print "Degeneracy: a1 = %i, a2 = %i"%(info['subsetant'][a1], info['subsetant'][a2])
    return initialant, solution_path, additional_solution_path, degeneracy_remove, (unsolved_ant == [])

def meanAngle(a, weights = None, axis = -1):
    return np.angle(np.average(np.exp(1.j*np.array(a)), weights = weights, axis = axis))

def medianAngle(a, axis = -1):
    return np.angle(nanmedian(np.cos(a), axis = axis) + 1.j * nanmedian(np.sin(a), axis = axis))

#def _medianAngle(data, result, axis = -1):
    #result_shape = collapse_shape(data.shape, axis)

    #np_result = np.frombuffer(result, dtype='float32')
    #np_result.shape = tuple(result_shape)
    #np_result[:] = medianAngle(data, axis = axis).reshape(result_shape)
    #return

# XXX utility function belongs in another file XXX appears unused
#def collapse_shape(shape, axis):
#    '''XXX DOCSTRING'''
#    if axis == 0 or axis == -len(shape):
#        return tuple(list(shape)[1:])
#    elif axis == -1 or axis == len(shape) - 1:
#        return tuple(list(shape)[:-1])
#    else:
#        return tuple(list(shape)[:axis] + list(shape)[axis+1:])

###curerntly suffering from slow initialization which is probably due to copying data into shared array. worth further investigation.
#def medianAngle_multithread(data, axis = -1, nthread = None, verbose = False):
    #if axis < 0:
        #axis = data.ndim + axis
    #parallel_axis2 = np.argmax(collapse_shape(data.shape, axis))#the axis after averaging
    #if parallel_axis2 >= axis:
        #parallel_axis1 = parallel_axis2 + 1
    #else:
        #parallel_axis1 = parallel_axis2
    #parallel_axis_len = data.shape[parallel_axis1]
    #if nthread is None:
        #nthread = min(mp.cpu_count() - 1, parallel_axis_len)
    #nthread = min(nthread, parallel_axis_len)
    #if nthread < 2 or data.ndim == 1:
        #return medianAngle(data, axis=axis)
    #else:
        #results = {}
        #np_results = {}

        #threads = {}
        #fchunk = {}
        #chunk = parallel_axis_len / int(nthread)
        #excess = parallel_axis_len % int(nthread)
        #kwarg = {"axis": axis}

####set up threads
        #for i in range(nthread):
            #if excess == 0:
                #fchunk[i] = (i * chunk, min((1 + i) * chunk, parallel_axis_len),)
            #elif i < excess:
                #fchunk[i] = (i * (chunk+1), min((1 + i) * (chunk+1), parallel_axis_len),)
            #else:
                #fchunk[i] = (fchunk[i-1][1], min(fchunk[i-1][1] + chunk, parallel_axis_len),)

            #result_shape = list(collapse_shape(data.shape, axis))
            #result_shape[parallel_axis2] = fchunk[i][1] - fchunk[i][0]

            #results[i] = mp.RawArray('f', np.prod(result_shape))
            #np_results[i] = np.frombuffer(results[i], dtype='float32')
            #np_results[i].shape = tuple(result_shape)
            #def _slice(a):
                #return a[fchunk[i][0]:fchunk[i][1]]
            #threads[i] = mp.Process(target = _medianAngle, args = (np.apply_along_axis(_slice, parallel_axis1, data), results[i]), kwargs=kwarg)

####start processing
        #if verbose:
            #print "Starting medianAngle Process",
            #sys.stdout.flush()
        #for i in range(nthread):
            #if verbose:
                #print "#%i"%i,
                #sys.stdout.flush()
            #threads[i].start()
        #if verbose:
            #print "Finished Process",
        #for i in range(nthread):
            #threads[i].join()
            #if verbose:
                #print "#%i"%i,
        #if verbose:
            #print ""
            #sys.stdout.flush()
        #return np.concatenate([np_results[i] for i in range(nthread)],axis=parallel_axis2)

def raw_calibrate(data, info, initant, solution_path, additional_solution_path, degeneracy_remove):
    '''XXX DOCSTRING'''
    result = np.ones(int(math.floor((len(data)*2.)**.5)), dtype='complex64')
    calpar = np.array([[]]*info['nAntenna']).tolist()
    calpar[initant] = [0]
    d=np.angle(data[info['subsetbl']][info['crossindex']])
    #d=np.angle(omni.apply_calpar(data, result, visibilityID)[info['subsetbl']][info['crossindex']])
    for ublindex, a in solution_path:
        calpar[ublindex[1-a]].append(calpar[ublindex[a]][0] + ((a-0.5)/-.5)*d[ublindex[2]])
    for i in range(len(calpar)):
        if len(calpar[i]) > 0:
            calpar[i] = [meanAngle(calpar[i])]
        else:
            calpar[i] = [0]

    #now deal with additional_solution_path which deal with antennas that are not included in the 2 ubls picked to be 0
    for solution in additional_solution_path:
        ubl_phases = np.array([s[-1]*(calpar[s[0]][0]-calpar[s[1]][0]+d[s[2]]) for s in solution[:-1]])
        ubl_phase = medianAngle(ubl_phases)
        #print np.angle(np.exp(1.j*ubl_phases))
        #print calpar[solution[-1][1-solution[-1][-1]]]
        calpar[solution[-1][1-solution[-1][-1]]] = [calpar[solution[-1][solution[-1][-1]]][0] + ((solution[-1][-1]-0.5)/-.5)*(d[solution[-1][2]] - ubl_phase * (solution[-1][-2]))]
        #print solution[-1]
        #print calpar[solution[-1][solution[-1][-1]]][0], ((solution[-1][-1]-0.5)/-.5),d[solution[-1][2]] , ubl_phase * (solution[-1][-2]), calpar[solution[-1][1-solution[-1][-1]]]

    calpar = (np.array(calpar).flatten() + np.pi) % (2 * np.pi) - np.pi
    #remove the effect of enforcing the two bls to be 0, rather, set the first two linearly independent antennas w.r.t initant to be 0

    calpar = calpar - degeneracy_remove[2].dot([calpar[degeneracy_remove[0]],calpar[degeneracy_remove[1]]])

    result[info['subsetant']] = np.exp(1.j*calpar)# * result[info['subsetant']]
    return result

# XXX utility class belongs in another file XXX appears unused
#class InverseCholeskyMatrix:
#    '''for a positive definite matrix, Cholesky decomposition is M = L.Lt, where L
#    lower triangular. This decomposition helps computing inv(M).v faster, by
#    avoiding calculating inv(M). Once we have L, the product is simply
#    inv(Lt).inv(L).v, and inverse of triangular matrices multiplying a vector is
#    fast. sla.solve_triangular(M, v) = inv(M).v'''
#    def __init__(self, matrix):
#        if type(matrix).__module__ != np.__name__ or len(matrix.shape) != 2:
#            raise TypeError("matrix must be a 2D numpy array");
#        try:
#            self.L = la.cholesky(matrix)#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
#            self.Lt = self.L.conjugate().transpose()
#            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
#        except:
#            raise TypeError("cholesky failed. matrix is not positive definite.")
#
#    @classmethod
#    def fromfile(cls, filename, n, dtype):
#        if not os.path.isfile(filename):
#            raise IOError("%s file not found!"%filename)
#        matrix = cls(np.array([[1,0],[0,1]]))
#        try:
#            matrix.L = np.fromfile(filename, dtype=dtype).reshape((n,n))#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
#            matrix.Lt = matrix.L.conjugate().transpose()
#            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
#        except:
#            raise TypeError("cholesky import failed. matrix is not %i by %i with dtype=%s."%(n, n, dtype))
#        return matrix
#
#    def dotv(self, vector):
#        try:
#            return la.solve_triangular(self.Lt, la.solve_triangular(self.L, vector, lower=True), lower=False)
#        except:
#            return np.empty_like(vector)+np.nan
#
#    def dotM(self, matrix):
#        return np.array([self.dotv(v) for v in matrix.transpose()]).transpose()
#
#    def astype(self, t):
#        self.L = self.L.astype(t)
#        self.Lt = self.Lt.astype(t)
#        return self
#
#    def tofile(self, filename, overwrite = False):
#        if os.path.isfile(filename) and not overwrite:
#            raise IOError("%s file exists!"%filename)
#        self.L.tofile(filename)

# XXX utility function belongs elsewhere
def solve_slope(A_in, b_in, tol, niter=30, step=1, verbose=False):
    '''solve for the solution vector x such that mod(A.x, 2pi) = b,
    where the values range from -p to p. solution will be seeked
    on the first axis of b'''
    p = np.pi
    A = np.array(A_in)
    b = np.array(b_in + p) % (2*p) - p
    if A.ndim != 2: raise TypeError("A matrix must be 2 dimensional. Input A is %i dimensional."%A.ndim)
    if A.shape[0] != b.shape[0]: raise TypeError("A and b has shape mismatch: %s and %s."%(A.shape, b.shape))
    if A.shape[1] != 2: raise TypeError("A matrix's second dimension must have size of 2. %i inputted."%A.shape[1])
    #find the shortest 2 non-parallel bls, candidate_vecs have all combinations of vectors in a summation or subtraction. Each entry is i,j, v0,v1 where Ai+Aj=(v0,v1), negative j means subtraction. Identical i,j means vector itself without add or subtract
    candidate_vecs = np.zeros((len(A)**2, 4), dtype = 'float32')
    n = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i < j: candidate_vecs[n] = [i, j, A[i,0]+A[j,0], A[i,1]+A[j,1]]
            elif i == j: candidate_vecs[n] = [i, j, A[i,0], A[i,1]]
            elif i > j: candidate_vecs[n] = [i, -j, A[i,0]-A[j,0], A[i,1]-A[j,1]]
            n = n + 1
    candidate_vecs = candidate_vecs[np.linalg.norm(candidate_vecs, axis=1)>tol]
    #construct coarse A that consists of the 2 shortest vecs
    coarseA = np.zeros((2,2), dtype = 'float32')
    if b.ndim > 1: coarseb = np.zeros(np.concatenate(([2], b.shape[1:])), dtype='float32')
    else: coarseb = np.zeros(2, dtype='float32')

    for n in np.argsort(np.linalg.norm(candidate_vecs[:, 2:4], axis=1)):
        v = candidate_vecs[n, 2:4]
        if la.norm(coarseA[0]) == 0: coarseA[0] = v
        else:
            perp_component = v - v.dot(coarseA[0])/(coarseA[0].dot(coarseA[0])) * coarseA[0]
            if la.norm(perp_component) > tol:
                coarseA[1] = v
                break
    if la.norm(coarseA[1]) == 0:
        raise Exception("Poorly constructed A matrix: cannot find a pair of orthogonal vectors")
    #construct coarse b that contains medianAngle off all bs correponding to the 2 shortest bls
    coarseb0_candidate_indices = np.arange(len(candidate_vecs))[(np.linalg.norm(candidate_vecs[:, 2:4] - coarseA[0], axis=-1) < tol)|(np.linalg.norm(candidate_vecs[:, 2:4] + coarseA[0], axis=-1) < tol)]#stores the indices in candidate_vecs that is revelant to coarseb0
    coarseb1_candidate_indices = np.arange(len(candidate_vecs))[(np.linalg.norm(candidate_vecs[:, 2:4] - coarseA[1], axis=-1) < tol)|(np.linalg.norm(candidate_vecs[:, 2:4] + coarseA[1], axis=-1) < tol)]#stores the indices in candidate_vecs that is revelant to coarseb1
    coarseb0_candidate_shape = np.array(coarseb.shape)
    coarseb0_candidate_shape[0] = len(coarseb0_candidate_indices)
    coarseb1_candidate_shape = np.array(coarseb.shape)
    coarseb1_candidate_shape[0] = len(coarseb1_candidate_indices)
    coarseb0_candidate = np.zeros(coarseb0_candidate_shape, dtype='float32')
    coarseb1_candidate = np.zeros(coarseb1_candidate_shape, dtype='float32')

    for nn, (coarseb_candidate_indices, coarseb_candidate) in enumerate(zip([coarseb0_candidate_indices, coarseb1_candidate_indices], [coarseb0_candidate, coarseb1_candidate])):
        for n, ind in enumerate(coarseb_candidate_indices):
            i = int(candidate_vecs[ind, 0])
            j = int(candidate_vecs[ind, 1])
            v = candidate_vecs[ind, 2:4]
            if la.norm(coarseA[nn] - v) < tol: bsign = 1
            else: bsign = -1
            if i < j: coarseb_candidate[n] = b[i]+b[j]#(b[i]+b[j]+p)%(2*p)-p
            elif i == j: coarseb_candidate[n] = b[i]
            elif i > j: coarseb_candidate[n] = b[i]-b[abs(j)]#(b[i]-b[abs(j)]+p)%(2*p)-p
            coarseb_candidate[n] = coarseb_candidate[n] * bsign

    coarseb[0] = medianAngle(coarseb0_candidate, axis=0)
    coarseb[1] = medianAngle(coarseb1_candidate, axis=0)
    if verbose:
        print coarseb0_candidate.shape
    # find coarse solutions
    try: icA = la.inv(coarseA)
    # XXX this bad
    except: raise Exception("Poorly constructed coarseA matrix: %s."%(coarseA))
    try:
        #iA = InverseCholeskyMatrix(A.transpose().dot(A))
        iA = la.inv(A.transpose().dot(A))
    # XXX this bad
    except: raise Exception("Poorly constructed A matrix: %s."%(A.transpose().dot(A)))
    if verbose: print iA.shape
    if b.ndim > 2:
        extra_shape = b.shape[1:]
        flat_extra_dim = 1
        for i in range(1, b.ndim): flat_extra_dim = flat_extra_dim * b.shape[i]
        coarseb.shape = (2, flat_extra_dim)
        b.shape = (len(b), flat_extra_dim)
    else: extra_shape = None
    result = icA.dot(coarseb)
    if verbose:
        print coarseA
        print result
    for i in range(niter):
        result = result + step * iA.dot(A.transpose().dot((b - A.dot(result) + p)%(2*p)-p))
        if verbose: print result
    if extra_shape is not None: result.shape = tuple(np.concatenate(([2], extra_shape)))
    return result

#def solve_slope_old(A_in, b_in, tol, niter=3, p = np.pi):#solve for the solution vector x such that mod(A.x, 2pi) = b, where the values range from -p to p. solution will be seeked on the first axis of b
    #A = np.array(A_in)
    #b = np.array(b_in + p) % (2*p) - p
    #if A.ndim != 2:
        #raise TypeError("A matrix must be 2 dimensional. Input A is %i dimensional."%A.ndim)
    #if A.shape[0] != b.shape[0]:
        #raise TypeError("A and b has shape mismatch: %s and %s."%(A.shape, b.shape))
    #if A.shape[1] != 2:
        #raise TypeError("A matrix's second dimension must have size of 2. %i inputted."%A.shape[1])

    ##find the shortest 2 non-parallel bls, candidate_vecs have all combinations of vectors in a summation or subtraction. Each entry is i,j, v0,v1 where Ai+Aj=(v0,v1), negative j means subtraction. Identical i,j means vector itself without add or subtract
    #candidate_vecs = np.zeros((len(A)**2, 4), dtype = 'float32')
    #n = 0
    #for i in range(len(A)):
        #for j in range(len(A)):
            #if i < j:
                #candidate_vecs[n] = [i, j, A[i,0]+A[j,0], A[i,1]+A[j,1]]
            #elif i == j:
                #candidate_vecs[n] = [i, j, A[i,0], A[i,1]]
            #elif i > j:
                #candidate_vecs[n] = [i, -j, A[i,0]-A[j,0], A[i,1]-A[j,1]]

            #n = n + 1

    #candidate_vecs = candidate_vecs[np.linalg.norm(candidate_vecs, axis=1)>tol]

    ##construct coarse A that consists of the 2 shortest vecs
    #coarseA = np.zeros((2,2), dtype = 'float32')
    #if b.ndim > 1:
        #coarseb = np.zeros(np.concatenate(([2], b.shape[1:])), dtype='float32')
    #else:
        #coarseb = np.zeros(2, dtype='float32')
    #for n in np.argsort(np.linalg.norm(candidate_vecs[:, 2:4], axis=1)):
        #i = candidate_vecs[n, 0]
        #j = candidate_vecs[n, 1]
        #v = candidate_vecs[n, 2:4]
        #if la.norm(coarseA[0]) == 0:
            #coarseA[0] = v
            #if i < j:
                #coarseb[0] = (b[i]+b[j]+p)%(2*p)-p
            #elif i == j:
                #coarseb[0] = b[i]
            #elif i > j:
                #coarseb[0] = (b[i]-b[abs(j)]+p)%(2*p)-p
        #else:
            #perp_component = v - v.dot(coarseA[0])/(coarseA[0].dot(coarseA[0])) * coarseA[0]
            #if la.norm(perp_component) > tol:
                #coarseA[1] = v
                #if i < j:
                    #coarseb[1] = (b[i]+b[j]+p)%(2*p)-p
                #elif i == j:
                    #coarseb[1] = b[i]
                #elif i > j:
                    #coarseb[1] = (b[i]-b[abs(j)]+p)%(2*p)-p
                #break

    #if la.norm(coarseA[1]) == 0:
        #raise Exception("Poorly constructed A matrix: cannot find a pair of orthogonal vectors")

    ## find coarse solutions
    #try:
        #icA = la.inv(coarseA)
    #except:
        #raise Exception("Poorly constructed coarseA matrix: %s."%(coarseA))
    #try:
        #iA = la.inv(A.transpose().dot(A))
    #except:
        #raise Exception("Poorly constructed A matrix: %s."%(A.transpose().dot(A)))

    #if b.ndim > 2:
        #extra_shape = b.shape[1:]
        #flat_extra_dim = 1
        #for i in range(1, b.ndim):
            #flat_extra_dim = flat_extra_dim * b.shape[i]
        #coarseb.shape = (2, flat_extra_dim)
        #b.shape = (len(b), flat_extra_dim)
    #else:
        #extra_shape = None

    #result = icA.dot(coarseb)
    #for i in range(niter):
        #result = result + iA.dot(A.transpose().dot((b - A.dot(result) + p)%(2*p)-p))

    #if extra_shape is not None:
        #result.shape = tuple(np.concatenate(([2], extra_shape)))

    #return result

def extract_crosspol_ubl(data, info):
    '''input data should be xy/yx (2,...,bl)'''
    if len(data) != 2:
        raise AttributeError('Datas first demension need to have length 2 corresponding to xy/yx. Current input shape %s.'%data.shape)

    output_shape = np.array(data.shape)
    output_shape[-1] = len(info.ublcount)
    output = np.empty(output_shape, dtype='complex64')
    chisq = np.zeros(output_shape[:-1], dtype='float32')

    for u in range(len(info.ublcount)):
        blindex = info['subsetbl'][info['crossindex'][info['ublindex'][u][:,2].astype(int)]]
        #ureversed = info['reversed'][info['ublindex'][u][:,2].astype(int)] == -1
        ureversed = n.ones_like(info['ublindex'][u][:,2], dtype=n.int) == -1
        nreversed = np.sum(ureversed)
        if nreversed == 0:#no reversed
            output[..., u] = np.mean(data[..., blindex], axis=-1)
            chisq += np.linalg.norm(output[..., u][...,None] - data[..., blindex], axis=-1)**2
        elif nreversed == info['ublcount'][u]:
            output[..., u] = np.conjugate(np.mean(data[::-1, ..., blindex], axis=-1))
            chisq += np.linalg.norm(output[..., u][...,None] - np.conjugate(data[::-1, ..., blindex]), axis=-1)**2
        else:
            output[..., u] = (np.mean(data[..., blindex[~ureversed]], axis=-1) * (info['ublcount'][u] - nreversed) + np.conjugate(np.mean(data[::-1, ..., blindex[ureversed]], axis=-1)) * nreversed) / info['ublcount'][u]
            chisq += np.linalg.norm(output[..., u][...,None] - data[..., blindex[~ureversed]], axis=-1)**2 + np.linalg.norm(output[..., u][...,None] - np.conjugate(data[::-1, ..., blindex[ureversed]]), axis=-1)**2
    return output, chisq

# XXX data compression stuff belongs in another file
def deconvolve_spectra(spectra, window, band_limit, correction_weight=1e-15):
    '''solve for band_limit * 2 -1 bins, returns the deconvolved solution and
    the norm of fitting error. All fft will be along first axis of spectra.
    Input and outputs are in fourier space, window in real space'''
    if len(spectra) != len(window):
        raise ValueError("Input spectra and window function have unequal lengths %i %i."%(len(spectra), len(window)))
    #if np.sum(window) <= 2* band_limit - 1:
        #return np.zeros(2*band_limit - 1, dtype=np.array(spectra).dtype), np.inf
    fwindow = np.fft.fft(window) / len(window)
    band_limit_pass = np.zeros(len(fwindow), dtype='bool')
    band_limit_pass[:band_limit] = True
    if band_limit > 1:
        band_limit_pass[-(band_limit-1):] = True

    m = la.toeplitz(fwindow, np.roll(fwindow[::-1], 1)).astype('complex128')[:, band_limit_pass]
    mmi = la.inv(m.transpose().conjugate().dot(m) + np.identity(m.shape[1])*correction_weight)
    deconv_fdata = mmi.dot(m.transpose().conjugate()).dot(spectra)
    model_fdata = m.dot(deconv_fdata)
    return deconv_fdata, np.linalg.norm(model_fdata-spectra, axis = 0)

# XXX data compression stuff belongs in another file
def deconvolve_spectra2(spectra, window, band_limit, var=None, correction_weight=1e-15, correction_weight2=1e6):
    '''solve for band_limit * 2 -1 bins, returns the deconvolved solution
    and the norm of fitting error. All fft will be along first axis of
    spectra. Input and outputs are in real space, window also in real space'''
    if len(spectra) != len(window):
        raise ValueError("Input spectra and window function have unequal lengths %i %i."%(len(spectra), len(window)))
    #if np.sum(window) <= 2* band_limit - 1:
        #return np.zeros(2*band_limit - 1, dtype=np.array(spectra).dtype), np.inf
    if var is None:
        var = np.ones(len(window))
    elif len(var) != len(window):
        raise ValueError("Input var and window function have unequal lengths %i %i."%(len(var), len(window)))

    if np.sum(window) == 0:
        return np.zeros([2* band_limit - 1] + list(spectra.shape[1:])), np.zeros(spectra.shape[1:]), np.zeros_like(spectra), np.zeros((2* band_limit - 1, 2* band_limit - 1))+np.inf

    fwindow = np.fft.fft(window) / len(window)
    band_limit_pass = np.zeros(len(fwindow), dtype='bool')
    band_limit_pass[:band_limit] = True
    if band_limit > 1:
        band_limit_pass[-(band_limit-1):] = True

    #m = la.inv(la.dft(len(window))).dot(la.toeplitz(fwindow, np.roll(fwindow[::-1], 1)).astype('complex128')[:, band_limit_pass].dot(la.dft(2*band_limit - 1)))
    m = np.fft.ifft(la.toeplitz(fwindow, np.roll(fwindow[::-1], 1)).astype('complex128')[:, band_limit_pass].dot(la.dft(2*band_limit - 1)), axis=0)
    Ni = 1./np.copy(var)
    Ni[window==0] = np.nanmax(Ni) * correction_weight2

    #Ni = np.ones_like(window) + (1-window) * correction_weight2
    mmi = la.inv((m.transpose().conjugate() * Ni).dot(m) + np.identity(m.shape[1])*np.median(Ni[window==1])*correction_weight)
    mult_window = np.copy(window)
    mult_window.shape = [len(mult_window)] + [1]*(spectra.ndim-1)
    deconv_fdata = (mmi.dot(m.transpose().conjugate()) * Ni).dot(spectra*mult_window)
    model_fdata = m.dot(deconv_fdata)
    return deconv_fdata, np.linalg.norm(model_fdata-spectra*mult_window, axis = 0), model_fdata-spectra*mult_window, mmi
