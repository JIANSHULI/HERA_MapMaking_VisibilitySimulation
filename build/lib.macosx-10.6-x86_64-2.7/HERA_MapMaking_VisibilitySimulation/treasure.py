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

class RedundantCalibrator:
    '''This class is the main tool for performing redundant calibration on data sets. 
    For a given redundant configuration, say 32 antennas with 3 bad antennas, the 
    user should create one instance of Redundant calibrator and reuse it for all data 
    collected from that array. In general, upon creating an instance, the user need 
    to create the info field of the instance by either computing it or reading it 
    from a text file. readyForCpp(verbose = True) should be a very helpful function 
    to provide information on what information is missing for running the calibration.'''

    # XXX treasure stuff is not necessary for core functionality.  make a subclass that adds treasure capability and move all treasure stuff to another file
    def absolutecal_w_treasure(self, treasure, pol, lsts, tolerance = None, MIN_UBL_COUNT = 50, static_treasure = True):#phase not yet implemented
        '''XXX DOCSTRING'''
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)

            if self.nTime != len(lsts):
                raise TypeError("Input lsts has wrong length of %i rather than expected %i."%(len(lsts), self.nTime))

            if type(treasure) == type('aa'):
                treasure = Treasure(treasure)
            if self.nFrequency != treasure.nFrequency:
                raise TypeError("Treasure has %i frequency bins rather than expected %i in calibrator."%(treasure.nFrequency, self.nFrequency))
            treasure_bls = treasure.ubls[pol]
            if tolerance is None:
                tolerance = treasure.tolerance
            else:
                treasure.tolerance = tolerance
            ubl_overlap = np.zeros(self.nUBL, dtype='bool')
            for i, ubl in enumerate(self.ubl):
                ubl_overlap[i] = (np.min(np.linalg.norm(treasure_bls - ubl, axis = 1)) < tolerance)

            if np.sum(ubl_overlap) > MIN_UBL_COUNT:

                original_data = self.rawCalpar[..., 3 + 2 * self.nAntenna::2] + 1.j * self.rawCalpar[..., 3 + 2 * self.nAntenna + 1::2]
                data = original_data[..., ubl_overlap]
                model = np.zeros_like(data)
                model_flag = np.zeros(data.shape, dtype='bool')

                iterants = zip(range(np.sum(ubl_overlap)), self.ubl[ubl_overlap])
                np.random.shuffle(iterants)
                for i, ubl in iterants:
                    coin = treasure.get_interpolated_coin((pol, ubl), lsts, static_treasure=static_treasure)
                    if coin is None:
                        coin = treasure.get_interpolated_coin((pol, -ubl), lsts, static_treasure=static_treasure)
                        if coin is None:
                            model_flag[..., i] = True
                        else:
                            model[..., i] = np.conjugate(coin.weighted_mean)
                            model_flag[..., i] = coin.count < 1

                    else:
                        model[..., i] = coin.weighted_mean
                        model_flag[..., i] = coin.count < 1

                #flatten the first 2 t/f axes
                data.shape = (data.shape[0]*data.shape[1], data.shape[2])
                model.shape = (model.shape[0]*model.shape[1], model.shape[2])
                model_flag.shape = (model_flag.shape[0]*model_flag.shape[1], model_flag.shape[2])


                #now try to find two types of time/freqs: not enough model or have enough model; then, among the time/freqs that have enough model, find the ubl models that is valid on all of them
                good_slot = (np.sum(~model_flag, axis=-1) > MIN_UBL_COUNT)
                ubl_valid = (np.sum(~model_flag[good_slot], axis=0) == np.sum(good_slot))#whether the ubl has any measurements in this entire lst range

                if np.sum(ubl_valid) > MIN_UBL_COUNT:
                    #now it's safe to assume that on good_slot and ubl_valid, all model_flag is False
                    if np.sum(model_flag[good_slot][:,ubl_valid]) > 0:
                        raise ValueError('Logic error: the flag here should be all False.')
                    ##data = data[..., ubl_valid]
                    ##model = model[..., ubl_valid]
                    ##model_flag = model_flag[..., ubl_valid]
                    N = 1. / self.ublcount[ubl_overlap][ubl_valid]

                    #amplitude
                    damp = np.abs(data[good_slot][:, ubl_valid])
                    mamp = np.abs(model[good_slot][:, ubl_valid])
                    amp_cal = np.ones(self.nTime*self.nFrequency, dtype='float32')
                    amp_cal[good_slot] = np.sum(mamp * damp / N[None, :], axis = -1) / np.sum(mamp**2 / N[None, :], axis = -1)
                    #print damp[30], mamp[30], amp_cal[good_slot][30]
                    ###ratio = data/model
                    ###model_flag = model_flag|np.isnan(ratio)|np.isinf(ratio)
                    ####amplitude
                    ###amp_weights = (~model_flag) * self.ublcount[ubl_overlap][None,None,ubl_valid]
                    ###no_piror = (np.sum(amp_weights, axis = -1) <= MIN_UBL_COUNT)#not enough ubl coverage
                    ###ratio[model_flag] = 0
                    ###amp_weights[np.sum(amp_weights, axis = -1) ==0] = 1.#avoid all 0 weights
                    ###amp_cal = np.average(np.abs(ratio), axis = -1, weights = amp_weights)
                    ###amp_cal[no_piror] = 1.

                    #phase
                    A = self.ubl[ubl_overlap][ubl_valid]
                    A = A - np.mean(A, axis = 0)[None, :]
                    AtNiAiAtNi = la.pinv((A.transpose()/N[None,:]).dot(A)).dot((A.transpose()/N[None,:]))

                    phs_sol = np.zeros((np.sum(good_slot), A.shape[1]), dtype='float32')
                    Del = np.angle(data[good_slot][:, ubl_valid]) - np.angle(model[good_slot][:, ubl_valid])
                    for i in range(25):
                        delphs = (Del - phs_sol.dot(A.transpose()) + PI)%TPI - PI
                        phs_sol = phs_sol + delphs.dot(AtNiAiAtNi.transpose())

                    phs_cal = np.zeros((self.nTime*self.nFrequency, A.shape[1]), dtype='float32')
                    phs_cal[good_slot] = phs_sol

                    #apply results to rawCalpar
                    amp_cal = amp_cal.reshape((self.nTime, self.nFrequency))
                    phs_cal = phs_cal.reshape((self.nTime, self.nFrequency, phs_cal.shape[1]))
                    good_slot = good_slot.reshape((self.nTime, self.nFrequency))

                    #antenna stuff
                    self.rawCalpar[..., 3: 3 + self.nAntenna] = self.rawCalpar[..., 3: 3 +  self.nAntenna] + np.log10(amp_cal[..., None]) / 2.
                    self.rawCalpar[..., 3 + self.nAntenna: 3 + 2 * self.nAntenna] = self.rawCalpar[..., 3 + self.nAntenna: 3 + 2 * self.nAntenna] + phs_cal.dot(self.antloc.transpose())

                    #ubl stuff
                    calibrated_data = original_data / amp_cal[..., None] / np.exp(1.j * phs_cal.dot(self.ubl.transpose()))
                    self.rawCalpar[..., 3 + 2 * self.nAntenna::2] = np.real(calibrated_data)
                    self.rawCalpar[..., 3 + 2 * self.nAntenna + 1::2] = np.imag(calibrated_data)



                    self.rawCalpar[..., 1] = self.rawCalpar[..., 2] / amp_cal**2
                    self.rawCalpar[~good_slot, 1] = -1
                    return amp_cal, phs_cal
                else:
                    return ubl_valid
            else:
                return ubl_overlap

    # XXX treasure stuff is not necessary for core functionality.  make a subclass that adds treasure capability and move all treasure stuff to another file
    def update_treasure(self, treasure, lsts, flags, pol, nsigma_cut = 5, verbose = False):#lsts in radians
        '''XXX DOCSTRING'''
        if type(treasure) == type('aa'):
            treasure = Treasure(treasure)
        if len(lsts) != self.nTime:
            raise TypeError("lsts has length %i which disagrees with RedundantCalibrator's nTime of %i."%(len(lsts), nTime))
        iterants = zip(range(self.nUBL), self.ubl)
        np.random.shuffle(iterants)
        for i, ublvec in iterants:
            if verbose:
                print ".",
                sys.stdout.flush()
            visibilities = self.rawCalpar[..., 3+2*self.nAntenna+2*i] + 1.j*self.rawCalpar[..., 4+2*self.nAntenna+2*i]
            visibilities[flags] = np.nan
            abscal_factor = 10**(4 * np.median(self.rawCalpar[..., 3:3+self.nAntenna], axis = -1))
            dof = (len(self.crossindex)-self.nAntenna-self.nUBL+2)
            treasure.update_coin((pol, ublvec), lsts, visibilities, self.rawCalpar[..., 2]/2./abscal_factor/dof/self.ublcount[i], nsigma_cut = nsigma_cut,verbose=verbose)#divide by 2 because epsilon^2 should be for real/imag separately

# XXX utility function belongs in another file
def read_ndarray(path, shape, dtype, ranges):
    '''read middle part of binary file of shape and dtype specified by ranges of the first dimension. ranges is [inclusive, exclusive)'''
    if not os.path.isfile(path):
        raise IOError(path + 'doesnt exist.')
    if len(ranges) != 2 or ranges[0] < 0 or ranges[0] >= ranges[1] or ranges[1] > shape[0]:
        raise ValueError("%s is not a vlid range."%ranges)
    nbytes = np.dtype(dtype).itemsize
    higher_dim_chunks = 1 # product of higher dimensions. if array is (2,3,4,5), this is 3*4*5
    for m in shape[1:]:
        higher_dim_chunks = higher_dim_chunks * m

    #print np.fromfile(path, dtype = dtype).shape
    with open(path, 'r') as f:
        f.seek(higher_dim_chunks * nbytes * ranges[0])
        #print higher_dim_chunks * nbytes * ranges[0]
        result = np.fromfile(f, dtype = dtype, count = (ranges[1] - ranges[0]) * higher_dim_chunks)
    new_shape = np.array(shape)
    new_shape[0] = (ranges[1] - ranges[0])
    #print result.shape,tuple(new_shape)
    return result.reshape(tuple(new_shape))

# XXX utility function belongs in another file
def write_ndarray(path, shape, dtype, ranges, data, check = True, max_retry = 3, task = 'unkown'):
    '''write middle part of binary file of shape and dtype specified by ranges of the first dimension. ranges is [inclusive, exclusive)'''
    if not os.path.isfile(path):
        raise IOError(path + 'doesnt exist.')
    if len(ranges) != 2 or ranges[0] < 0 or ranges[0] >= ranges[1] or ranges[1] > shape[0]:
        raise ValueError("%s is not a vlid range."%ranges)
    if data.dtype != dtype or data.shape[1:] != shape[1:] or data.shape[0] != ranges[1] - ranges[0]:
        raise ValueError("data shape %s cannot be fit into data file shape %s."%(data.shape, shape))
    nbytes = np.dtype(dtype).itemsize
    higher_dim_chunks = 1 # product of higher dimensions. if array is (2,3,4,5), this is 3*4*5
    for m in shape[1:]:
        higher_dim_chunks = higher_dim_chunks * m
    with open(path, 'r+') as f:
        f.seek(higher_dim_chunks * nbytes * ranges[0])
        data.tofile(f)
    if check:
        tries = 0
        while not (data == read_ndarray(path, shape, dtype, ranges)).all() and tries < max_retry:

            time.sleep(1)
            tries = tries + 1
            with open(path, 'r+') as f:
                f.seek(higher_dim_chunks * nbytes * ranges[0])
                data.tofile(f)
        if not (data == read_ndarray(path, shape, dtype, ranges)).all():
            raise IOError("write_ndarray failed on %s with shape %s between %s with task %s."%(path, shape, ranges, task))
    return

#  _____                             
# |_   _| _ ___ __ _ ____  _ _ _ ___ 
#   | || '_/ -_) _` (_-< || | '_/ -_)
#   |_||_| \___\__,_/__/\_,_|_| \___|

# XXX treasure stuff is not necessary for core functionality.  make a subclass that adds treasure capability and move all treasure stuff to another file
class Treasure:
    '''XXX DOCSTRING'''
    def __init__(self, folder_path, nlst = int(TPI/1e-3), nfreq = 1024, tolerance = .1):
        folder_path = os.path.expanduser(folder_path) + '/'
        if os.path.isdir(folder_path):
            self.folderPath = folder_path
            with open(self.folderPath + '/header.txt', 'r') as f:
                self.nTime = int(f.readline())
                self.nFrequency = int(f.readline())
                self.lsts = np.arange(0, TPI, TPI/self.nTime)
                self.frequencies = np.arange(self.nFrequency)
                self.coinShape = (self.nTime, self.nFrequency, int(f.readline()))#N, real(v), imag(v), real(v)^2, imag(v)^2, epsilon^-2, real(v)epsilon^-2, imag(v)epsilon^-2, placeholder1, placeholder2; epsilon^2 should be for only real part/imag part, and should be same for both
                self.coinDtype = f.readline().replace('\n', '')
                self.sealDtype = f.readline().replace('\n', '')
                self.sealSize = int(f.readline())
                self.tolerance = float(f.readline())
                self.sealPosition = None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ubldata = np.loadtxt(f, dtype={'names':('x','y','z','pol'),'formats':('f','f','f','S10')})
                self.ubls = {}
                for x,y,z,pol in ubldata:
                    if pol not in self.ubls.keys():
                        self.ubls[pol] = np.array([[x, y, z]], dtype= 'float64')
                    else:
                        self.ubls[pol] = np.append(self.ubls[pol], [[x,y,z]], axis = 0)
        else:
            self.folderPath = folder_path
            self.nTime = nlst
            self.nFrequency = nfreq
            self.lsts = np.arange(0, TPI, TPI/self.nTime)
            self.frequencies = np.arange(self.nFrequency)
            self.coinShape = (self.nTime, self.nFrequency, 10)#N, real(v), imag(v), real(v)^2, imag(v)^2, epsilon^-2, real(v)epsilon^-2, imag(v)epsilon^-2, placeholder1, placeholder2; epsilon^2 should be for only real part/imag part, and should be same for both
            self.coinDtype = 'float64'
            self.sealDtype = 'bool'
            self.sealSize = 4096
            self.tolerance = tolerance
            self.sealPosition = None
            self.ubls = {}
            self.duplicate_treasure(folder_path)

    def __repr__(self):
        return "Treasure instance at %s with %i time slices and %i frequency slices on %s polarizations."%(self.folderPath, self.nTime, self.nFrequency, self.ubls.keys())

    def __str__(self):
        return self.__repr__()

    def coin_name(self, polvec):
        '''XXX DOCSTRING'''
        pol, ublvec = polvec
        if pol in self.ubls.keys():
            match_flag = np.linalg.norm(self.ubls[pol] - ublvec, axis = 1) <= self.tolerance
            if np.sum(match_flag) > 0:
                return self.folderPath + '/%s%i.coin'%(pol, np.arange(len(self.ubls[pol]))[match_flag][0])
        return None

    def seal_name(self, polvec):
        '''XXX DOCSTRING'''
        pol, ublvec = polvec
        coinname = self.coin_name(polvec)
        if coinname is None:
            return None
        else:
            return coinname.replace('coin', 'seal')

    def have_coin(self, polvec):
        '''XXX DOCSTRING'''
        pol, ublvec = polvec
        return (self.coin_name(polvec) is not None)

    def duplicate_treasure(self, folder_path):
        '''XXX DOCSTRING'''
        pol, ublvec = polvec
        folder_path = os.path.expanduser(folder_path)
        if os.path.exists(folder_path):
            raise IOError("Requested folder path %s already exists."%folder_path)
        os.makedirs(folder_path)
        for pol in self.ubls.keys():
            for ublvec in self.ubls[pol]:
                polvec = (pol, ublvec)
                if not self.seize_coin(polvec):
                    shutil.rmtree(folder_path)
                    return None

                shutil.copy(self.coin_name(polvec), folder_path)
                self.release_coin(polvec)
                np.zeros(self.sealSize, dtype=self.sealDtype).tofile(self.seal_name(polvec))
        with open(folder_path + '/header.txt', 'w') as f:
            f.write('%i\n'%self.nTime)
            f.write('%i\n'%self.nFrequency)
            f.write('%i\n'%self.coinShape[2])
            f.write(self.coinDtype + '\n')
            f.write(self.sealDtype + '\n')
            f.write('%i\n'%self.sealSize)
            f.write('%.3e\n'%self.tolerance)
            for pol in self.ubls.keys():
                for ublvec in self.ubls[pol]:
                    f.write('%f %f %f %s\n'%(ublvec[0], ublvec[1], ublvec[2], pol))
        new_treasure = Treasure(folder_path)
        return new_treasure

    def update_coin(self, polvec, lsts, visibilities, epsilonsqs, nsigma_cut = None, verbose=False):
        '''lsts should be [0,TPI); visibilities should be 2D np array nTime by 
        nFrequency, epsilonsqs should be for real/imag parts seperately (should 
        be same though); to flag any data point make either visibilities or epsilonsqs 
        np.nan, or make epsilonsqs 0'''
        lsts = np.array(lsts)
        if visibilities.shape != (len(lsts), self.nFrequency):
            raise ValueError("visibilities array has wrong shape %s that does not agree with %s."%(visibilities.shape, (len(lsts), self.nFrequency)))
        if epsilonsqs.shape != (len(lsts), self.nFrequency):
            raise ValueError("epsilonsqs array has wrong shape %s that does not agree with %s."%(epsilonsqs.shape, (len(lsts), self.nFrequency)))
        if np.max(lsts) > TPI or np.min(lsts) < -TPI/self.nTime:
            raise ValueError("lsts range [%f, %f] is not inside the expected [%.2f, 2pi)."%(np.max(lsts), np.min(lsts), -TPI/self.nTime))
        if len(lsts) > 1 and (np.max(lsts[1:] - lsts[:-1]) > TPI/self.nTime * 1.001):
            raise ValueError("lsts interval is %f, which is larger than desired grid size %f."%(np.max(lsts[1:] - lsts[:-1]), TPI/self.nTime))

        n_wrap = np.sum(lsts[1:] - lsts[:-1] < 0)
        if n_wrap > 1:
            raise ValueError("lsts is not a continuous list of times. Only one wrap around from 2pi to 0 allowed.")
        elif n_wrap == 1:
            iwrap = int(np.argsort(lsts[1:] - lsts[:-1])[0]) + 1#wrappping happend between iwrap-1 and iwrap
            if iwrap > 1:
                self.update_coin(polvec, lsts[:iwrap], visibilities[:iwrap], epsilonsqs[:iwrap], verbose=verbose)
            self.update_coin(polvec, np.append([lsts[iwrap-1] - TPI], lsts[iwrap:]), visibilities[iwrap-1:], epsilonsqs[iwrap-1:], verbose=verbose)
            return
        else:
            if not self.have_coin(polvec):
                if verbose:
                    print "Adding new coin %s"%polvec
                self.add_coin(polvec)

            update_range = np.array([np.ceil(lsts[0]/(TPI/self.nTime)), np.ceil(lsts[-1]/(TPI/self.nTime))], dtype = 'int32') # [inclusive, exclusive)


            update_flag = np.ceil(lsts[:-1]/(TPI/self.nTime)) == np.floor(lsts[1:]/(TPI/self.nTime))
            update_lsts = np.floor(lsts[1:]/(TPI/self.nTime))[update_flag] * TPI/self.nTime
            left_lsts = lsts[:-1][update_flag]
            right_lsts = lsts[1:][update_flag]
            left_distance = update_lsts - left_lsts
            right_distance = right_lsts - update_lsts
            weight_left = right_distance / (lsts[1:][update_flag] - lsts[:-1][update_flag])
            weight_right = left_distance / (lsts[1:][update_flag] - lsts[:-1][update_flag])
            update_visibilities = weight_left[:, None] * visibilities[:-1][update_flag] + weight_right[:, None] * visibilities[1:][update_flag]#sp.interpolate.interp1d(lsts, visibilities, kind='linear', axis=0, copy=True, bounds_error=True, assume_sorted=True)(self.lsts[update_range[0]:update_range[1]])
            #update_epsilonsqs = sp.interpolate.interp1d(lsts**2, epsilonsqs, kind='linear', axis=0, copy=True, bounds_error=True, assume_sorted=True)(self.lsts[update_range[0]:update_range[1]]**2)
            update_epsilonsqs = (weight_left**2)[:, None] * epsilonsqs[:-1][update_flag] + (weight_right**2)[:, None] * epsilonsqs[1:][update_flag]
            #print weight_left, weight_right
            good_flag = ~(np.isnan(update_epsilonsqs) | np.isnan(update_visibilities) | np.isinf(update_epsilonsqs) | np.isinf(update_visibilities) | (update_epsilonsqs == 0))


            if not self.seize_coin(polvec):
                return False
            coin_name = self.coin_name(polvec)
            coin_content = read_ndarray(coin_name, self.coinShape, self.coinDtype, update_range)

            if nsigma_cut is not None and nsigma_cut > 0:#only update data within a certain nsigma range of existing weighted mean and weighted variance
                coin = Coin(coin_content)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",category=RuntimeWarning)
                    nsigma_flag = (coin.count > 0) & (np.abs(coin.weighted_mean - update_visibilities) > nsigma_cut * (coin.weighted_variance * coin.count)**.5)
                good_flag = good_flag & (~nsigma_flag)

            coin_content[..., 0] = coin_content[..., 0] + good_flag
            coin_content[good_flag, 1] = coin_content[good_flag, 1] + np.real(update_visibilities[good_flag])
            coin_content[good_flag, 2] = coin_content[good_flag, 2] + np.imag(update_visibilities[good_flag])
            coin_content[good_flag, 3] = coin_content[good_flag, 3] + np.real(update_visibilities[good_flag])**2
            coin_content[good_flag, 4] = coin_content[good_flag, 4] + np.imag(update_visibilities[good_flag])**2
            coin_content[good_flag, 5] = coin_content[good_flag, 5] + np.real(update_visibilities[good_flag])/update_epsilonsqs[good_flag]
            coin_content[good_flag, 6] = coin_content[good_flag, 6] + np.imag(update_visibilities[good_flag])/update_epsilonsqs[good_flag]
            coin_content[good_flag, 7] = coin_content[good_flag, 7] + update_epsilonsqs[good_flag]**-1
            #print coin_content[good_flag, 7]
            write_ndarray(coin_name, self.coinShape, self.coinDtype, update_range, coin_content, check=True, task = 'update_coin')
            self.release_coin(polvec)
            return True

    #def get_coin_index(self, ublvec):
        #if len(ublvec) != 3:
            #raise ValueError("ublvec %s is not a 3D vector."%ublvec)
        #if len(self.ubls) < 1:
            #return None
        #deltas = np.linalg.norm(self.ubls - ublvec, axis = 1)
        #if np.min(deltas) <= self.tolerance:
            #return np.argsort(deltas)[0]
        #else:
            #return None

    def add_coin(self, polvec, coin_data=None):
        '''XXX DOCSTRING'''
        pol, ublvec = polvec
        if len(ublvec) != 3:
            raise ValueError("ublvec %s is not a 3D vector."%ublvec)
        if coin_data is not None:
            if coin_data.shape != self.coinShape or coin_data.dtype != self.coinDtype:
                raise TypeError("Input coin data %s %s does not agree with treasure's %s %s."%(coin_data.shape, coin_data.dtype, self.coinShape, self.coinDtype))
            if self.have_coin(polvec):
                raise IOError("Treasure already has coin %s."%polvec)
        if not self.have_coin(polvec):
            if pol not in self.ubls.keys():
                self.ubls[pol] = np.array([ublvec], dtype= 'float64')
            else:
                self.ubls[pol] = np.append(self.ubls[pol], [ublvec] ,axis = 0)
            if coin_data is None:
                np.zeros(self.coinShape, dtype = self.coinDtype).tofile(self.coin_name(polvec))
            else:
                coin_data.tofile(self.coin_name(polvec))
            np.zeros(self.sealSize, dtype = self.sealDtype).tofile(self.seal_name(polvec))
            with open(self.folderPath + '/header.txt', 'a') as f:
                f.write('%f %f %f %s\n'%(ublvec[0], ublvec[1], ublvec[2], pol))
        return

    def get_coin(self, polvec, ranges=None, retry_wait = 1, max_wait = 30, static_treasure=False):
        '''ranges is index range [incl, exc)'''
        if ranges is not None:
            if len(ranges) != 2 or ranges[0] < 0 or ranges[1] > self.nTime:
                raise ValueError("range specification %s is not allowed."%ranges)
            ranges = [int(ranges[0]), int(ranges[1])]
        if static_treasure or self.seize_coin(polvec, retry_wait = retry_wait, max_wait = max_wait):
            if ranges is None:
                coin = Coin(np.fromfile(self.coin_name(polvec), dtype = self.coinDtype).reshape(self.coinShape))
            else:
                coin = Coin(read_ndarray(self.coin_name(polvec), self.coinShape, self.coinDtype, ranges))
            if not static_treasure:
                self.release_coin(polvec)
            return coin
        else:
            return None

    def get_interpolated_coin(self, polvec, lsts, retry_wait = 1, max_wait = 10, static_treasure = False):
        '''lsts in [0, 2pi)'''
        if not self.have_coin(polvec):
            return None
        lsts = np.array(lsts)
        if np.min(lsts) <= 0 or np.max(lsts) > TPI:
            raise ValueError("lsts is not inside [0, 2pi)")
        if np.ceil(np.max(lsts)/(TPI/self.nTime)) >= self.nTime:
            ranges = [int(np.floor(np.min(lsts)/(TPI/self.nTime))), int(np.ceil(np.max(lsts)/(TPI/self.nTime)))]
            coin = self.get_coin(polvec, ranges = ranges, retry_wait = retry_wait, max_wait = max_wait, static_treasure=static_treasure)
            coin2 = self.get_coin(polvec, ranges = [0,1], retry_wait = retry_wait, max_wait = max_wait, static_treasure=static_treasure)
            if coin is None or coin2 is None:
                return None
            coin.data = np.concatenate((coin.data, coin2.data))
            grid_lsts = np.append(self.lsts[ranges[0]:ranges[1]], TPI)
        else:
            ranges = [int(np.floor(np.min(lsts)/(TPI/self.nTime))), int(np.ceil(np.max(lsts)/(TPI/self.nTime))) + 1]
            coin = self.get_coin(polvec, ranges = ranges, retry_wait = retry_wait, max_wait = max_wait, static_treasure=static_treasure)
            if coin is None:
                return None
            grid_lsts = self.lsts[ranges[0]:ranges[1]]

        interp_coin = FakeCoin()
        interp_coin.count = interpolate.interp1d(grid_lsts, coin.count, axis = 0)(lsts)
        interp_coin.weighted_mean = interpolate.interp1d(grid_lsts, coin.weighted_mean, axis = 0)(lsts)
        interp_coin.mean = interpolate.interp1d(grid_lsts, coin.mean, axis = 0)(lsts)
        interp_coin.variance_re = interpolate.interp1d(grid_lsts, coin.variance_re, axis = 0)(lsts)
        interp_coin.variance_im = interpolate.interp1d(grid_lsts, coin.variance_im, axis = 0)(lsts)
        interp_coin.weighted_variance = interpolate.interp1d(grid_lsts, coin.weighted_variance, axis = 0)(lsts)

        zero_count_flag = (coin.count[np.floor((lsts-np.min(lsts))/(TPI/self.nTime)).astype(int)] == 0) | (coin.count[np.ceil((lsts-np.min(lsts))/(TPI/self.nTime)).astype(int)] == 0)
        interp_coin.count[zero_count_flag] = 0
        interp_coin.variance_re[zero_count_flag] = np.inf
        interp_coin.variance_im[zero_count_flag] = np.inf
        interp_coin.weighted_variance[zero_count_flag] = np.inf
        return interp_coin

    def seal_all(self):
        '''XXX DOCSTRING'''
        for pol in self.ubls.keys():
            for u in self.ubls[pol]:
                np.zeros(self.sealSize, dtype = self.sealDtype).tofile(self.seal_name((pol, u)))

    def get_coin_now(self, polvec, ranges=None):
        '''XXX DOCSTRING'''
        return self.get_coin(polvec, ranges=ranges, retry_wait = 0.1, max_wait = .5 )

    def seize_coin(self, polvec, retry_wait = 1, max_wait = 30):
        '''XXX DOCSTRING'''
        if self.sealPosition is not None:
            raise TypeError("Treasure class is trying to seize coin without properly release previous seizure.")
        if not self.have_coin(polvec):
            return False

        start_time = time.time()
        while not self.try_coin(polvec) and time.time()-start_time < max_wait:
            time.sleep(retry_wait)
        if time.time()-start_time > max_wait:
            return False
        seal_position = np.random.random_integers(self.sealSize) - 1
        seal_name = self.seal_name(polvec)
        write_ndarray(seal_name, (self.sealSize,), self.sealDtype, [seal_position, seal_position + 1], np.array([1], dtype=self.sealDtype), check = True, max_retry = max_wait, task = 'seize_coin')
        if np.sum(np.fromfile(seal_name, dtype=self.sealDtype)) == 1:
            self.sealPosition = seal_position
            return True
        else:
            write_ndarray(seal_name, (self.sealSize,), self.sealDtype, [seal_position, seal_position + 1], np.array([0], dtype=self.sealDtype), check = True, max_retry = 10 * max_wait, task = 'abort_seize_coin')
            return False

    def release_coin(self, polvec):
        '''XXX DOCSTRING'''
        if self.sealPosition is None:
            raise TypeError("Treasure class is trying to release coin without a previous seizure.")

        write_ndarray(self.seal_name(polvec), (self.sealSize,), self.sealDtype, [self.sealPosition, self.sealPosition + 1], np.array([0], dtype=self.sealDtype), check = True, max_retry = 60, task = 'release_coin')
        self.sealPosition = None

    def try_coin(self, polvec):
        '''XXX DOCSTRING'''
        if not self.have_coin(polvec):
            self.add_coin(polvec)
        return np.sum(np.fromfile(self.seal_name(polvec), dtype=self.sealDtype)) == 0

    def burn(self):
        '''XXX DOCSTRING'''
        shutil.rmtree(self.folderPath)

#   ___     _      
#  / __|___(_)_ _  
# | (__/ _ \ | ' \ 
#  \___\___/_|_||_|

class Coin:
    '''XXX DOCSTRING'''
    #def __init__(self, path, shape, dtype):
        #if not os.path.isfile(path):
            #raise IOError("%s doesnt exist."%path)
        #self.data = np.fromfile(path, dtype=dtype).reshape(shape)
        #self.attributes = ['count', 'mean', 'variance_re', 'variance_im', 'weighted_mean', 'weighted_variance']

    def __init__(self, data):
        if data.shape[-1] < 8:
            raise TypeError("Data shape %s cannot be constructed as a Coin."%data.shape)
        self.data = data
        self.attributes = ['count', 'mean', 'variance_re', 'variance_im', 'weighted_mean', 'weighted_variance']

    def __getattr__(self, attr):
        if attr not in self.attributes:
            raise AttributeError("Coin class has no attribute named %s. Valid attributes are:\n %s"%(attr, self.attributes))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            if attr == 'count':
                return self.data[..., 0]
            elif attr == 'mean':
                result = (self.data[..., 1] + 1.j * self.data[..., 2]) / self.data[..., 0]
                result[np.isinf(result)|np.isnan(result)] = 0
                return result
            elif attr == 'variance_re':
                n = self.data[..., 0]
                return (n * self.data[..., 3] - self.data[..., 1]**2) / n / (n-1) / n
            elif attr == 'variance_im':
                n = self.data[..., 0]
                return (n * self.data[..., 4] - self.data[..., 2]**2) / n / (n-1) / n
            elif attr == 'weighted_mean':
                result = (self.data[..., 5] + 1.j * self.data[..., 6]) / self.data[..., 7]
                result[np.isinf(result)|np.isnan(result)] = 0
                return result
            elif attr == 'weighted_variance':
                return 1/self.data[..., 7]

    def __repr__(self):
        return "Coin instance with shape %s and type %s."%(self.data.shape, self.data.dtype)

    def __str__(self):
        return self.__repr__()

class FakeCoin:
    '''XXX DOCSTRING'''
    pass

