'''XXX DOCSTRING'''
# XXX lots of imports... are all necessary?  can code be separated into files with smaller dependency lists?
# XXX this file has gotten huge. need to break into smaller files
# XXX clean house on commented code?
# XXX obey python style conventions
import datetime
import socket, math, random, traceback, ephem, string, commands, datetime, shutil, resource, threading, time
import multiprocessing as mp
from time import ctime
import aipy as ap
import struct
import numpy as np
import os, sys
from optparse import OptionParser
import omnical._omnical as _O
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

FILENAME = "calibration_omni.py"
julDelta = 2415020.# =julian date - pyephem's Observer date
PI = np.pi
TPI = 2 * np.pi

# XXX all this meta stuff about "info" almost assuredly means info needs to be a class
infokeys = ['nAntenna','nUBL','nBaseline','subsetant','antloc','subsetbl','ubl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','ublindex','bl1dmatrix','degenM','A','B','At','Bt','AtAi','BtBi']#,'AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']
infokeys_optional = ['totalVisibilityId']
binaryinfokeys=['nAntenna','nUBL','nBaseline','subsetant','antloc','subsetbl','ubl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','ublindex','bl1dmatrix','degenM','A','B']
cal_name = {0: "Lincal", 1: "Logcal"}

int_infokeys = ['nAntenna','nUBL','nBaseline']
intarray_infokeys = ['subsetant','subsetbl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','ublindex','bl1dmatrix','A','B','At','Bt']
intarray_infokeys_optional = ['totalVisibilityId']

float_infokeys = ['antloc','ubl','degenM','AtAi','BtBi']#,'AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']

def read_redundantinfo_txt(infopath, verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "read_redundantinfo_txt"
    if not os.path.isfile(infopath):
        raise Exception('Error: file %s does not exist!'%infopath)
    timer = time.time()
    with open(infopath) as f:
        rawinfo = np.array([np.array([float(x) for x in line.split()]) for line in f])
    if len(rawinfo) < len(infokeys):
        raise Exception('Error: number of rows in %s (%i) is less than expected length of %i!'%(infopath, len(rawinfo), len(infokeys)))
    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:",  "Reading redundant info...",

    info = {}
    infocount = 0;
    info['nAntenna'] = int(rawinfo[infocount][0]) #number of good antennas among all (64) antennas, same as the length of subsetant
    infocount += 1

    info['nUBL'] = int(rawinfo[infocount][0]) #number of unique baselines
    infocount += 1

    nbl = int(rawinfo[infocount][0])
    info['nBaseline'] = nbl
    infocount += 1


    info['subsetant'] = rawinfo[infocount].astype(int) #the index of good antennas in all (64) antennas
    infocount += 1

    info['antloc'] = rawinfo[infocount].reshape((info['nAntenna'],3)) #the index of good antennas in all (64) antennas
    infocount += 1

    info['subsetbl'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['ubl'] = rawinfo[infocount].reshape((info['nUBL'],3)) #unique baseline vectors
    infocount += 1
    info['bltoubl'] = rawinfo[infocount].astype(int) #cross bl number to ubl index
    infocount += 1
    info['reversed'] = rawinfo[infocount].astype(int) #cross only bl if reversed -1, otherwise 1
    infocount += 1
    info['reversedauto'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['autoindex'] = rawinfo[infocount].astype(int)  #index of auto bls among good bls
    infocount += 1
    info['crossindex'] = rawinfo[infocount].astype(int)  #index of cross bls among good bls
    infocount += 1
    ncross = len(info['crossindex'])
    #info['ncross'] = ncross
    info['bl2d'] = rawinfo[infocount].reshape(nbl, 2).astype(int) #from 1d bl index to a pair of antenna numbers
    infocount += 1
    info['ublcount'] = rawinfo[infocount].astype(int) #for each ubl, the number of good cross bls corresponding to it
    infocount += 1
    info['ublindex'] = range((info['nUBL'])) #//for each ubl, the vector<int> contains (ant1, ant2, crossbl)
    tmp = rawinfo[infocount].reshape(ncross, 3).astype(int)
    infocount += 1
    cnter = 0
    for i in range(info['nUBL']):
        info['ublindex'][i] = np.zeros((info['ublcount'][i],3))
        for j in range(len(info['ublindex'][i])):
            info['ublindex'][i][j] = tmp[cnter]
            cnter+=1
    info['ublindex'] = np.asarray(info['ublindex'])

    info['bl1dmatrix'] = rawinfo[infocount].reshape((info['nAntenna'], info['nAntenna'])).astype(int) #a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
    infocount += 1
    #matrices
    info['degenM'] = rawinfo[infocount].reshape((info['nAntenna'] + info['nUBL'], info['nAntenna']))
    infocount += 1
    info['A'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #A matrix for logcal amplitude
    infocount += 1
    info['B'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #B matrix for logcal phase
    infocount += 1
    ##The sparse matrices are treated a little differently because they are not rectangular
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        info['At'] = info['A'].transpose()
        info['Bt'] = info['B'].transpose()
        info['AtAi'] = la.pinv(info['At'].dot(info['A']).todense(), cond = 10**(-6))#(AtA)^-1
        info['BtBi'] = la.pinv(info['Bt'].dot(info['B']).todense(), cond = 10**(-6))#(BtB)^-1
        info['AtAiAt'] = info['AtAi'].dot(info['At'].todense())#(AtA)^-1At
        info['BtBiBt'] = info['BtBi'].dot(info['Bt'].todense())#(BtB)^-1Bt
        info['PA'] = info['A'].dot(info['AtAiAt'])#A(AtA)^-1At
        info['PB'] = info['B'].dot(info['BtBiBt'])#B(BtB)^-1Bt
        info['ImPA'] = sps.identity(ncross) - info['PA']#I-PA
        info['ImPB'] = sps.identity(ncross) - info['PB']#I-PB
    if verbose:
        print "done. nAntenna, nUBL, nBaseline = %i, %i, %i. Time taken: %f minutes."%(len(info['subsetant']), info['nUBL'], info['nBaseline'], (time.time()-timer)/60.)
    return info


def write_redundantinfo_txt(info, infopath, overwrite = False, verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "*write_redundantinfo_txt*"
    timer = time.time()
    if (not overwrite) and os.path.isfile(infopath):
        raise Exception("Error: a file exists at " + infopath + ". Use overwrite = True to overwrite.")
        return
    if (overwrite) and os.path.isfile(infopath):
        os.remove(infopath)
    f_handle = open(infopath,'a')
    for key in infokeys:
        if key in ['antloc', 'ubl', 'degenM', 'AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']:
            np.savetxt(f_handle, [np.array(info[key]).flatten()])
        elif key == 'ublindex':
            np.savetxt(f_handle, [np.vstack(info[key]).flatten()], fmt = '%d')
        elif key in ['At','Bt']:
            tmp = []
            for i in range(info[key].shape[0]):
                for j in range(info[key].shape[1]):
                    if info[key][i,j] != 0:
                        tmp += [i, j, info[key][i,j]]
            np.savetxt(f_handle, [np.array(tmp).flatten()], fmt = '%d')
        elif key in ['A','B']:
            np.savetxt(f_handle, info[key].todense().flatten(), fmt = '%d')
        else:
            np.savetxt(f_handle, [np.array(info[key]).flatten()], fmt = '%d')
    f_handle.close()
    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:", "Info file successfully written to %s. Time taken: %f minutes."%(infopath, (time.time()-timer)/60.)
    return


def write_redundantinfo_old(info, infopath, overwrite = False, verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "*write_redundantinfo*"
    timer = time.time()
    if (not overwrite) and os.path.isfile(infopath):
        raise Exception("Error: a file exists at " + infopath + ". Use overwrite = True to overwrite.")
        return
    if (overwrite) and os.path.isfile(infopath):
        os.remove(infopath)
    marker = 9999999
    datachunk = [0 for i in range(len(binaryinfokeys)+1)]
    count = 0
    datachunk[count] = np.array([marker])         #start with a marker
    count += 1
    if verbose:
                print "appending",
    for key in binaryinfokeys:
        if key in ['antloc', 'ubl','degenM', 'AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']:  #'antloc',
            add = np.append(np.array(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
        elif key == 'ublindex':
            add = np.append(np.vstack(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
        elif key in ['A','B']:
            add = np.append(np.array(info[key].todense().flatten()).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
        else:
            add = np.append(np.array(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
    if verbose:
        print ""
    datachunkarray = array('d',np.concatenate(tuple(datachunk)))
    outfile=open(infopath,'wb')
    datachunkarray.tofile(outfile)
    outfile.close()
    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:", "Info file successfully written to %s. Time taken: %f minutes."%(infopath, (time.time()-timer)/60.)
    return

def write_redundantinfo(info, infopath, overwrite = False, verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "*write_redundantinfo*"
    timer = time.time()
    infopath = os.path.expanduser(infopath)
    if (not overwrite) and os.path.isfile(infopath):
        raise Exception("Error: a file exists at " + infopath + ". Use overwrite = True to overwrite.")
        return
    if (overwrite) and os.path.isfile(infopath):
        os.remove(infopath)

    binaryinfokeysnew = binaryinfokeys[:]
    threshold = 128
    if info['nAntenna'] > threshold:
        binaryinfokeysnew.extend(['AtAi','BtBi'])
    if 'totalVisibilityId' in info.keys():
        binaryinfokeysnew.extend(['totalVisibilityId'])
    else:
        print "warning: info doesn't have the key 'totalVisibilityId'"
    marker = 9999999
    datachunk = [0 for i in range(len(binaryinfokeysnew)+1)]
    count = 0
    datachunk[count] = np.array([marker])         #start with a marker
    count += 1
    if verbose:
                print "appending",
    for key in binaryinfokeysnew:
        if key in ['antloc', 'ubl','degenM', 'AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB','totalVisibilityId']:  #'antloc',
            add = np.append(np.array(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
        elif key == 'ublindex':
            add = np.append(np.vstack(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
        elif key in ['A','B']:
            if info['nAntenna'] > threshold:
                row = info[key].nonzero()[0]           #row index of non zero entries
                column = info[key].nonzero()[1]        #column index of non zero entries
                nonzero = np.transpose(np.array([row,column]))       #a list of non zero entries
                temp = np.array([np.array([row[i],column[i],info[key][nonzero[i,0],nonzero[i,1]]]) for i in range(len(nonzero))])
                add = np.append(np.array(temp.flatten()),[marker])
                datachunk[count] = add
            else:
                add = np.append(np.array(info[key].todense().flatten()).flatten(),[marker])
                datachunk[count] = add
            count += 1
            if verbose:
                print key,
        else:
            add = np.append(np.array(info[key]).flatten(),[marker])
            datachunk[count] = add
            count += 1
            if verbose:
                print key,
    print ""
    datachunkarray = array('d',np.concatenate(tuple(datachunk)))
    outfile=open(infopath,'wb')
    datachunkarray.tofile(outfile)
    outfile.close()
    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:", "Info file successfully written to %s. Time taken: %f minutes."%(infopath, (time.time()-timer)/60.)
    return

def read_redundantinfo_old(infopath, verbose = False):
    '''XXX DOCSTRING'''
    METHODNAME = "read_redundantinfo"
    timer = time.time()
    if not os.path.isfile(infopath):
        raise Exception('Error: file path %s does not exist!'%infopath)
    with open(infopath) as f:
        farray=array('d')
        farray.fromstring(f.read())
        datachunk = np.array(farray)
        marker = 9999999
        markerindex=np.where(datachunk == marker)[0]
        rawinfo=np.array([np.array(datachunk[markerindex[i]+1:markerindex[i+1]]) for i in range(len(markerindex)-1)])

    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:",  "Reading redundant info...",

    info = {}
    infocount = 0;
    info['nAntenna'] = int(rawinfo[infocount][0]) #number of good antennas among all (64) antennas, same as the length of subsetant
    infocount += 1
    info['nUBL'] = int(rawinfo[infocount][0]) #number of unique baselines
    infocount += 1
    nbl = int(rawinfo[infocount][0])
    info['nBaseline'] = nbl
    infocount += 1
    info['subsetant'] = rawinfo[infocount].astype(int) #the index of good antennas in all (64) antennas
    infocount += 1
    info['antloc'] = rawinfo[infocount].reshape((info['nAntenna'],3)) #the index of good antennas in all (64) antennas
    infocount += 1
    info['subsetbl'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['ubl'] = rawinfo[infocount].reshape((info['nUBL'],3)) #unique baseline vectors
    infocount += 1
    info['bltoubl'] = rawinfo[infocount].astype(int) #cross bl number to ubl index
    infocount += 1
    info['reversed'] = rawinfo[infocount].astype(int) #cross only bl if reversed -1, otherwise 1
    infocount += 1
    info['reversedauto'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['autoindex'] = rawinfo[infocount].astype(int)  #index of auto bls among good bls
    infocount += 1
    info['crossindex'] = rawinfo[infocount].astype(int)  #index of cross bls among good bls
    infocount += 1
    ncross = len(info['crossindex'])
    info['ncross'] = ncross
    info['bl2d'] = rawinfo[infocount].reshape(nbl, 2).astype(int) #from 1d bl index to a pair of antenna numbers
    infocount += 1
    info['ublcount'] = rawinfo[infocount].astype(int) #for each ubl, the number of good cross bls corresponding to it
    infocount += 1
    info['ublindex'] = range((info['nUBL'])) #//for each ubl, the vector<int> contains (ant1, ant2, crossbl)
    tmp = rawinfo[infocount].reshape(ncross, 3).astype(int)
    infocount += 1
    cnter = 0
    for i in range(info['nUBL']):
        info['ublindex'][i] = np.zeros((info['ublcount'][i],3))
        for j in range(len(info['ublindex'][i])):
            info['ublindex'][i][j] = tmp[cnter]
            cnter+=1
    info['ublindex'] = np.asarray(info['ublindex'])

    info['bl1dmatrix'] = rawinfo[infocount].reshape((info['nAntenna'], info['nAntenna'])).astype(int) #a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
    infocount += 1
    #matrices
    info['degenM'] = rawinfo[infocount].reshape((info['nAntenna'] + info['nUBL'], info['nAntenna']))
    infocount += 1
    info['A'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #A matrix for logcal amplitude
    infocount += 1
    info['B'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #B matrix for logcal phase
    infocount += 1
    ##The sparse matrices are treated a little differently because they are not rectangular
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        info['At'] = info['A'].transpose()
        info['Bt'] = info['B'].transpose()
        info['AtAi'] = la.pinv(info['At'].dot(info['A']).todense(), cond = 10**(-6))#(AtA)^-1
        info['BtBi'] = la.pinv(info['Bt'].dot(info['B']).todense(), cond = 10**(-6))#(BtB)^-1
        #info['AtAiAt'] = info['AtAi'].dot(info['At'].todense())#(AtA)^-1At
        #info['BtBiBt'] = info['BtBi'].dot(info['Bt'].todense())#(BtB)^-1Bt
        #info['PA'] = info['A'].dot(info['AtAiAt'])#A(AtA)^-1At
        #info['PB'] = info['B'].dot(info['BtBiBt'])#B(BtB)^-1Bt
        #info['ImPA'] = sps.identity(ncross) - info['PA']#I-PA
        #info['ImPB'] = sps.identity(ncross) - info['PB']#I-PB
    if verbose:
        print "done. nAntenna, nUBL, nBaseline = %i, %i, %i. Time taken: %f minutes."%(len(info['subsetant']), info['nUBL'], info['nBaseline'], (time.time()-timer)/60.)
    return info


# XXX all these different read/writes should be subclasses of Info
def read_redundantinfo(infopath, verbose = False, DoF_only = False):
    '''XXX DOCSTRING'''
    METHODNAME = "read_redundantinfo"
    timer = time.time()
    infopath = os.path.expanduser(infopath)
    if not os.path.isfile(infopath):
        raise IOError('Error: file path %s does not exist!'%infopath)
    with open(infopath) as f:
        farray = array('d')
        farray.fromstring(f.read())
        datachunk = np.array(farray)
        marker = 9999999
        markerindex = np.where(datachunk == marker)[0]
        rawinfo = np.array([np.array(datachunk[markerindex[i]+1:markerindex[i+1]]) for i in range(len(markerindex)-1)])
    if verbose:
        print FILENAME + "*" + METHODNAME + " MSG:",  "Reading redundant info...",

    info = {}
    infocount = 0;
    info['nAntenna'] = int(rawinfo[infocount][0]) #number of good antennas among all (64) antennas, same as the length of subsetant
    infocount += 1
    info['nUBL'] = int(rawinfo[infocount][0]) #number of unique baselines
    infocount += 1
    nbl = int(rawinfo[infocount][0])
    info['nBaseline'] = nbl
    infocount += 1
    info['subsetant'] = rawinfo[infocount].astype(int) #the index of good antennas in all (64) antennas
    infocount += 1
    info['antloc'] = rawinfo[infocount].reshape((info['nAntenna'],3)) #the index of good antennas in all (64) antennas
    infocount += 1
    info['subsetbl'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['ubl'] = rawinfo[infocount].reshape((info['nUBL'],3)) #unique baseline vectors
    infocount += 1
    info['bltoubl'] = rawinfo[infocount].astype(int) #cross bl number to ubl index
    infocount += 1
    info['reversed'] = rawinfo[infocount].astype(int) #cross only bl if reversed -1, otherwise 1
    infocount += 1
    info['reversedauto'] = rawinfo[infocount].astype(int) #the index of good baselines (auto included) in all baselines
    infocount += 1
    info['autoindex'] = rawinfo[infocount].astype(int)  #index of auto bls among good bls
    infocount += 1
    info['crossindex'] = rawinfo[infocount].astype(int)  #index of cross bls among good bls
    infocount += 1
    ncross = len(info['crossindex'])
    info['ncross'] = ncross
    if DoF_only:
        return ncross - info['nUBL'] - info['nAntenna'] + 2

    info['bl2d'] = rawinfo[infocount].reshape(nbl, 2).astype(int) #from 1d bl index to a pair of antenna numbers
    infocount += 1
    info['ublcount'] = rawinfo[infocount].astype(int) #for each ubl, the number of good cross bls corresponding to it
    infocount += 1
    info['ublindex'] = range((info['nUBL'])) #//for each ubl, the vector<int> contains (ant1, ant2, crossbl)
    tmp = rawinfo[infocount].reshape(ncross, 3).astype(int)
    infocount += 1
    cnter = 0
    for i in range(info['nUBL']):
        info['ublindex'][i] = np.zeros((info['ublcount'][i],3))
        for j in range(len(info['ublindex'][i])):
            info['ublindex'][i][j] = tmp[cnter]
            cnter+=1
    info['ublindex'] = np.asarray(info['ublindex'])

    info['bl1dmatrix'] = rawinfo[infocount].reshape((info['nAntenna'], info['nAntenna'])).astype(int) #a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
    infocount += 1
    #matrices
    info['degenM'] = rawinfo[infocount].reshape((info['nAntenna'] + info['nUBL'], info['nAntenna']))
    infocount += 1
    threshold = 128
    if info['nAntenna'] > threshold:
        sparse_entries = rawinfo[infocount].reshape((len(rawinfo[infocount])/3,3))
        row = sparse_entries[:,0]
        column = sparse_entries[:,1]
        value = sparse_entries[:,2]
        info['A'] = sps.csr_matrix((value,(row,column)),shape=(ncross, info['nAntenna'] + info['nUBL']))
    else:
        info['A'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #A matrix for logcal amplitude
    infocount += 1
    if info['nAntenna'] > threshold:
        sparse_entries = rawinfo[infocount].reshape((len(rawinfo[infocount])/3,3))
        row = sparse_entries[:,0]
        column = sparse_entries[:,1]
        value = sparse_entries[:,2]
        info['B'] = sps.csr_matrix((value,(row,column)),shape=(ncross, info['nAntenna'] + info['nUBL']))
    else:
        info['B'] = sps.csr_matrix(rawinfo[infocount].reshape((ncross, info['nAntenna'] + info['nUBL'])).astype(int)) #B matrix for logcal phase
    infocount += 1
    if info['nAntenna'] > threshold:
        info['AtAi'] = rawinfo[infocount].reshape((info['nAntenna'] + info['nUBL'],info['nAntenna'] + info['nUBL']))
        infocount += 1
        info['BtBi'] = rawinfo[infocount].reshape((info['nAntenna'] + info['nUBL'],info['nAntenna'] + info['nUBL']))
        infocount += 1
    if len(rawinfo) > infocount:     #make sure the code is compatible the old files (saved without totalVisibilityId)
        info['totalVisibilityId'] = rawinfo[infocount].reshape(-1,2).astype(int)

    ##The sparse matrices are treated a little differently because they are not rectangular
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        info['At'] = info['A'].transpose()
        info['Bt'] = info['B'].transpose()
        if info['nAntenna'] <= threshold:
            info['AtAi'] = la.pinv(info['At'].dot(info['A']).todense(), cond = 10**(-6))#(AtA)^-1
            info['BtBi'] = la.pinv(info['Bt'].dot(info['B']).todense(), cond = 10**(-6))#(BtB)^-1
        #info['AtAiAt'] = info['AtAi'].dot(info['At'].todense())#(AtA)^-1At
        #info['BtBiBt'] = info['BtBi'].dot(info['Bt'].todense())#(BtB)^-1Bt
        #info['PA'] = info['A'].dot(info['AtAiAt'])#A(AtA)^-1At
        #info['PB'] = info['B'].dot(info['BtBiBt'])#B(BtB)^-1Bt
        #info['ImPA'] = sps.identity(ncross) - info['PA']#I-PA
        #info['ImPB'] = sps.identity(ncross) - info['PB']#I-PB
    if verbose:
        print "done. nAntenna, nUBL, nBaseline = %i, %i, %i. Time taken: %f minutes."%(len(info['subsetant']), info['nUBL'], info['nBaseline'], (time.time()-timer)/60.)
    return info

def get_xy_AB(info):
    '''return xyA, xyB, yxA, yxB for logcal cross polarizations'''
    na = info['nAntenna']
    nu = info['nUBL']
    A = info['A'].todense()
    B = info['B'].todense()
    bl2dcross = info['bl2d'][info['crossindex']]
    #print na, nu, B.shape wesdcxaz

    xyA = np.zeros((len(info['crossindex']), 2*na+nu), dtype='int8')
    yxA = np.zeros_like(xyA)
    xyB = np.zeros_like(xyA)
    yxB = np.zeros_like(xyA)
    xyA[:, 2*na:] = A[:, na:]
    xyB[:, 2*na:] = B[:, na:]
    for i in range(len(xyA)):
        xyA[i, bl2dcross[i,0]] = A[i, bl2dcross[i,0]]
        xyA[i, na + bl2dcross[i,1]] = A[i, bl2dcross[i,1]]
        xyB[i, bl2dcross[i,0]] = B[i, bl2dcross[i,0]]
        xyB[i, na + bl2dcross[i,1]] = B[i, bl2dcross[i,1]]
    yxA[:, :na] = xyA[:, na:2*na]
    yxA[:, na:2*na] = xyA[:, :na]
    yxA[:, 2*na:] = xyA[:, 2*na:]
    yxB[:, :na] = xyB[:, na:2*na]
    yxB[:, na:2*na] = xyB[:, :na]
    yxB[:, 2*na:] = xyB[:, 2*na:]

    return xyA, xyB, yxA, yxB

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

def apply_calpar(data, calpar, visibilityID):
    '''apply complex calpar for all antennas onto all baselines, calpar's dimension will be assumed to mean: 1D: constant over time and freq; 2D: constant over time; 3D: change over time and freq'''
    METHODNAME = "*apply_calpar*"
    if calpar.shape[-1] <= np.amax(visibilityID) or data.shape[-1] != len(visibilityID):
        raise Exception("Dimension mismatch! Either number of antennas in calpar " + str(calpar.shape[-1]) + " is less than implied in visibility ID "  + str(1 + np.amax(visibilityID)) + ", or the length of the last axis of data "  + str(data.shape[-1]) + " is not equal to length of visibilityID "  + str(len(visibilityID)) + ".")
    if len(calpar.shape) == 3 and len(data.shape) == 3 and calpar.shape[:2] == data.shape[:2]:
        return data/(np.conjugate(calpar[:,:,visibilityID[:,0].astype(int)]) * calpar[:,:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 2 and (len(data.shape) == 3 or len(data.shape) == 2) and calpar.shape[0] == data.shape[-2]:
        return data/(np.conjugate(calpar[:,visibilityID[:,0].astype(int)]) * calpar[:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 1 and len(data.shape) <= 3:
        return data/(np.conjugate(calpar[visibilityID[:,0].astype(int)]) * calpar[visibilityID[:,1].astype(int)])
    else:
        raise Exception("Dimension mismatch! I don't know how to interpret data dimension of " + str(data.shape) + " and calpar dimension of " + str(calpar.shape) + ".")

def apply_calpar2(data, calpar, calpar2, visibilityID):
    '''apply complex calpar for all antennas onto all baselines, calpar's dimension will be assumed to mean: 1D: constant over time and freq; 2D: constant over time; 3D: change over time and freq'''
    METHODNAME = "*apply_calpar2*"
    if calpar.shape[-1] <= np.amax(visibilityID) or data.shape[-1] != len(visibilityID) or calpar.shape != calpar2.shape:
        raise Exception("Dimension mismatch! Either number of antennas in calpar " + str(calpar.shape[-1]) + " is less than implied in visibility ID "  + str(1 + np.amax(visibilityID)) + ", or the length of the last axis of data "  + str(data.shape[-1]) + " is not equal to length of visibilityID "  + str(len(visibilityID)) + ", or calpars have different dimensions:" + str(calpar.shape) + str(calpar.shape) + '.')
    if len(calpar.shape) == 3 and len(data.shape) == 3 and calpar.shape[:2] == data.shape[:2]:
        return data/(np.conjugate(calpar[:,:,visibilityID[:,0].astype(int)]) * calpar2[:,:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 2 and (len(data.shape) == 3 or len(data.shape) == 2) and calpar.shape[0] == data.shape[-2]:
        return data/(np.conjugate(calpar[:,visibilityID[:,0].astype(int)]) * calpar2[:,visibilityID[:,1].astype(int)])
    elif len(calpar.shape) == 1 and len(data.shape) <= 3:
        return data/(np.conjugate(calpar[visibilityID[:,0].astype(int)]) * calpar2[visibilityID[:,1].astype(int)])
    else:
        raise Exception("Dimension mismatch! I don't know how to interpret data dimension of " + str(data.shape) + " and calpar dimension of " + str(calpar.shape) + ".")

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


# XXX utility function, should be separate file
def stdmatrix(length, polydegree):
    '''to find out the error in fitting y by a polynomial poly(x), one compute error vector by (I-A.(At.A)^-1 At).y, where Aij = i^j. This function returns (I-A.(At.A)^-1 At)'''
    A = np.array([[i**j for j in range(polydegree + 1)] for i in range(length)], dtype='int')
    At = A.transpose()
    return np.identity(length) - A.dot(la.pinv(At.dot(A), cond = 10**(-6)).dot(At))

# XXX part of Info class?
def compare_info(info1,info2, verbose=True, tolerance = 10**(-5)):
    '''input two different redundant info, output True if they are the same and False if they are different'''
    try:
        floatkeys=float_infokeys#['antloc','ubl','AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']
        intkeys = ['nAntenna','nUBL','nBaseline','subsetant','subsetbl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','bl1dmatrix']
        infomatrices=['A','B','At','Bt']
        specialkeys = ['ublindex']
        allkeys= floatkeys + intkeys + infomatrices + specialkeys#['antloc','ubl','nAntenna','nUBL','nBaseline','subsetant','subsetbl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','bl1dmatrix','AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB','A','B','At','Bt']
        diff=[]
        #10**5 for floating point errors
        for key in floatkeys:
            try:
                diff.append(round(la.norm(np.array(info1[key])-np.array(info2[key]))/tolerance)==0)
            except:
                diff.append(False)
        for key in intkeys:
            try:
                diff.append(la.norm(np.array(info1[key])-np.array(info2[key]))==0)
            except:
                diff.append(False)
        for key in infomatrices:
            try:
                diff.append(la.norm((info1[key]-info2[key]).todense())==0)
            except:
                diff.append(False)

        diff.append(True)
        try:
            for i,j in zip(info1['ublindex'],info2['ublindex']):
                diff[-1] = diff[-1] and (la.norm(np.array(i) - np.array(j))==0)
        except:
            diff[-1] = False
        bool = True
        for i in diff:
            bool = bool and i
        #print the first key found different (this will only trigger when the two info's have the same shape, so probably not very useful)
        if verbose and bool == False:
            for i in range(len(diff)):
                if diff[i] == False:
                    print allkeys[i]
        return bool
    except ValueError:
        print "info doesn't have the same shape"
        return False

def omnical2omnigain(omnicalPath, utctimePath, info, outputPath = None):
    '''outputPath should be a path without extensions like .omnigain which will be appended'''
    if outputPath is None:
        outputPath = omnicalPath.replace('.omnical', '')

    #info = redundantCalibrator.info

    if not os.path.isfile(utctimePath):
        raise Exception("File %s does not exist!"%utctimePath)
    with open(utctimePath) as f:
        utctimes = f.readlines()
    calpars = np.fromfile(omnicalPath, dtype='float32')

    nT = len(utctimes)
    nF = len(calpars) / nT / (3 + 2 * info['nAntenna'] + 2 * info['nUBL'])
    #if nF != redundantCalibrator.nFrequency:
        #raise Exception('Error: time and frequency count implied in the infput files (%d %d) does not agree with those speficied in redundantCalibrator (%d %d). Exiting!'%(nT, nF, redundantCalibrator.nTime, redundantCalibrator.nFrequency))
    calpars = calpars.reshape((nT, nF, (3 + 2 * info['nAntenna'] + 2 * info['nUBL'])))

    jd = np.zeros((len(utctimes), 2), dtype='float32')#Julian dat is the only double in this whole thing so im storing it in two chunks as float
    sa = ephem.Observer()
    for utctime, t in zip(utctimes, range(len(utctimes))):
        sa.date = utctime
        jd[t, :] = struct.unpack('ff', struct.pack('d', sa.date + julDelta))

    opchisq = np.zeros((nT, 2 + 1 + 2 * nF), dtype = 'float32')
    opomnigain = np.zeros((nT, info['nAntenna'], 2 + 1 + 1 + 2 * nF), dtype = 'float32')
    opomnifit = np.zeros((nT, info['nUBL'], 2 + 3 + 1 + 2 * nF), dtype = 'float32')

    opchisq[:, :2] = jd
    opchisq[:, 2] = float(nF)
    #opchisq[:, 3::2] = calpars[:, :, 0]#number of lincal iters
    opchisq[:, 3:] = calpars[:, :, 2]#chisq which is sum of squares of errors in each visbility

    opomnigain[:, :, :2] = jd[:, None]
    opomnigain[:, :, 2] = np.array(info['subsetant']).astype('float32')
    opomnigain[:, :, 3] = float(nF)
    gains = (10**calpars[:, :, 3:(3 + info['nAntenna'])] * np.exp(1.j * math.pi * calpars[:, :, (3 + info['nAntenna']):(3 + 2 * info['nAntenna'])] / 180)).transpose((0,2,1))
    opomnigain[:, :, 4::2] = np.real(gains)
    opomnigain[:, :, 5::2] = np.imag(gains)

    opomnifit[:, :, :2] = jd[:, None]
    opomnifit[:, :, 2:5] = np.array(info['ubl']).astype('float32')
    opomnifit[:, :, 5] = float(nF)
    opomnifit[:, :, 6::2] = calpars[:, :, 3 + 2 * info['nAntenna']::2].transpose((0,2,1))
    opomnifit[:, :, 7::2] = calpars[:, :, 3 + 2 * info['nAntenna'] + 1::2].transpose((0,2,1))


    opchisq.tofile(outputPath + '.omnichisq')
    opomnigain.tofile(outputPath + '.omnigain')
    opomnifit.tofile(outputPath + '.omnifit')

#  ___        _              _          _   ___       __     
# | _ \___ __| |_  _ _ _  __| |__ _ _ _| |_|_ _|_ _  / _|___ 
# |   / -_) _` | || | ' \/ _` / _` | ' \  _|| || ' \|  _/ _ \
# |_|_\___\__,_|\_,_|_||_\__,_\__,_|_||_\__|___|_||_|_| \___/

class RedundantInfo(_O.RedundantInfo):
    '''a class that contains redundant calibration information that should only be passed into C++'''
    def __init__(self, info, verbose=False):
        _O.RedundantInfo.__init__(self)
        if type(info) == type('a'):
            info = read_redundantinfo(info)
        elif type(info) != type({}):
            raise Exception("Error: info argument not recognized. It must be of either dictionary type (an info dictionary) *OR* string type (path to the info file).")
        if verbose:
            print "Converting info:",
            sys.stdout.flush()
        for key in info.keys():
            if verbose:
                print key,
                sys.stdout.flush()
            try:
                if key in ['At','Bt']:
                    tmp = []
                    nonzeros = np.array(info[key].nonzero()).transpose()
                    for i,j in nonzeros:
                        tmp += [[i, j, info[key][i,j]]]
                    #for i in range(info[key].shape[0]):
                        #for j in range(info[key].shape[1]):
                            #if info[key][i,j] != 0:
                                #tmp += [[i, j, info[key][i,j]]]
                    self.__setattr__(key+'sparse', np.array(tmp, dtype = 'int32'))
                elif key in ['A','B']:
                    self.__setattr__(key, info[key].todense().astype('int32'))
                elif key in ['ublindex']:
                    #tmp = []
                    #for i in range(len(info[key])):
                    #    for j in range(len(info[key][i])):
                    #        tmp += [[i, j, info[key][i][j][0], info[key][i][j][1], info[key][i][j][2]]]
                    #self.__setattr__(key, np.array(tmp, dtype='int32'))
                    self.__setattr__(key, np.concatenate(info[key]).astype(np.int32))
                elif key in int_infokeys:
                    self.__setattr__(key, int(info[key]))
                elif key in intarray_infokeys and key != 'ublindex':
                    self.__setattr__(key, np.array(info[key]).astype('int32'))
                elif key in intarray_infokeys_optional:
                    try:
                        self.__setattr__(key, np.array(info[key]).astype('int32'))
                    except KeyError:
                        pass
                elif key in float_infokeys:
                    self.__setattr__(key, np.array(info[key]).astype('float32'))
            except:
                raise Exception("Error parsing %s item."%key)
        if verbose:
            print "Done."
            sys.stdout.flush()
    def __getitem__(self,k): return self.__getattribute__(k)
    def __getattribute__(self, key):
        try:
            if key in ['A','B']:
                #print key
                return sps.csr_matrix(_O.RedundantInfo.__getattribute__(self, key))
            elif key in ['At','Bt']:
                tmp = _O.RedundantInfo.__getattribute__(self, key+'sparse')
                matrix = np.zeros((self.nAntenna + self.nUBL, len(self.crossindex)))
                for i in tmp:
                    matrix[i[0],i[1]] = i[2]
                return sps.csr_matrix(matrix)
            elif key in ['ublindex']:
                ublindex = []
                for i in _O.RedundantInfo.__getattribute__(self, key):
                    while len(ublindex) < i[0] + 1:
                        ublindex.append(np.zeros((1,3)))
                    while len(ublindex[i[0]]) < i[1] + 1:
                        ublindex[i[0]] = np.array(ublindex[i[0]].tolist() + [[0,0,0]])
                    ublindex[i[0]][i[1]] = np.array(i[2:])
                return ublindex

            else:
                return _O.RedundantInfo.__getattribute__(self, key)
        except:
            raise Exception("Error retrieving %s item."%key)


    def get_info(self):
        info = {}
        for key in infokeys:
            try:
                info[key] = self.__getattribute__(key)
            except:
                raise Exception("Error retrieving %s item."%key)
        for key in infokeys_optional:
            try:
                info[key] = self.__getattribute__(key)
            except:
                pass
        return info

def _redcal(data, rawCalpar, Info, additivein, additive_out, removedegen=0, uselogcal=1, maxiter=50, conv=1e-3, stepsize=.3, computeUBLFit = 1, trust_period = 1):
    '''same as _O.redcal, but does not return additiveout. Rather it puts additiveout into an inputted container'''

    np_rawCalpar = np.frombuffer(rawCalpar, dtype='float32')
    np_rawCalpar.shape=(data.shape[0], data.shape[1], len(rawCalpar) / data.shape[0] / data.shape[1])
    #print np_rawCalpar.dtype, np_rawCalpar.shape

    np_additive_out = np.frombuffer(additive_out, dtype='complex64')
    np_additive_out.shape = data.shape
    _O.redcal2(data, np_rawCalpar, Info, additivein, np_additive_out, removedegen=removedegen, uselogcal=uselogcal, maxiter=int(maxiter), conv=float(conv), stepsize=float(stepsize), computeUBLFit = int(computeUBLFit), trust_period = int(trust_period))

    #np_additive_out = _O.redcal(data, np_rawCalpar, Info, additivein, removedegen=removedegen, uselogcal=uselogcal, maxiter=int(maxiter), conv=float(conv), stepsize=float(stepsize), computeUBLFit = int(computeUBLFit), trust_period = int(trust_period))
    #additive_out[:len(additive_out)/2] = np.real(np_additive_out.flatten())
    #additive_out[len(additive_out)/2:] = np.imag(np_additive_out.flatten())


#  ___        _              _          _    ___      _ _ _             _           
# | _ \___ __| |_  _ _ _  __| |__ _ _ _| |_ / __|__ _| (_) |__ _ _ __ _| |_ ___ _ _ 
# |   / -_) _` | || | ' \/ _` / _` | ' \  _| (__/ _` | | | '_ \ '_/ _` |  _/ _ \ '_|
# |_|_\___\__,_|\_,_|_||_\__,_\__,_|_||_\__|\___\__,_|_|_|_.__/_| \__,_|\__\___/_|  

class RedundantCalibrator:
    '''This class is the main tool for performing redundant calibration on data sets. 
    For a given redundant configuration, say 32 antennas with 3 bad antennas, the 
    user should create one instance of Redundant calibrator and reuse it for all data 
    collected from that array. In general, upon creating an instance, the user need 
    to create the info field of the instance by either computing it or reading it 
    from a text file. readyForCpp(verbose = True) should be a very helpful function 
    to provide information on what information is missing for running the calibration.'''
    def __init__(self, nTotalAnt, info = None):
        methodName = '.__init__.'
        self.className = '.RedundantCalibrator.'
        self.nTotalAnt = nTotalAnt
        self.nTotalBaselineAuto = (self.nTotalAnt + 1) * self.nTotalAnt / 2
        self.nTotalBaselineCross = (self.nTotalAnt - 1) * self.nTotalAnt / 2
        self.antennaLocation = np.zeros((self.nTotalAnt, 3))
        side = int(nTotalAnt**.5)
        for a in range(nTotalAnt):
            self.antennaLocation[a] = np.array([a/side, a%side, 0])
        self.antennaLocationTolerance = 10**(-6)
        self.badAntenna = []
        self.badUBL = []
        self.badUBLpair = []
        self.ubl2goodubl = None
        self.totalVisibilityId = np.concatenate([[[i,j] for i in range(j + 1)] for j in range(self.nTotalAnt)])#PAPER miriad convention by default
        self.totalVisibilityId_dic = None
        self.totalVisibilityUBL = None
        self.Info = None
        self.removeDegeneracy = True
        self.removeAdditive = False
        self.removeAdditivePeriod = -1
        self.convergePercent = 0.01 #convergence criterion in relative change of chi^2. By default it stops when reaches 0.01, namely 1% decrease in chi^2.
        self.maxIteration = 50 #max number of iterations in lincal
        self.stepSize = 0.3 #step size for lincal. (0, 1]. < 0.4 recommended.
        self.computeUBLFit = True
        self.trust_period = 1 #How many time slices does lincal start from logcal result rather than the previous time slice's lincal result. default 1 means always start from logcal. if 10, it means lincal start from logcal results (or g = 1's) every 10 time slices

        self.nTime = 0
        self.nFrequency = 0

        self.utctime = None
        self.rawCalpar = None
        self.omnichisq = None
        self.omnigain = None
        self.omnifit = None

        if info is not None:
            if type(info) == type({}):

                self.Info = RedundantInfo(info)
            elif type(info) == type('a'):
                self.read_redundantinfo(info)
            else:
                raise Exception(self.className + methodName + "Error: info argument not recognized. It must be of either dictionary type (an info dictionary) *OR* string type (path to the info file).")

    def __repr__(self,):
        return self.__str__()

    def __str__(self,):
        if self.Info is None:
            return "<Uninitialized %i antenna RedundantCalibrator with no RedundantInfo.>"%self.nTotalAnt
        else:
            return "<RedundantCalibrator for an %i antenna array: %i good baselines including %i good antennas and %i unique baselines.>"%(self.nTotalAnt, len(self.Info.crossindex), self.Info.nAntenna, self.Info.nUBL)

    def __getattr__(self, name):
        try:
            return self.Info.__getattribute__(name)
        except:
            raise AttributeError("RedundantCalibrator has no attribute named %s"%name)

    def read_redundantinfo(self, infopath, verbose = False):
        '''redundantinfo is necessary for running redundant calibration. The text file 
        should contain 29 lines each describes one item in the info.'''
        info = read_redundantinfo(infopath, verbose = verbose)
        try:
            self.totalVisibilityId = info['totalVisibilityId']
        except KeyError:
            info['totalVisibilityId'] = self.totalVisibilityId
        self.Info = RedundantInfo(info, verbose = verbose)

    def write_redundantinfo(self, infoPath, overwrite = False, verbose = False):
        methodName = '.write_redundantinfo.'
        write_redundantinfo(self.Info.get_info(), infoPath, overwrite = overwrite, verbose = verbose)

    def read_arrayinfo(self, arrayinfopath, verbose = False):
        '''array info is the minimum set of information to uniquely describe a 
        redundant array, and is needed to compute redundant info. It includes, 
        in each line, bad antenna indices, bad unique baseline indices, tolerance 
        of error when checking redundancy, antenna locations, and visibility's 
        antenna pairing conventions. Unlike redundant info which is a self-contained 
        dictionary, items in array info each have their own fields in the instance.'''
        methodName = ".read_arrayinfo."
        if not os.path.isfile(arrayinfopath):
            raise IOError(self.className + methodName + "Error: Array info file " + arrayinfopath + " doesn't exist!")
        with open(arrayinfopath) as f:
            rawinfo = [[float(x) for x in line.split()] for line in f]
        if verbose:
            print self.className + methodName + " MSG:",  "Reading", arrayinfopath, "...",

        self.badAntenna = np.array(rawinfo[0]).astype(int)
        if self.badAntenna[0] < 0:
            self.badAntenna = np.zeros(0)

        if len(np.array(rawinfo[1])) == 0 or len(np.array(rawinfo[1]))%2 != 0 or min(np.array(rawinfo[1]).astype(int)) < 0:
            self.badUBLpair = np.array([])
            #raise Exception(self.className + methodName +"Error: Format error in " + arrayinfopath + "badUBL should be specified by pairs of antenna, not odd numbers of antenna")
        else:
            rawpair = np.array(rawinfo[1]).astype(int)
            self.badUBLpair = np.reshape(rawpair,(len(rawpair)/2,2))
        #if self.badUBL[0] < 0:#todonow
        #    self.badUBL = np.zeros(0)

        self.antennaLocationTolerance = rawinfo[2][0]

        for a in range(len(self.antennaLocation)):
            if len(rawinfo[a + 3]) != 3:
                raise ValueError(self.className + methodName + "Error: Format error in " + arrayinfopath + ": The antenna locations should start on the 4th line, with 3 numbers in each line!")
            else:
                self.antennaLocation[a] = np.array(rawinfo[a + 3])

        #for bl in range(len(self.totalVisibilityId)):
            #if len(rawinfo[bl + 3 + len(self.antennaLocation)]) != 2:
                #raise Exception(self.className + methodName + "Error: Format error in " + arrayinfopath + ": The baseline to antenna mapping should start after antenna locations, with 2 numbers (conj index, index) in each line!")
            #else:
        bl = 0
        self.totalVisibilityId = []
        max_bl_cnt = self.nTotalAnt * (self.nTotalAnt + 1) / 2
        maxline = len(rawinfo)
        while len(rawinfo[bl + 3 + len(self.antennaLocation)]) == 2:
            if bl >= max_bl_cnt:
                raise Exception("Number of total visibility ids exceeds the maximum possible number of baselines of %i"%(max_bl_cnt))
            self.totalVisibilityId.append(np.array(rawinfo[bl + 3 + len(self.antennaLocation)]).astype(int))
            bl = bl + 1
            if bl + 3 + len(self.antennaLocation) >= maxline:
                break
        self.totalVisibilityId = np.array(self.totalVisibilityId).astype(int)
        if verbose:
            print "Total number of visibilities:", bl,
            print "Bad antenna indices:", self.badAntenna,
            print "Bad UBL indices:", self.badUBLpair


    def lincal(self, data, additivein, nthread = None, verbose = False):
        '''for best performance, try setting nthread to larger than number of cores.'''
        if data.ndim != 3 or data.shape[-1] != len(self.totalVisibilityId):
            raise ValueError("Data shape error: it must be a 3D numpy array of dimensions time * frequency * baseline(%i)"%len(self.totalVisibilityId))
        if data.shape != additivein.shape:
            raise ValueError("Data shape error: data and additive in have different shapes.")
        self.nTime = len(data)
        self.nFrequency = len(data[0])
        if self.rawCalpar is None:
            self.rawCalpar = np.zeros((self.nTime, self.nFrequency, 3 + 2 * (self.Info.nAntenna + self.Info.nUBL)), dtype = 'float32')
        elif self.rawCalpar.shape != (len(data), len(data[0]), 3 + 2 * (self.Info.nAntenna + self.Info.nUBL)):
            raise ValueError("ERROR: lincal called without a properly shaped self.rawCalpar! Excpeted shape is (%i, %i, %i)!"%(len(data), len(data[0]), 3 + 2 * (self.Info.nAntenna + self.Info.nUBL)))
        if nthread is None:
            nthread = nthread = min(mp.cpu_count() - 1, self.nFrequency)
        if nthread < 2:
            return _O.redcal(data[:,:,self.Info.subsetbl], self.rawCalpar, self.Info, additivein[:,:,self.Info.subsetbl], removedegen = int(self.removeDegeneracy), uselogcal = 0, maxiter=int(self.maxIteration), conv=float(self.convergePercent), stepsize=float(self.stepSize), computeUBLFit = int(self.computeUBLFit), trust_period = self.trust_period)
        else:
            return self._redcal_multithread(data, additivein, 0, nthread, verbose = verbose)        ##self.chisq = self.rawCalpar[:, :, 2]
        ##self.calpar = np.zeros((len(self.rawCalpar), len(self.rawCalpar[0]), self.nTotalAnt), dtype='complex64')
        ##self.calpar[:,:,self.Info.subsetant] = (10**(self.rawCalpar[:, :, 3: (3 + self.Info.nAntenna)])) * np.exp(1.j * self.rawCalpar[:, :, (3 + self.Info.nAntenna): (3 + 2 * self.Info.nAntenna)])
        ##self.bestfit = self.rawCalpar[:, :, (3 + 2 * self.Info.nAntenna):: 2] + 1.j * self.rawCalpar[:, :, (4 + 2 * self.Info.nAntenna):: 2]

    def logcal(self, data, additivein, nthread = None, verbose = False):
        '''XXX DOCSTRING'''
        if data.ndim != 3 or data.shape[-1] != len(self.totalVisibilityId):
            raise ValueError("Data shape error: it must be a 3D numpy array of dimensions time * frequency * baseline(%i)"%len(self.totalVisibilityId))
        if data.shape != additivein.shape:
            raise ValueError("Data shape error: data and additive in have different shapes.")
        self.nTime = len(data)
        self.nFrequency = len(data[0])
        self.rawCalpar = np.zeros((len(data), len(data[0]), 3 + 2 * (self.Info.nAntenna + self.Info.nUBL)), dtype = 'float32')

        if nthread is None:
            nthread = min(mp.cpu_count() - 1, self.nFrequency)
        if nthread < 2:
            return _O.redcal(data[:,:,self.Info.subsetbl], self.rawCalpar, self.Info, additivein[:,:,self.Info.subsetbl], removedegen = int(self.removeDegeneracy), uselogcal = 1, maxiter=int(self.maxIteration), conv=float(self.convergePercent), stepsize=float(self.stepSize), computeUBLFit = int(self.computeUBLFit))
        else:
            return self._redcal_multithread(data, additivein, 1, nthread, verbose = verbose)

    def _redcal_multithread(self, data, additivein, uselogcal, nthread, verbose = False):
        '''XXX DOCSTRING'''
        #if data.ndim != 3 or data.shape[-1] != len(self.totalVisibilityId):
            #raise ValueError("Data shape error: it must be a 3D numpy array of dimensions time * frequency * baseline(%i)"%len(self.totalVisibilityId))
        #if data.shape != additivein.shape:
            #raise ValueError("Data shape error: data and additive in have different shapes.")
        #self.nTime = len(data)
        #self.nFrequency = len(data[0])
        #self.rawCalpar = np.zeros((len(data), len(data[0]), 3 + 2 * (self.Info.nAntenna + self.Info.nUBL)), dtype = 'float32')
        nthread = min(nthread, self.nFrequency)
        additiveouts = {}
        np_additiveouts = {}
        #additiveout = np.zeros_like(data[:, :, self.Info.subsetbl])
        rawCalpar = {}
        np_rawCalpar = {}
        threads = {}
        fchunk = {}
        chunk = int(self.nFrequency) / int(nthread)
        excess = int(self.nFrequency) % int(nthread)
        kwarg = {"removedegen": int(self.removeDegeneracy), "uselogcal": uselogcal, "maxiter": int(self.maxIteration), "conv": float(self.convergePercent), "stepsize": float(self.stepSize), "computeUBLFit": int(self.computeUBLFit), "trust_period": self.trust_period}

        for i in range(nthread):
            if excess == 0:
                fchunk[i] = (i * chunk, min((1 + i) * chunk, self.nFrequency),)
            elif i < excess:
                fchunk[i] = (i * (chunk+1), min((1 + i) * (chunk+1), self.nFrequency),)
            else:
                fchunk[i] = (fchunk[i-1][1], min(fchunk[i-1][1] + chunk, self.nFrequency),)
            #if verbose:
                #print fchunk[i],
            rawCalpar[i] = mp.RawArray('f', self.nTime * (fchunk[i][1] - fchunk[i][0]) * (self.rawCalpar.shape[2]))
            np_rawCalpar[i] = np.frombuffer(rawCalpar[i], dtype='float32')
            np_rawCalpar[i].shape = (self.rawCalpar.shape[0], fchunk[i][1]-fchunk[i][0], self.rawCalpar.shape[2])
            np_rawCalpar[i][:] = self.rawCalpar[:, fchunk[i][0]:fchunk[i][1]]

            additiveouts[i] = mp.RawArray('f', self.nTime * (fchunk[i][1] - fchunk[i][0]) * len(self.Info.subsetbl) * 2)#factor of 2 for re/im
            np_additiveouts[i] = np.frombuffer(additiveouts[i], dtype='complex64')
            np_additiveouts[i].shape = (data.shape[0], fchunk[i][1]-fchunk[i][0], len(self.Info.subsetbl))

            threads[i] = mp.Process(target = _redcal, args = (data[:, fchunk[i][0]:fchunk[i][1], self.Info.subsetbl], rawCalpar[i], self.Info, additivein[:, fchunk[i][0]:fchunk[i][1], self.Info.subsetbl], additiveouts[i]), kwargs=kwarg)
            #threads[i] = threading.Thread(target = _O.redcal2, args = (data[:, fchunk[i][0]:fchunk[i][1], self.Info.subsetbl], rawCalpar[i], self.Info, additivein[:, fchunk[i][0]:fchunk[i][1], self.Info.subsetbl], additiveouts[i]), kwargs=kwarg)
        if verbose:
            print "Starting %s Process"%cal_name[uselogcal],
            sys.stdout.flush()
        for i in range(nthread):
            if verbose:
                print "#%i"%i,
                sys.stdout.flush()
            threads[i].start()
        if verbose:
            print "Finished Process",
        for i in range(nthread):
            threads[i].join()
            if verbose:
                print "#%i"%i,
        if verbose:
            print ""
            sys.stdout.flush()
        self.rawCalpar = np.concatenate([np_rawCalpar[i] for i in range(nthread)],axis=1)
        return np.concatenate([np_additiveouts[i] for i in range(nthread)],axis=1)

    def get_calibrated_data(self, data, additivein = None):
        '''XXX DOCSTRING'''
        if data.ndim != 3 or data.shape != (self.nTime, self.nFrequency, len(self.totalVisibilityId)):
            raise ValueError("Data shape error: it must be a 3D numpy array of dimensions time * frequency * baseline (%i, %i, %i)"%(self.nTime, self.nFrequency, len(self.totalVisibilityId)))
        if additivein is not None and data.shape != additivein.shape:
            raise ValueError("Data shape error: data and additivein have different shapes.")
        if data.shape[:2] != self.rawCalpar.shape[:2]:
            raise ValueError("Data shape error: data and self.rawCalpar have different first two dimensions.")

        calpar = np.ones((len(self.rawCalpar), len(self.rawCalpar[0]), self.nTotalAnt), dtype='complex64')
        calpar[:,:,self.Info.subsetant] = (10**(self.rawCalpar[:, :, 3: (3 + self.Info.nAntenna)])) * np.exp(1.j * self.rawCalpar[:, :, (3 + self.Info.nAntenna): (3 + 2 * self.Info.nAntenna)])
        if additivein is None:
            return apply_calpar(data, calpar, self.totalVisibilityId)
        else:
            return apply_calpar(data - additivein, calpar, self.totalVisibilityId)

    def get_modeled_data(self):
        '''XXX DOCSTRING'''
        if self.rawCalpar is None:
            raise ValueError("self.rawCalpar doesn't exist. Please calibrate first using logcal() or lincal().")
        if len(self.totalVisibilityId) <= np.max(self.Info.subsetbl):
            raise ValueError("self.totalVisibilityId of length %i is shorter than max index in subsetbl %i. Probably you are using an outdated version of redundantinfo."%(len(self.totalVisibilityId), np.max(self.Info.subsetbl)))
        mdata = np.zeros((self.rawCalpar.shape[0], self.rawCalpar.shape[1], len(self.totalVisibilityId)), dtype='complex64')
        mdata[..., self.Info.subsetbl[self.Info.crossindex]] = (self.rawCalpar[..., 3 + 2 * (self.Info.nAntenna)::2] + 1.j * self.rawCalpar[..., 4 + 2 * (self.Info.nAntenna)::2])[..., self.Info.bltoubl]
        mdata[..., self.Info.subsetbl[self.Info.crossindex]] = np.abs(mdata[..., self.Info.subsetbl[self.Info.crossindex]]) * np.exp(self.Info.reversed * 1.j * np.angle(mdata[..., self.Info.subsetbl[self.Info.crossindex]])) * 10.**(self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,0]] + self.rawCalpar[..., 3 + self.Info.bl2d[self.Info.crossindex,1]]) * np.exp(-1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,0]] + 1.j * self.rawCalpar[..., 3 + self.Info.nAntenna + self.Info.bl2d[self.Info.crossindex,1]])
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
        omnifit = np.zeros((self.nTime, self.Info.nUBL , 2 + 3 + 1 + 2 * self.nFrequency), dtype = 'float32')
        omnifit[:, :, :2] = jd[:, None]
        omnifit[:, :, 2:5] = np.array(self.Info.ubl).astype('float32')
        omnifit[:, :, 5] = float(self.nFrequency)
        omnifit[:, :, 6::2] = self.rawCalpar[:, :, 3 + 2 * self.Info.nAntenna::2].transpose((0,2,1))
        omnifit[:, :, 7::2] = self.rawCalpar[:, :, 3 + 2 * self.Info.nAntenna + 1::2].transpose((0,2,1))
        return omnifit

    def set_badUBL(self, badUBL):
        '''XXX DOCSTRING'''
        if np.array(badUBL).shape[-1] != 3 or len(np.array(badUBL).shape) != 2:
            raise Exception("ERROR: badUBL need to be a list of coordinates each with 3 numbers.")
        badindex = []
        UBL = self.compute_UBL(self.antennaLocationTolerance)
        for badubl in badUBL:
            for i, ubl in zip(range(len(UBL)), UBL):
                if la.norm(badubl - ubl) < self.antennaLocationTolerance:
                    badindex += [i]
                    break
        if len(badindex) != len(badUBL):
            raise Exception("ERROR: some badUBL not found in self.computeUBL!")
        else:
            self.badUBL = np.sort(badindex).tolist()



    def diagnose(self, data = None, additiveout = None, flag = None, verbose = True, healthbar = 2, ubl_healthbar = 50, warn_low_redun = False, ouput_txt = False):
        '''XXX DOCSTRING'''
        errstate = np.geterr()
        np.seterr(invalid = 'ignore')

        if self.rawCalpar is None:
            raise Exception("No calibration has been performed since rawCalpar does not exist.")

        if flag is None:
            flag = np.zeros(self.rawCalpar.shape[:2], dtype='bool')
        elif flag.shape != self.rawCalpar.shape[:2]:
            raise TypeError('flag and self.rawCalpar have different shapes %s %s.'%(flag.shape, self.rawCalpar.shape[:2]))

        checks = 1
        timer = Timer()
        bad_count = np.zeros((3,self.Info.nAntenna), dtype='int')
        bad_ubl_count = np.zeros(self.Info.nUBL, dtype='int')
        median_level = nanmedian(nanmedian(self.rawCalpar[:,:,3:3+self.Info.nAntenna], axis= 0), axis= 1)
        bad_count[0] = np.array([(np.abs(self.rawCalpar[:,:,3+a] - median_level) >= .15)[~flag].sum() for a in range(self.Info.nAntenna)])**2
        #timer.tick(1)


        if data is not None and data.shape[:2] == self.rawCalpar.shape[:2]:
            checks += 1
            subsetbl = self.Info.subsetbl
            crossindex = self.Info.crossindex
            ncross = len(self.Info.crossindex)
            bl1dmatrix = self.Info.bl1dmatrix
            ant_level = np.array([np.median(np.abs(data[:, :, [subsetbl[crossindex[bl]] for bl in bl1dmatrix[a] if (bl < ncross and bl >= 0)]]), axis = -1) for a in range(self.Info.nAntenna)])
            #timer.tick(2)
            median_level = nanmedian(ant_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_count[1] = np.array([(np.abs(ant_level[a] - median_level)/median_level >= .667)[~flag].sum() for a in range(self.Info.nAntenna)])**2
        #timer.tick(2)

        if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2]:
            checks += 1

            subsetbl = self.Info.subsetbl
            crossindex = self.Info.crossindex
            ncross = len(self.Info.crossindex)
            bl1dmatrix = self.Info.bl1dmatrix
            ant_level = np.array([np.median(np.abs(additiveout[:, :, [crossindex[bl] for bl in bl1dmatrix[a] if bl < ncross]]), axis = 2) for a in range(self.Info.nAntenna)])
            #timer.tick(3)
            median_level = np.median(ant_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_count[2] = np.array([(np.abs(ant_level[a] - median_level)/median_level >= .667)[~flag].sum() for a in range(self.Info.nAntenna)])**2
            #timer.tick(3)
            ublindex = [np.array(index).astype('int')[:,2] for index in self.Info.ublindex]
            ubl_level = np.array([np.median(np.abs(additiveout[:, :, [crossindex[bl] for bl in ublindex[u]]]), axis = 2) for u in range(self.Info.nUBL)])
            median_level = np.median(ubl_level, axis = 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=RuntimeWarning)
                bad_ubl_count += np.array([((ubl_level[u] - median_level)/median_level >= .667)[~flag].sum() for u in range(self.Info.nUBL)])**2
            #print median_level
        #timer.tick(3)

        np.seterr(invalid = errstate['invalid'])
        bad_count = (np.mean(bad_count,axis=0)/float(np.sum(~flag))**2 * 100).astype('int')
        bad_ubl_count = (bad_ubl_count/float(self.nTime * self.nFrequency)**2 * 100).astype('int')
        if verbose:
            #print bad_ant_cnt, bad_ubl_cnt
            print "DETECTED BAD ANTENNA ABOVE HEALTH THRESHOLD %i: "%healthbar
            for a in range(len(bad_count)):
                if bad_count[a] > healthbar:
                    print "antenna #%i, vector = %s, badness = %i"%(self.Info.subsetant[a], self.Info.antloc[a], bad_count[a])
            #print ""
            if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2] and ubl_healthbar != 100:
                print "DETECTED BAD BASELINE TYPE ABOVE HEALTH THRESHOLD %i: "%ubl_healthbar
                for a in range(len(bad_ubl_count)):
                    if bad_ubl_count[a] > ubl_healthbar and (self.Info.ublcount[a] > 5 or (warn_low_redun)):
                        print "index #%i, vector = %s, redundancy = %i, badness = %i"%(a, self.Info.ubl[a], self.Info.ublcount[a], bad_ubl_count[a])
                #print ""
        if not ouput_txt:
            return bad_count, bad_ubl_count
        else:
            txt = ''
            txt += "DETECTED BAD ANTENNA ABOVE HEALTH THRESHOLD %i: \n"%healthbar
            for a in range(len(bad_count)):
                if bad_count[a] > healthbar:
                    txt += "antenna #%i, vector = %s, badness = %i\n"%(self.Info.subsetant[a], self.Info.antloc[a], bad_count[a])
            #print ""
            if additiveout is not None and additiveout.shape[:2] == self.rawCalpar.shape[:2] and ubl_healthbar != 100:
                txt += "DETECTED BAD BASELINE TYPE ABOVE HEALTH THRESHOLD %i: \n"%ubl_healthbar
                for a in range(len(bad_ubl_count)):
                    if bad_ubl_count[a] > ubl_healthbar and (self.Info.ublcount[a] > 5 or (warn_low_redun)):
                        txt += "index #%i, vector = %s, redundancy = %i, badness = %i\n"%(a, self.Info.ubl[a], self.Info.ublcount[a], bad_ubl_count[a])
            return txt

    def flag(self, mode = '12', twindow = 5, fwindow = 5, nsigma = 4, _dbg_plotter = None, _niter = 3):
        '''return true if flagged False if good and unflagged'''
        if self.rawCalpar is None or (self.rawCalpar[:,:,2] == 0).all():
            raise Exception("flag cannot be run before lincal.")

        chisq = np.copy(self.rawCalpar[:,:,2])
        nan_flag = np.isnan(np.sum(self.rawCalpar,axis=-1))|np.isinf(np.sum(self.rawCalpar,axis=-1))

        #chisq flag: spike_flag
        spike_flag = np.zeros_like(nan_flag)
        if '1' in mode:
            median_level = nanmedian(nanmedian(chisq))

            thresh = nsigma * (2. / (len(self.subsetbl) - self.nAntenna - self.nUBL + 2))**.5 # relative sigma is sqrt(2/k)

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
        #baseline fit flag
        if '2' in mode:
            nubl = 10
            short_ubl_index = np.argsort(np.linalg.norm(self.ubl, axis=1))[:min(nubl, self.nUBL)]
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
        else:
            ubl_flag = np.zeros_like(nan_flag)

        return_flag = (nan_flag|spike_flag|ubl_flag)
        return return_flag

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

    def compute_redundantinfo(self, arrayinfoPath = None, verbose = False, badAntenna = [], badUBLpair = [], antennaLocationTolerance = 1e-6):
        '''XXX DOCSTRING'''
        self.antennaLocationTolerance = antennaLocationTolerance
        self.badAntenna += badAntenna
        self.badUBLpair += badUBLpair

        if arrayinfoPath is not None and os.path.isfile(arrayinfoPath):
            self.read_arrayinfo(arrayinfoPath)
        if np.linalg.norm(self.antennaLocation) == 0:
            raise Exception("Error: compute_redundantinfo() called before self.antennaLocation is specified. Use configFilePath option when calling compute_redundantinfo() to specify array info file, or manually set self.antennaLocation for the RedundantCalibrator instance.")
        if verbose:
            timer = Timer()

        #antennalocation quality check: make sure there's no crazy constant added to antlocs
        bad_ant_mask = np.array([a in self.badAntenna for a in range(len(self.antennaLocation))]).astype('bool')
        array_center = la.norm(np.mean(self.antennaLocation[~bad_ant_mask], axis = 0))
        array_std = la.norm(np.std(self.antennaLocation[~bad_ant_mask], axis = 0))
        #print array_center, array_std
        if array_std / array_center < 1e-3:
            raise TypeError("Average antenna location is %s whereas the typical variation among locations is %s, which is too small and will cause many problems. Please remove the large overall offset from antenna locations."%(np.mean(self.antennaLocation[~bad_ant_mask], axis = 0), np.std(self.antennaLocation[~bad_ant_mask], axis = 0)))

        #nAntenna and subsetant : get rid of the bad antennas
        nant=len(self.antennaLocation)
        #subsetant=[i for i in range(nant) if i not in self.badAntenna]
        ant2goodant = -np.ones(len(self.antennaLocation), dtype=int)
        subsetant = []
        for a in range(len(self.antennaLocation)):
            if a not in self.badAntenna:
                subsetant.append(a)
                ant2goodant[a] = len(subsetant) - 1

        nAntenna=len(subsetant)
        antloc=[self.antennaLocation[ant] for ant in subsetant]
        if verbose:
            timer.tick('a')
        ##########################################################################################
        #find out ubl
        #use the function compute_UBL to find the ubl
        tolerance=self.antennaLocationTolerance;
        ublall=self.compute_UBL(tolerance)
        if verbose:
            timer.tick('b')
        #################################################################################################
        #calculate the norm of the difference of two vectors (just la.norm actually)
        def dis(a1,a2):
            return np.linalg.norm(np.array(a1)-np.array(a2))
        #find badUBL with badUBLpair
        def find_ublindex_all(pair):
            #print pair
            for i in range(len(ublall)):
                if dis(self.antennaLocation[pair[0]]-self.antennaLocation[pair[1]],ublall[i]) < tolerance or dis(self.antennaLocation[pair[0]]-self.antennaLocation[pair[1]],-ublall[i]) < tolerance:
                    return i
            return None
            #raise Exception("Error: something wrong in identifying badUBL from badUBLpair")    #delete this line afterwards
        #print self.badUBLpair, len(self.badUBLpair),self.badUBLpair[0]
        for p in self.badUBLpair:
            self.badUBL.append(find_ublindex_all(p))
        self.badUBL = [i for i in self.badUBL if i is not None]
        self.ubl2goodubl = -np.ones(len(ublall), dtype=int)
        goodu = 0
        for u in range(len(ublall)):
            if u not in self.badUBL:
                self.ubl2goodubl[u] = goodu
                goodu = goodu + 1

        #delete the bad ubl's
        ubl=np.delete(ublall,np.array(self.badUBL).astype('int'),0)
        nUBL=len(ubl);
        badbl=[ublall[i] for i in self.badUBL]
        #find nBaseline (include auto baselines) and subsetbl
        nbl=0;
        goodpairs=[];
        for i in range(len(antloc)):
            for j in range(i+1):
                bool=False
                for bl in badbl:
                    bool = bool or dis(antloc[i]-antloc[j],bl)<tolerance or dis(antloc[i]-antloc[j],-bl)<tolerance
                if bool == False:
                    nbl+=1
                    goodpairs.append([i,j])
        ####for a1, a2 in self.totalVisibilityId:
            ####i = ant2goodant[a1]
            ####j = ant2goodant[a2]
            ####if i >= 0 and j >= 0:
                ####bool=False
                ####for bl in badbl:
                    ####bool = bool or dis(antloc[i]-antloc[j],bl)<tolerance or dis(antloc[i]-antloc[j],-bl)<tolerance
                ####if bool == False:
                    ####nbl+=1
                    ####goodpairs.append([i,j])

        self.totalVisibilityId_dic = {}
        for bll, (a1, a2) in enumerate(self.totalVisibilityId):
            self.totalVisibilityId_dic[(a1,a2)] = bll

        #correct the orders of pairs in goodpair
        def correct_pairorder(pair):
            ####try:
                ####self.totalVisibilityId.tolist().index([pair[0],pair[1]])
                ####return True
            ####except:
                ####try:
                    ####self.totalVisibilityId.tolist().index([pair[1], pair[0]])
                    ####return False
                ####except:
                    ####return None
            if (pair[0], pair[1]) in self.totalVisibilityId_dic:
                return True
            elif (pair[1], pair[0]) in self.totalVisibilityId_dic:
                return False
            else:
                return None
        if verbose:
            timer.tick('c')
        #exclude pairs that are not in totalVisibilityId
        temp = []
        for p in goodpairs:
            cond = correct_pairorder([subsetant[p[0]],subsetant[p[1]]])
            if cond == True:
                temp.append(p)
            if cond == False:
                #print "correcting"
                temp.append(p[::-1])
        goodpairs = temp

        #goodpairs = [correct_pairorder([subsetant[p[0]],subsetant[p[1]]]) for p in goodpairs if (correct_pairorder([subsetant[p[0]],subsetant[p[1]]]) != None and correct_pairorder([subsetant[p[0]],subsetant[p[1]]]) == True)]  #correct_pairorder([subsetant[p[0]],subsetant[p[1]]])
        nBaseline=len(goodpairs)
        if verbose:
            timer.tick('c')
        #from a pair of good antenna index to baseline index
        subsetbl = np.array([self.get_baseline([subsetant[bl[0]],subsetant[bl[1]]]) for bl in goodpairs])
        if verbose:
            timer.tick('c')
        ##################################################################################
        #bltoubl: cross bl number to ubl index
        ####def findublindex(pair,ubl=ubl):
            ####i=pair[0]
            ####j=pair[1]
            ####for k in range(len(ubl)):
                ####if dis(antloc[i]-antloc[j],ubl[k])<tolerance or dis(antloc[i]-antloc[j],-ubl[k])<tolerance:
                    ####return k
            ####print pair
            ####return "no match"
        def findublindex(pair):
            if (subsetant[pair[0]], subsetant[pair[1]]) in self.totalVisibilityUBL:
                return self.ubl2goodubl[self.totalVisibilityUBL[(subsetant[pair[0]], subsetant[pair[1]])]]
        bltoubl=[];
        for p in goodpairs:
            if p[0]!=p[1]:
                bltoubl.append(findublindex(p))
        if verbose:
            timer.tick('d')
        #################################################################################
        #reversed:   cross only bl if reversed -1, otherwise 1
        crosspair=[]
        for p in goodpairs:
            if p[0]!=p[1]:
                crosspair.append(p)
        reverse=[]
        for k in range(len(crosspair)):
            i=crosspair[k][0]
            j=crosspair[k][1]
            if dis(antloc[i]-antloc[j],ubl[bltoubl[k]])<tolerance:
                reverse.append(-1)
            elif dis(antloc[i]-antloc[j],-ubl[bltoubl[k]])<tolerance:
                reverse.append(1)
            else :
                print "something's wrong with bltoubl", crosspair[k], antloc[i]-antloc[j], bltoubl[k], ubl[bltoubl[k]]
                print i,j, subsetant[i], subsetant[j]
                print self.totalVisibilityUBL[(subsetant[i], subsetant[j])]
                exit(1)
        if verbose:
            timer.tick('e')
        ######################################################################################
        #reversedauto: the index of good baselines (auto included) in all baselines
        #autoindex: index of auto bls among good bls
        #crossindex: index of cross bls among good bls
        #ncross
        reversedauto = range(len(goodpairs))
        #find the autoindex and crossindex in goodpairs
        autoindex=[]
        crossindex=[]
        for i in range(len(goodpairs)):
            if goodpairs[i][0]==goodpairs[i][1]:
                autoindex.append(i)
            else:
                crossindex.append(i)
        for i in autoindex:
            reversedauto[i]=1
        for i in range(len(crossindex)):
            reversedauto[crossindex[i]]=reverse[i]
        reversedauto=np.array(reversedauto)
        autoindex=np.array(autoindex)
        crossindex=np.array(crossindex)
        ncross=len(crossindex)
        if verbose:
            timer.tick('f')
        ###################################################
        #bl2d:  from 1d bl index to a pair of antenna numbers
        bl2d=[]
        for pair in goodpairs:
            bl2d.append(pair)#(pair[::-1])
        bl2d=np.array(bl2d)
        if verbose:
            timer.tick('g')
        ###################################################
        #ublcount:  for each ubl, the number of good cross bls corresponding to it

        countdict={}
        for bl in bltoubl:
            countdict[bl]=0

        for bl in bltoubl:
            countdict[bl]+=1


        ublcount=[]
        for i in range(nUBL):
            ublcount.append(countdict[i])
        ublcount=np.array(ublcount)
        if verbose:
            timer.tick('h')
        ####################################################################################
        #ublindex:  //for each ubl, the vector<int> contains (ant1, ant2, crossbl)
        countdict={}
        for bl in bltoubl:
            countdict[bl]=[]

        for i in range(len(crosspair)):
            ant1=crosspair[i][0]
            ant2=crosspair[i][1]
            countdict[bltoubl[i]].append([ant1,ant2,i])  #([ant1,ant2,i])

        ublindex=[]
        for i in range(nUBL):
            ublindex.append(countdict[i])
        #turn each list in ublindex into np array
        for i in range(len(ublindex)):
            ublindex[i]=np.array(ublindex[i])
        ublindex=np.array(ublindex)
        if verbose:
            timer.tick('i')
        ###############################################################################
        #bl1dmatrix: a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
                #I suppose 99999 for bad and auto baselines?
        bl1dmatrix=(2**31-1)*np.ones([nAntenna,nAntenna],dtype='int32')
        for i in range(len(crosspair)):
            bl1dmatrix[crosspair[i][1]][crosspair[i][0]]=i
            bl1dmatrix[crosspair[i][0]][crosspair[i][1]]=i
        if verbose:
            timer.tick('j')
        ####################################################################################3
        #degenM:
        a=[]
        for i in range(len(antloc)):
            a.append(np.append(antloc[i],1))
        a=np.array(a)

        d=[]
        for i in range(len(ubl)):
            d.append(np.append(ubl[i],0))
        d=np.array(d)

        m1=-a.dot(la.pinv(np.transpose(a).dot(a))).dot(np.transpose(a))
        m2=d.dot(la.pinv(np.transpose(a).dot(a))).dot(np.transpose(a))
        degenM = np.append(m1,m2,axis=0)
        #####################################################################################
        #A: A matrix for logcal amplitude
        A=np.zeros([len(crosspair),nAntenna+len(ubl)])
        for i in range(len(crosspair)):
            A[i][crosspair[i][0]]=1
            A[i][crosspair[i][1]]=1
            A[i][nAntenna+bltoubl[i]]=1
        A=sps.csr_matrix(A)
        #################################################################################
        #B: B matrix for logcal phase
        B=np.zeros([len(crosspair),nAntenna+len(ubl)])
        for i in range(len(crosspair)):
            B[i][crosspair[i][0]]=reverse[i]*-1   #1
            B[i][crosspair[i][1]]=reverse[i]*1  #-1
            B[i][nAntenna+bltoubl[i]]=1
        B=sps.csr_matrix(B)
        if verbose:
            timer.tick('k')
        ###########################################################################
        #create info dictionary
        info={}
        info['nAntenna']=nAntenna
        info['nUBL']=nUBL
        info['nBaseline']=nBaseline
        info['subsetant']=subsetant
        info['antloc']=antloc
        info['subsetbl']=subsetbl
        info['ubl']=ubl
        info['bltoubl']=bltoubl
        info['reversed']=reverse
        info['reversedauto']=reversedauto
        info['autoindex']=autoindex
        info['crossindex']=crossindex
        #info['ncross']=ncross
        info['bl2d']=bl2d
        info['ublcount']=ublcount
        info['ublindex']=ublindex
        info['bl1dmatrix']=bl1dmatrix
        info['degenM']=degenM
        info['A']=A
        info['B']=B
        if verbose:
            timer.tick('l')
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=DeprecationWarning)
                info['At'] = info['A'].transpose()
                info['Bt'] = info['B'].transpose()
                if verbose:
                    timer.tick('m')
                info['AtAi'] = la.pinv(info['At'].dot(info['A']).todense(), cond = 10**(-6))#(AtA)^-1
                info['BtBi'] = la.pinv(info['Bt'].dot(info['B']).todense(), cond = 10**(-6))#(BtB)^-1
                #if verbose:
                    #timer.tick('m')
                #info['AtAiAt'] = info['AtAi'].dot(info['At'].todense())#(AtA)^-1At
                #info['BtBiBt'] = info['BtBi'].dot(info['Bt'].todense())#(BtB)^-1Bt
                #if verbose:
                    #timer.tick('m')
                #info['PA'] = info['A'].dot(info['AtAiAt'])#A(AtA)^-1At
                #info['PB'] = info['B'].dot(info['BtBiBt'])#B(BtB)^-1Bt
                #if verbose:
                    #timer.tick('m')
                #info['ImPA'] = sps.identity(ncross) - info['PA']#I-PA
                #info['ImPB'] = sps.identity(ncross) - info['PB']#I-PB
        info['totalVisibilityId'] = self.totalVisibilityId
        if verbose:
            timer.tick('m')
        self.Info = RedundantInfo(info)
        if verbose:
            timer.tick('n')


    def get_baseline(self,pair):
        '''inverse function of totalVisibilityId, calculate the baseline index from 
        the antenna pair. It allows flipping of a1 and a2, will return same result'''
        if not (type(pair) == list or type(pair) == np.ndarray or type(pair) == tuple):
            raise Exception("input needs to be a list of two numbers")
            return
        elif len(np.array(pair)) != 2:
            raise Exception("input needs to be a list of two numbers")
            return
        elif type(pair[0]) == str or type(pair[0]) == np.string_:
            raise Exception("input needs to be number not string")
            return
        ####try:
            ####return self.totalVisibilityId.tolist().index([pair[0],pair[1]])
        ####except:
            ####try:
                ####return self.totalVisibilityId.tolist().index([pair[1], pair[0]])
            ####except:
                #####raise Exception("Error: antenna pair %s not found in self.totalVisibilityId."%pair)
                ####return None
        if self.totalVisibilityId_dic is None:
            self.totalVisibilityId_dic = {}
            for bll, (a1, a2) in enumerate(self.totalVisibilityId):
                self.totalVisibilityId_dic[(a1,a2)] = bll
        if (pair[0],pair[1]) in self.totalVisibilityId_dic:
            return self.totalVisibilityId_dic[(pair[0],pair[1])]
        elif (pair[1],pair[0]) in self.totalVisibilityId_dic:
            return self.totalVisibilityId_dic[(pair[1],pair[0])]
        else:
            return None


    def compute_UBL_old2(self,tolerance = 0.1):
        '''compute_UBL returns the average of all baselines in that ubl group'''
        #check if the tolerance is not a string
        if type(tolerance) == str:
            raise Exception("tolerance needs to be number not string")
            return
        #remove the bad antennas
        nant=len(self.antennaLocation)
        subsetant=[i for i in range(nant) if i not in self.badAntenna]
        nAntenna=len(subsetant)
        antloc = np.array([self.antennaLocation[ant] for ant in subsetant])
        ubllist = np.array([np.array([np.array([0,0,0]),1])]);
        for i in range(len(antloc)):
            #for j in range(i+1,len(antloc)):    #(this gives the same redundant info as the correct info saved in test)
            for j in range(i+1):
                bool = True
                for k in range(len(ubllist)):
                    if  la.norm(antloc[i]-antloc[j]-ubllist[k][0])<tolerance:
                        n=ubllist[k][1]
                        ubllist[k][0]=1/(n+1.0)*(n*ubllist[k][0]+antloc[i]-antloc[j])
                        ubllist[k][1]+=1
                        bool = False
                    elif  la.norm(antloc[i]-antloc[j]+ubllist[k][0])<tolerance:
                        n=ubllist[k][1]
                        ubllist[k][0]=1/(n+1.0)*(n*ubllist[k][0]-(antloc[i]-antloc[j]))
                        ubllist[k][1]+=1
                        bool = False
                if bool :
                    ubllist = np.append(ubllist,np.array([np.array([antloc[j]-antloc[i],1])]),axis=0)
        ubllist = np.delete(ubllist,0,0)
        ublall=[]
        for ubl in ubllist:
            ublall.append(ubl[0])
        ublall=np.array(ublall)
        return ublall


    def compute_UBL_old(self,tolerance = 0.1):
        '''compute_UBL returns the average of all baselines in that ubl group'''
        #check if the tolerance is not a string
        if type(tolerance) == str:
            raise Exception("tolerance needs to be number not string")
            return
        ubllist = np.array([np.array([np.array([0,0,0]),1])]);
        for pair in self.totalVisibilityId:
            if pair[0] not in self.badAntenna and pair[1] not in self.badAntenna:
                [i,j] = pair
                bool = True
                for k in range(len(ubllist)):
                    if  la.norm(self.antennaLocation[i]-self.antennaLocation[j]-ubllist[k][0])<tolerance:
                        n=ubllist[k][1]
                        ubllist[k][0]=1/(n+1.0)*(n*ubllist[k][0]+self.antennaLocation[i]-self.antennaLocation[j])
                        ubllist[k][1]+=1
                        bool = False
                    elif  la.norm(self.antennaLocation[i]-self.antennaLocation[j]+ubllist[k][0])<tolerance:
                        n=ubllist[k][1]
                        ubllist[k][0]=1/(n+1.0)*(n*ubllist[k][0]-(self.antennaLocation[i]-self.antennaLocation[j]))
                        ubllist[k][1]+=1
                        bool = False
                if bool :
                    ubllist = np.append(ubllist,np.array([np.array([self.antennaLocation[j]-self.antennaLocation[i],1])]),axis=0)
        ubllist = np.delete(ubllist,0,0)
        ublall=[]
        for ubl in ubllist:
            ublall.append(ubl[0])
        ublall=np.array(ublall)
        return ublall

    def compute_UBL(self,tolerance = 0.1):
        '''XXX DOCSTRING'''
        if tolerance == 0:
            tolerance = np.min(np.linalg.norm(np.array(self.antennaLocation) - self.antennaLocation[0], axis = 1)) / 1.e6
        ubl = {}
        for bl, (a1, a2) in enumerate(self.totalVisibilityId):
            if a1 != a2 and a1 not in self.badAntenna and a2 not in self.badAntenna:
                loc_tuple = tuple(np.round((self.antennaLocation[a2] - self.antennaLocation[a1]) / float(tolerance)) * tolerance)
                neg_loc_tuple = tuple(np.round((self.antennaLocation[a1] - self.antennaLocation[a2]) / float(tolerance)) * tolerance)
                if loc_tuple in ubl:
                    ubl[loc_tuple].add(bl + 1)
                elif neg_loc_tuple in ubl:
                    ubl[neg_loc_tuple].add(- bl - 1)
                else:
                    if loc_tuple[0] >= 0:
                        ubl[loc_tuple] = set([bl + 1])
                    else:
                        ubl[neg_loc_tuple] = set([-bl - 1])

        #calculate actual average of the gridded baseline vectors to get an accurate representation of the ubl vector
        ubl_vec = np.zeros((len(ubl), 3))
        self.totalVisibilityUBL = {}

        ublcount = np.zeros(len(ubl))
        for u, grid_ubl_vec in enumerate(ubl):
            for bl in ubl[grid_ubl_vec]:
                assert bl != 0
                a1, a2 = self.totalVisibilityId[abs(bl) - 1]
                if bl > 0:
                    ubl_vec[u] = ubl_vec[u] + self.antennaLocation[a2] - self.antennaLocation[a1]
                else:
                    ubl_vec[u] = ubl_vec[u] + self.antennaLocation[a1] - self.antennaLocation[a2]
                self.totalVisibilityUBL[(a1, a2)] = u
            ublcount[u] = len(ubl[grid_ubl_vec])
            ubl_vec[u] = ubl_vec[u] / ublcount[u]

        reorder = (ubl_vec[:,1]*1e9 + ubl_vec[:,0]).argsort()
        rereorder = reorder.argsort()
        for key in self.totalVisibilityUBL:
            self.totalVisibilityUBL[key] = rereorder[self.totalVisibilityUBL[key]]
        ubl_vec = ubl_vec[reorder]

        #now I need to deal with the fact that no matter how coarse my grid is, it's possible for a single group of ubl to fall into two adjacent grids. So I'm going to check if any of the final ubl vectors are seperated by less than tolerance. If so, merge them
        ublmap = {}
        for u1 in range(len(ubl_vec)):
            for u2 in range(u1):
                if la.norm(ubl_vec[u2] - ubl_vec[u1]) < tolerance or la.norm(ubl_vec[u2] + ubl_vec[u1]) < tolerance:
                    ublmap[u1] = u2
                    ubl_vec[u2] = (ubl_vec[u1] * ublcount[u1] + ubl_vec[u2] * ublcount[u2]) / (ublcount[u1] + ublcount[u2])
                    break
            ublmap[u1] = u1

        merged_ubl_vec = []
        for u in range(len(ubl_vec)):
            if ublmap[u] == u:
                merged_ubl_vec.append(ubl_vec[u])
                ublmap[u] = len(merged_ubl_vec) - 1
            else:
                ublmap[u] = ublmap[ublmap[u]]
        merged_ubl_vec = np.array(merged_ubl_vec)

        for key in self.totalVisibilityUBL:
            self.totalVisibilityUBL[key] = ublmap[self.totalVisibilityUBL[key]]
        return ubl_vec


    def get_ublindex(self,antpair):
        '''need to do compute_redundantinfo first for this function to work (needs 'bl1dmatrix')
        input the antenna pair(as a list of two numbers), return the corresponding ubl index'''
        #check if the input is a list, tuple, np.array of two numbers
        if not (type(antpair) == list or type(antpair) == np.ndarray or type(antpair) == tuple):
            raise Exception("input needs to be a list of two numbers")
            return
        elif len(np.array(antpair)) != 2:
            raise Exception("input needs to be a list of two numbers")
            return
        elif type(antpair[0]) == str or type(antpair[0]) == np.string_:
            raise Exception("input needs to be number not string")
            return

        #check if self.info['bl1dmatrix'] exists
        try:
            _ = self.Info.bl1dmatrix
        except:
            raise Exception("needs Info.bl1dmatrix")

        crossblindex=self.Info.bl1dmatrix[antpair[0]][antpair[1]]
        if antpair[0]==antpair[1]:
            return "auto correlation"
        elif crossblindex == 99999:
            return "bad ubl"
        return self.Info.bltoubl[crossblindex]


    def get_reversed(self,antpair):
        '''need to do compute_redundantinfo first
        input the antenna pair, return -1 if it is a reversed baseline and 1 if it is not reversed'''
        #check if the input is a list, tuple, np.array of two numbers
        if not (type(antpair) == list or type(antpair) == np.ndarray or type(antpair) == tuple):
            raise Exception("input needs to be a list of two numbers")
            return
        elif len(np.array(antpair)) != 2:
            raise Exception("input needs to be a list of two numbers")
            return
        elif type(antpair[0]) == str or type(antpair[0]) == np.string_:
            raise Exception("input needs to be number not string")
            return

        #check if self.info['bl1dmatrix'] exists
        try:
            _ = self.Info.bl1dmatrix
        except:
            raise Exception("needs Info.bl1dmatrix")

        crossblindex=self.Info.bl1dmatrix[antpair[0]][antpair[1]]
        if antpair[0] == antpair[1]:
            return 1
        if crossblindex == 99999:
            return 'badbaseline'
        return self.Info.reversed[crossblindex]

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

        self.badAntenna = range(16) + range(56,60) + [16,19,50]

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

def load_omnichisq(path):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise IOError("Path %s does not exist."%path)

    omnichisq = np.fromfile(path, dtype = 'float32')
    NF = int(omnichisq[2])
    omnichisq.shape = (len(omnichisq) / (NF + 3), (NF + 3))
    return omnichisq

def load_omnigain(path, info=None):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise IOError("Path %s does not exist."%path)
    if info is None:
        info = path.replace('.omnigain', '.binfo')
    if type(info) == type('a'):
        info = read_redundantinfo(info)


    omnigain = np.fromfile(path, dtype = 'float32')
    omnigain.shape = (omnigain.shape[0] / (info['nAntenna']) / (2 + 1 + 1 + 2 * int(omnigain[3])), info['nAntenna'], 2 + 1 + 1 + 2 * int(omnigain[3]))
    return omnigain

def load_omnifit(path, info=None):
    '''XXX DOCSTRING'''
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise IOError("Path %s does not exist."%path)
    if info is None:
        info = path.replace('.omnifit', '.binfo')
    if type(info) == type('a'):
        info = read_redundantinfo(info)


    omnifit = np.fromfile(path, dtype = 'float32')
    omnifit.shape = (omnifit.shape[0] / (info['nUBL']) / (2 + 3 + 1 + 2 * int(omnifit[5])), info['nUBL'], 2 + 3 + 1 + 2 * int(omnifit[3]))
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
    '''plot_3: only plot the 3 most redundant ones. plot_1: counting start from 0 the most redundant baseline'''
    import matplotlib.pyplot as plt
    data = np.array(data_in)
    try:#in case info is Info class
        info = info.get_info()
    except:
        pass
    if plot_3 and info['nUBL'] < 3:
        plot_3 = False

    colors=[]
    colorgrid = int(math.ceil((info['nUBL']/12.+1)**.34))
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
                    ax.scatter(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]),np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')])*info['reversed'][np.array(info['ublindex'][ubl][:,2]).astype('int')], marker=marker, color=color)
                    outputdata[i] = outputdata[i] + [(np.real(d[np.array(info['ublindex'][ubl][:,2]).astype('int')]) + 1.j * np.imag(d[np.array(info['ublindex'][ubl][:,2]).astype('int')])*info['reversed'][np.array(info['ublindex'][ubl][:,2]).astype('int')], marker, color, info['ubl'][ubl])]

                ubl += 1
                if ubl == info['nUBL']:
                    #if i == 1:
                        #ax.text(-(len(ds)-1 + 0.7)*plotrange, -0.7*plotrange, "#Ant:%i\n#UBL:%i"%(info['nAntenna'],info['nUBL']),bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
                    ax.set_title(title + "\nGood Antenna count: %i\nUBL count: %i"%(info['nAntenna'],info['nUBL']))
                    ax.grid(True)
                    ax.set(adjustable='datalim', aspect=1)
                    ax.set_xlabel('Real')
                    ax.set_ylabel('Imag')
                    break
            if ubl == info['nUBL']:
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
    if len(v1) != len(v2):
        raise Exception("Length mismatch %i vs %i."%(len(v1), len(v2)))
    if la.norm(v1) == 0:
        return True
    return la.norm(np.dot(v1, v2)/np.dot(v1, v1) * v1 - v2) <= tol

# XXX utility function belongs in another file
def _f(rawcal_ubl=[], verbose=False):
    '''run this function twice in a row and its christmas'''
    if verbose and rawcal_ubl != []:
        print "Starting ubl:", rawcal_ubl
    if rawcal_ubl == []:
        rawcal_ubl += [2,3]
    if verbose:
        print "ubl:", rawcal_ubl

def find_solution_path(info, input_rawcal_ubl=[], tol = 0.0, verbose=False):
    '''return (intialantenna, solution_path) for raw calibration. solution path
    contains a list of [(a0, a1, crossubl), a] = [(ublindex entry), (which ant is
    solved, 0 or 1)]. When raw calibrating, initialize calpar to have [0] at
    initial antenna, then simply iterate along the solution_path, use crossubl and
    a0 or a1 specified by a to solve for the other a1 or a0 and append it to
    calpar. Afterwards, use mean angle on calpars'''
    ###select 2 ubl for calibration
    rawcal_ubl = list(input_rawcal_ubl)
    if verbose and rawcal_ubl != []:
        print "Starting ubl:", rawcal_ubl
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
            except:
                raise Exception("Cannot find two unique baselines that are linearly independent!")
            ublcnt_tmp[rawcal_ubl[-1]] = np.nan
    if verbose:
        print "ubl:", info['ubl'][rawcal_ubl[0]], info['ubl'][rawcal_ubl[1]]

    if info['ublcount'][rawcal_ubl[0]] + info['ublcount'][rawcal_ubl[1]] <= info['nAntenna'] + 2:
        raise Exception('Array not redundant enough! Two most redundant baselines %s and %s have %i and %i baselines, which is not larger than 2 + %i'%(info['ubl'][rawcal_ubl[0]],info['ubl'][rawcal_ubl[1]], info['ublcount'][rawcal_ubl[0]],info['ublcount'][rawcal_ubl[1]], info['nAntenna']))

    ublindex = np.concatenate((np.array(info['ublindex'][rawcal_ubl[0]]).astype('int'), np.array(info['ublindex'][rawcal_ubl[1]]).astype('int')))#merge ublindex since we set both ubl phase to 0

    ###The overarching goal is to find a solution path (a sequence of unique baselines to solve) that can get multiple solutions to multiple antennas using just two sets of ubls
    solution_path = []

    antcnt = np.bincount(np.array(ublindex)[:,:2].flatten(), minlength = info['nAntenna'])#how many times each antenna appear in the two sets of ubls. at most 4. not useful if < 2
    unsolved_ant = []
    for a in range(len(antcnt)):
        if antcnt[a] == 0:
            unsolved_ant.append(a)
    if verbose:
        print "antcnt", antcnt, "Antennas", np.array(info['subsetant'])[unsolved_ant], "not directly solvable."


    ###Status string for ubl: NoUse: none of the two ants have been solved; Solvable: at least one of the ants have solutions; Done: used to generate one antennacalpar
    ublstatus = ["NoUse" for i in ublindex]

    ###antenna calpars, a list for each antenna
    calpar = np.array([[]] * info['nAntenna']).tolist()
    ###select initial antenna
    initialant = int(np.argmax(antcnt))
    if verbose:
        print "initialant", np.array(info['subsetant'])[initialant]
    calpar[initialant].append(0)
    for i in range(len(ublstatus)):
        if initialant in ublindex[i, 0:2]:
            ublstatus[i] = "Solvable"

    ###start looping
    solvecnt = 10#number of solved baselines in each loop, 10 is an arbitrary starting point
    if verbose:
        print "new ant solved",
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
                        if verbose:
                            print np.array(info['subsetant'])[ublindex[i, 1]],
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
                        if verbose:
                            print np.array(info['subsetant'])[ublindex[i, 0]],
                        for j in range(len(ublstatus)):
                            if (ublindex[i, 0] in ublindex[j, 0:2]) and ublstatus[j] == "NoUse":
                                ublstatus[j] = "Solvable"
    if verbose:
        print ""
        if len(solution_path) != len(ublindex):
            print "Solution path has %i entries where as total candidates in ublindex have %i. The following baselines form their isolated isaland:"%(len(solution_path), len(ublindex))
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
            if verbose:
                print "trying to solve for ", np.array(info['subsetant'])[a],
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
                if verbose:
                    print "trying ubl ", third_ubl,
                third_ubl_good = False #assume false and start checking if this ubl 1) has this antenna 2) has another baseline whose two ants are both solved
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
                        get_ubl_fit.append([a1, a2, bl, info['reversed'][bl]])
                for a1, a2, bl in info['ublindex'][third_ubl].astype('int'):
                    if (a1 not in unsolved_ant) and (a2 == a):
                        get_ubl_fit.append([a1, a2, bl, info['reversed'][bl], 0])
                        break
                    if (a2 not in unsolved_ant) and (a1 == a):
                        get_ubl_fit.append([a1, a2, bl, info['reversed'][bl], 1])
                        break
                additional_solution_path.append(get_ubl_fit)
                ant_solved += 1
                unsolved_ant.remove(a)

    #remove the effect of enforcing the two baselines to be 0, rather, set the first two linearly independent antennas w.r.t initant to be 0
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
    if verbose:
        print "Degeneracy: a1 = %i, a2 = %i"%(info['subsetant'][a1], info['subsetant'][a2])
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

# XXX utility function belongs in another file
def collapse_shape(shape, axis):
    '''XXX DOCSTRING'''
    if axis == 0 or axis == -len(shape):
        return tuple(list(shape)[1:])
    elif axis == -1 or axis == len(shape) - 1:
        return tuple(list(shape)[:-1])
    else:
        return tuple(list(shape)[:axis] + list(shape)[axis+1:])

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
    #remove the effect of enforcing the two baselines to be 0, rather, set the first two linearly independent antennas w.r.t initant to be 0

    calpar = calpar - degeneracy_remove[2].dot([calpar[degeneracy_remove[0]],calpar[degeneracy_remove[1]]])

    result[info['subsetant']] = np.exp(1.j*calpar)# * result[info['subsetant']]
    return result

# XXX utility class belongs in another file
class InverseCholeskyMatrix:
    '''for a positive definite matrix, Cholesky decomposition is M = L.Lt, where L
    lower triangular. This decomposition helps computing inv(M).v faster, by
    avoiding calculating inv(M). Once we have L, the product is simply
    inv(Lt).inv(L).v, and inverse of triangular matrices multiplying a vector is
    fast. sla.solve_triangular(M, v) = inv(M).v'''
    def __init__(self, matrix):
        if type(matrix).__module__ != np.__name__ or len(matrix.shape) != 2:
            raise TypeError("matrix must be a 2D numpy array");
        try:
            self.L = la.cholesky(matrix)#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
            self.Lt = self.L.conjugate().transpose()
            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
        except:
            raise TypeError("cholesky failed. matrix is not positive definite.")

    @classmethod
    def fromfile(cls, filename, n, dtype):
        if not os.path.isfile(filename):
            raise IOError("%s file not found!"%filename)
        matrix = cls(np.array([[1,0],[0,1]]))
        try:
            matrix.L = np.fromfile(filename, dtype=dtype).reshape((n,n))#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
            matrix.Lt = matrix.L.conjugate().transpose()
            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
        except:
            raise TypeError("cholesky import failed. matrix is not %i by %i with dtype=%s."%(n, n, dtype))
        return matrix

    def dotv(self, vector):
        try:
            return la.solve_triangular(self.Lt, la.solve_triangular(self.L, vector, lower=True), lower=False)
        except:
            return np.empty_like(vector)+np.nan

    def dotM(self, matrix):
        return np.array([self.dotv(v) for v in matrix.transpose()]).transpose()

    def astype(self, t):
        self.L = self.L.astype(t)
        self.Lt = self.Lt.astype(t)
        return self

    def tofile(self, filename, overwrite = False):
        if os.path.isfile(filename) and not overwrite:
            raise IOError("%s file exists!"%filename)
        self.L.tofile(filename)

# XXX utility function belongs elsewhere
def solve_slope(A_in, b_in, tol, niter=30, step=1, verbose=False):
    '''solve for the solution vector x such that mod(A.x, 2pi) = b, 
    where the values range from -p to p. solution will be seeked 
    on the first axis of b'''
    timer = Timer()
    p = np.pi
    A = np.array(A_in)
    b = np.array(b_in + p) % (2*p) - p
    if A.ndim != 2:
        raise TypeError("A matrix must be 2 dimensional. Input A is %i dimensional."%A.ndim)
    if A.shape[0] != b.shape[0]:
        raise TypeError("A and b has shape mismatch: %s and %s."%(A.shape, b.shape))
    if A.shape[1] != 2:
        raise TypeError("A matrix's second dimension must have size of 2. %i inputted."%A.shape[1])
    if verbose:
        timer.tick("a")
    #find the shortest 2 non-parallel baselines, candidate_vecs have all combinations of vectors in a summation or subtraction. Each entry is i,j, v0,v1 where Ai+Aj=(v0,v1), negative j means subtraction. Identical i,j means vector itself without add or subtract
    candidate_vecs = np.zeros((len(A)**2, 4), dtype = 'float32')
    n = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i < j:
                candidate_vecs[n] = [i, j, A[i,0]+A[j,0], A[i,1]+A[j,1]]
            elif i == j:
                candidate_vecs[n] = [i, j, A[i,0], A[i,1]]
            elif i > j:
                candidate_vecs[n] = [i, -j, A[i,0]-A[j,0], A[i,1]-A[j,1]]

            n = n + 1
    if verbose:
        timer.tick("b")
    candidate_vecs = candidate_vecs[np.linalg.norm(candidate_vecs, axis=1)>tol]

    #construct coarse A that consists of the 2 shortest vecs
    coarseA = np.zeros((2,2), dtype = 'float32')
    if b.ndim > 1:
        coarseb = np.zeros(np.concatenate(([2], b.shape[1:])), dtype='float32')
    else:
        coarseb = np.zeros(2, dtype='float32')

    for n in np.argsort(np.linalg.norm(candidate_vecs[:, 2:4], axis=1)):
        v = candidate_vecs[n, 2:4]
        if la.norm(coarseA[0]) == 0:
            coarseA[0] = v
        else:
            perp_component = v - v.dot(coarseA[0])/(coarseA[0].dot(coarseA[0])) * coarseA[0]
            if la.norm(perp_component) > tol:
                coarseA[1] = v
                break
    if la.norm(coarseA[1]) == 0:
        raise Exception("Poorly constructed A matrix: cannot find a pair of orthogonal vectors")
    if verbose:
        timer.tick("c")
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
            if la.norm(coarseA[nn] - v) < tol:
                bsign = 1
            else:
                bsign = -1
            if i < j:
                coarseb_candidate[n] = b[i]+b[j]#(b[i]+b[j]+p)%(2*p)-p
            elif i == j:
                coarseb_candidate[n] = b[i]
            elif i > j:
                coarseb_candidate[n] = b[i]-b[abs(j)]#(b[i]-b[abs(j)]+p)%(2*p)-p
            coarseb_candidate[n] = coarseb_candidate[n] * bsign
    if verbose:
        timer.tick("d")

    coarseb[0] = medianAngle(coarseb0_candidate, axis=0)
    coarseb[1] = medianAngle(coarseb1_candidate, axis=0)
    if verbose:
        print coarseb0_candidate.shape
        timer.tick("d")
    # find coarse solutions
    try:
        icA = la.inv(coarseA)
    except:
        raise Exception("Poorly constructed coarseA matrix: %s."%(coarseA))
    try:
        #iA = InverseCholeskyMatrix(A.transpose().dot(A))
        iA = la.inv(A.transpose().dot(A))
    except:
        raise Exception("Poorly constructed A matrix: %s."%(A.transpose().dot(A)))
    if verbose:
        print iA.shape
        timer.tick("e")
    if b.ndim > 2:
        extra_shape = b.shape[1:]
        flat_extra_dim = 1
        for i in range(1, b.ndim):
            flat_extra_dim = flat_extra_dim * b.shape[i]
        coarseb.shape = (2, flat_extra_dim)
        b.shape = (len(b), flat_extra_dim)
    else:
        extra_shape = None
    if verbose:
        timer.tick("f")
    result = icA.dot(coarseb)
    if verbose:
        print coarseA
        print result
    for i in range(niter):
        result = result + step * iA.dot(A.transpose().dot((b - A.dot(result) + p)%(2*p)-p))
        if verbose:
            print result
    if extra_shape is not None:
        result.shape = tuple(np.concatenate(([2], extra_shape)))
    if verbose:
        timer.tick("g")
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

    ##find the shortest 2 non-parallel baselines, candidate_vecs have all combinations of vectors in a summation or subtraction. Each entry is i,j, v0,v1 where Ai+Aj=(v0,v1), negative j means subtraction. Identical i,j means vector itself without add or subtract
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
    output_shape[-1] = info['nUBL']
    output = np.empty(output_shape, dtype='complex64')
    chisq = np.zeros(output_shape[:-1], dtype='float32')

    for u in range(info['nUBL']):
        blindex = info['subsetbl'][info['crossindex'][info['ublindex'][u][:,2].astype(int)]]
        ureversed = info['reversed'][info['ublindex'][u][:,2].astype(int)] == -1
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

#  _____ _               
# |_   _(_)_ __  ___ _ _ 
#   | | | | '  \/ -_) '_|
#   |_| |_|_|_|_\___|_|  

# XXX utility stuff belongs in another file
class Timer:
    '''XXX DOCSTRING'''
    def __init__(self):
        self.time = time.time()
        self.start_time = self.time
        self.last_msg = None
        self.repeat_msg = 0

    def tick(self, msg='', mute=False):
        '''XXX DOCSTRING'''
        msg = str(msg)
        t = (float(time.time() - self.time)/60.)
        m = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        if msg == self.last_msg:
            self.repeat_msg += 1
            if not mute:
                print msg + '*' + str(self.repeat_msg), "time elapsed: %f min"%t,
        else:
            self.repeat_msg = 0
            self.last_msg = msg
            if not mute:
             print msg, "Time elapsed: %f min."%t,
        if not mute:
            print "Memory usage 0: %.3fMB."%m
        sys.stdout.flush()
        self.time = time.time()
        return t, m

# XXX should be a method of Info class
def remove_one_antenna(Info,badant):
    '''XXX DOCSTRING'''
    # XXX this is never called anywhere, as far as I can tell
    info = Info.get_info()
    #nAntenna and antloc
    nAntenna = info['nAntenna']-1
    badindex = list(info['subsetant']).index(badant)     #the index of the bad antenna in previous subsetant

    subsetant = list(info['subsetant'])[:]
    subsetant.pop(badindex)      #delete the bad antenna from subsetant
    antloc = np.delete(np.array(info['antloc']),badindex,0)   #delete the bad antenna from antloc

    #ubl and nUBL
    index = 0              #to keep track of the index of ubl the loop is at
    deletelist = []
    for ubl in info['ublindex']:
        if len(ubl) > 1:
            index += 1
        elif ubl[0,0] == subsetant[badant] or ubl[0,1] == subsetant[badant] :
            deletelist.append(index)
            index += 1

    ubl = info['ubl'][:]
    ubl = np.array([ubl[i] for i in range(len(ubl)) if i not in deletelist])
    nUBL=len(ubl);

    #subsetbl and nBaseline
    goodpairs_old = [i[::-1] for i in info['bl2d']]    #the old goodpairs
    goodpairs_index = [i for i in range(len(goodpairs_old)) if badindex not in goodpairs_old[i]]       #the index of goodpairs that doesn't contain the bad antenna
    temp = np.array([goodpairs_old[i] for i in goodpairs_index])   #the new goodpairs with antenna number (all antenna)
    goodpairs = np.zeros(temp.shape)
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if temp[i,j] > badindex:
                goodpairs[i,j] = temp[i,j]-1
            else:
                goodpairs[i,j] = temp[i,j]

    subsetbl = [info['subsetbl'][i] for i in goodpairs_index]  #the new subsetbl
    nBaseline = len(subsetbl)

    counter = 0
    ubl_old2new = np.zeros([len(info['ubl'])],dtype = 'int')     #from old ubl index to new ubl index
    for i in range(len(info['ubl'])):
        if i in deletelist:
            ubl_old2new[i] = counter
        else:
            ubl_old2new[i] = counter
            counter += 1

    bltoubl = []
    for i in range(len(info['crossindex'])):
        pair = [info['subsetant'][index] for index in info['bl2d'][info['crossindex'][i]]]   #get the pair of antenna from each crossindex
        if badant in pair:
            pass
        else:
            bltoubl.append(ubl_old2new[info['bltoubl'][i]])   #append the new ubl index that doesn't have the bad antenna
    bltoubl = np.array(bltoubl)
    #################################################################################
    #reversed:   cross only bl if reversed -1, otherwise 1
    def dis(a1,a2):    #calculate the norm of the difference of two vectors
        return np.linalg.norm(np.array(a1)-np.array(a2))

    crosspair_old = []
    for p in goodpairs_old:
        if p[0]!=p[1]:
            crosspair_old.append(p)
    goodcross = []
    for i in range(len(crosspair_old)):
        if badindex not in crosspair_old[i]:
            goodcross.append(i)

    crosspair=[]
    for p in goodpairs:
        if p[0]!=p[1]:
            crosspair.append(p)

    reverse=[info['reversed'][i] for i in goodcross]
    ######################################################################################
    #reversedauto: the index of good baselines (auto included) in all baselines
    #autoindex: index of auto bls among good bls
    #crossindex: index of cross bls among good bls
    #ncross
    reversedauto = range(len(goodpairs))
    #find the autoindex and crossindex in goodpairs
    autoindex=[]
    crossindex=[]
    for i in range(len(goodpairs)):
        if goodpairs[i][0]==goodpairs[i][1]:
            autoindex.append(i)
        else:
            crossindex.append(i)
    for i in autoindex:
        reversedauto[i]=1
    for i in range(len(crossindex)):
        reversedauto[crossindex[i]]=reverse[i]
    reversedauto=np.array(reversedauto)
    autoindex=np.array(autoindex)
    crossindex=np.array(crossindex)
    ncross=len(crossindex)
    ###################################################
    #bl2d:  from 1d bl index to a pair of antenna numbers
    bl2d=[]
    for pair in goodpairs:
        bl2d.append(pair[::-1])
    bl2d=np.array(bl2d)

    ###################################################
    #ublcount:  for each ubl, the number of good cross bls corresponding to it
    countdict={}
    for bl in bltoubl:
        countdict[bl]=0

    for bl in bltoubl:
        countdict[bl]+=1

    ublcount=[]
    for i in range(nUBL):
        ublcount.append(countdict[i])
    ublcount=np.array(ublcount)

    ####################################################################################
    #ublindex:  //for each ubl, the vector<int> contains (ant1, ant2, crossbl)
    countdict={}
    for bl in bltoubl:
        countdict[bl]=[]

    for i in range(len(crosspair)):
        ant1=crosspair[i][1]
        ant2=crosspair[i][0]
        countdict[bltoubl[i]].append([ant1,ant2,i])

    ublindex=[]
    for i in range(nUBL):
        ublindex.append(countdict[i])
    #turn each list in ublindex into np array
    for i in range(len(ublindex)):
        ublindex[i]=np.array(ublindex[i])
    ublindex=np.array(ublindex)

    ###############################################################################
    #bl1dmatrix: a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
            #I suppose 99999 for bad and auto baselines?
    bl1dmatrix=99999*np.ones([nAntenna,nAntenna],dtype='int16')
    for i in range(len(crosspair)):
        bl1dmatrix[crosspair[i][1]][crosspair[i][0]]=i
        bl1dmatrix[crosspair[i][0]][crosspair[i][1]]=i

    ####################################################################################3
    #degenM:
    a=[]
    for i in range(len(antloc)):
        a.append(np.append(antloc[i],1))
    a=np.array(a)

    d=[]
    for i in range(len(ubl)):
        d.append(np.append(ubl[i],0))
    d=np.array(d)

    m1=-a.dot(la.pinv(np.transpose(a).dot(a))).dot(np.transpose(a))
    m2=d.dot(la.pinv(np.transpose(a).dot(a))).dot(np.transpose(a))
    degenM = np.append(m1,m2,axis=0)
    #####################################################################################
    #A: A matrix for logcal amplitude
    A=np.zeros([len(crosspair),nAntenna+len(ubl)])
    for i in range(len(crosspair)):
        A[i][crosspair[i][0]]=1
        A[i][crosspair[i][1]]=1
        A[i][nAntenna+bltoubl[i]]=1
    A=sps.csr_matrix(A)
    #################################################################################
    #B: B matrix for logcal phase
    B=np.zeros([len(crosspair),nAntenna+len(ubl)])
    for i in range(len(crosspair)):
        B[i][crosspair[i][0]]=reverse[i]*1
        B[i][crosspair[i][1]]=reverse[i]*-1
        B[i][nAntenna+bltoubl[i]]=1
    B=sps.csr_matrix(B)
    ############################################################################
    #create info dictionary
    info={}
    info['nAntenna']=nAntenna
    info['nUBL']=nUBL
    info['nBaseline']=nBaseline
    info['subsetant']=subsetant
    info['antloc']=antloc
    info['subsetbl']=subsetbl
    info['ubl']=ubl
    info['bltoubl']=bltoubl
    info['reversed']=reverse
    info['reversedauto']=reversedauto
    info['autoindex']=autoindex
    info['crossindex']=crossindex
    #info['ncross']=ncross
    info['bl2d']=bl2d
    info['ublcount']=ublcount
    info['ublindex']=ublindex
    info['bl1dmatrix']=bl1dmatrix
    info['degenM']=degenM
    info['A']=A
    info['B']=B
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            info['At'] = info['A'].transpose()
            info['Bt'] = info['B'].transpose()
            info['AtAi'] = la.pinv(info['At'].dot(info['A']).todense(), cond = 10**(-6))#(AtA)^-1
            info['BtBi'] = la.pinv(info['Bt'].dot(info['B']).todense(), cond = 10**(-6))#(BtB)^-1
            ##info['AtAiAt'] = info['AtAi'].dot(info['At'].todense())#(AtA)^-1At
            ##info['BtBiBt'] = info['BtBi'].dot(info['Bt'].todense())#(BtB)^-1Bt
            ##info['PA'] = info['A'].dot(info['AtAiAt'])#A(AtA)^-1At
            ##info['PB'] = info['B'].dot(info['BtBiBt'])#B(BtB)^-1Bt
            ##info['ImPA'] = sps.identity(ncross) - info['PA']#I-PA
            ##info['ImPB'] = sps.identity(ncross) - info['PB']#I-PB
    return RedundantInfo(info)
