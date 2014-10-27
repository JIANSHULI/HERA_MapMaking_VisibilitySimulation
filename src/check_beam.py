import numpy as np
import healpy as hp
import scipy.linalg as la
import os
import sys
from array import array
##########################################
#get the appropriate nside that has the desired accuracy
##########################################
def check_beam(data, precision = None, verbose = False):
    nside = int((len(data)/12)**0.5)

    nsidelist = []
    n = 1
    while n <= nside:
        nsidelist.append(n)
        n = 2*n
         
    truncatemaps = {}
    for n in nsidelist:
        beam_alm = hp.sphtfunc.map2alm(data,lmax = 3*n-1,iter=10)
        truncatemaps[n] = hp.sphtfunc.alm2map(beam_alm,nside,verbose=False)
    
    errorlist = {}
    for n in nsidelist:
        diff = truncatemaps[n] - data
        errorlist[n] = [la.norm(diff)/la.norm(data),la.norm(diff)/(12*nside**2)**0.5/max(data),max(abs(diff))/max(data)]
        if verbose:
            print [n,nside]
            print ["%.5f" %i for i in errorlist[n]]
    
    if type(precision) == float:
        for n in nsidelist:
            if errorlist[n][0] < precision and errorlist[n][1] < precision:# and errorlist[n][2] < precision:
                if verbose:
                    print 'nside = %.d has the desired precision' %n
                return n
        print 'Need larger nside than the input to have the desired precision'
    return errorlist



if __name__ == '__main__':
    path = sys.argv[1]
    precision = None
    verbose = False
    if len(sys.argv) >= 3:
        precision = float(sys.argv[2])
    if len(sys.argv) >= 4:
        verbose = (sys.argv[3] == 'True')
    #check if file exists
    if os.path.isfile(path) == False:
        raise Exception('File does not exist')
    if path[-3:] == 'bin':
        with open(path) as f:
            farray = array('f')
            farray.fromstring(f.read())
            rawdata = np.array(farray)
    elif path[-3:] == 'txt':
        with open(path) as f:
            rawdata = np.loadtxt(f)
    data = rawdata.flatten()
    test = check_beam(data,precision,verbose)





