import numpy as np
import healpy as hp
import scipy.linalg as la
import os
import sys
from array import array
import simulate_visibilities.simulate_visibilities as sv



if __name__ == '__main__':
    path = sys.argv[1]
    precision = None
    verbose = False
    if len(sys.argv) >= 3:
        precision = float(sys.argv[2])
    #if len(sys.argv) >= 4:
        #verbose = (sys.argv[3] == 'True')
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
    sv.check_beam(data,precision,verbose=True)





