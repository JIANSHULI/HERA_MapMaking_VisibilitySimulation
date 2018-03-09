import _Bulm
import numpy as np

def compute_Bulm(Blm, L, freq, d, L1):#freq in Mhz, d in equatorial coor in meters
    if d[0] == 0 and d[1] == 0 and  d[2] == 0:
        Bulm = np.zeros((L+1, 2*L+1), dtype='complex64')
        Bulm[:L1+1,:L1+1] = Blm[:,:L1+1]
        Bulm[:L1+1,-(L1+1):] = Blm[:,-(L1+1):]
        return  Bulm
    else:
        return _Bulm.compute_Bulm(Blm, L, freq, d[0], d[1], d[2], L1)
