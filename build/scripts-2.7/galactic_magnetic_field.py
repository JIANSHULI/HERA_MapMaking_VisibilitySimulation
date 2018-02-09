__author__ = 'omniscope'
import numpy as np

###Jansson 2012 A NEW MODEL OF THE GALACTIC MAGNETIC FIELD

def transition_logistic_L(z, h, w):#eq 5
    return 1. / (1 + np.exp(-2. * (np.abs(z) - h) / w))

def disk_field_B(r, phi, i, b_list, bring, hdisk, wdisk):#section 5.1.1


def toroidal_halo_B(r, z, z0, hdisk, wdisk, Bn, Bs, rn, rs, wh):#eq6
    B = {True: Bn, False: Bs}
    rns = {True: rn, False: rs}
    return np.exp(-np.abs(z) / z0) * transition_logistic_L(z, hdisk, wdisk) * B[z>0] * (1 - transition_logistic_L(r, rns[z>0], wh))

