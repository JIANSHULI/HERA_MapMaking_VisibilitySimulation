__author__ = 'omniscope'

import numpy as np
import numpy.linalg as la
try:
    import healpy.visufunc as hpv
except:
    pass
import matplotlib.pyplot as plt

########################################
#load data
result_filename = '/mnt/data0/omniscope/polarized foregrounds/result_25+7_nside_128_smooth_6.28E-02_edge_8.73E-02_rmvcmb_1_UV0_v2.0_principal_6_step_1.00.npz'
f = np.load(result_filename)
w_nf = f['w_nf']#n_principal by frequency
x_ni = f['x_ni']#n_principal by pixel
M = f['M_for_w']#np.transpose(x_ni).dot(w_nf) = np.transpose(np.transpose(la.inv(M)).dot(x_ni)).dot(M.dot(w_nf)), see line 39
freqs = f['freqs']#GHz
n_f = len(freqs)
n_principal = len(w_nf)
########################################
#embarassing fact: I have not been able to unify the units between sub-CMB, CMB, and above_CMB frequencies. If you guys know how to put those 3 into the same unit, it'll be super helpful.
normalization = f['normalization']

################################################
#plot orthogonal results
for n in range(n_principal):
    try:
        hpv.mollview(x_ni[n], nest=True, sub=(2, n_principal, n + 1))
    except:
        print "NEED HEALPY PACKAGE FOR PLOTTING!"
    plt.subplot(2, n_principal, n_principal + n + 1)
    plt.plot(np.log10(freqs), w_nf[n])
    plt.ylim([-1, 1])
plt.show()

################################################
#try jeff's preliminary M matrix
x_ni_physical = np.transpose(la.inv(M)).dot(x_ni)
w_nf_physical = M.dot(w_nf)
#check that M does not affect the product
print np.allclose(np.transpose(x_ni).dot(w_nf), np.transpose(x_ni_physical).dot(w_nf_physical))
#plot physical results
for n in range(n_principal):
    try:
        hpv.mollview(x_ni_physical[n], nest=True, sub=(2, n_principal, n + 1))
    except:
        pass
    plt.subplot(2, n_principal, n_principal + n + 1)
    plt.plot(np.log10(freqs), w_nf_physical[n])
    plt.ylim([-1, 1])
plt.show()

######################################################
##quick example of using eigen values in w_nf to search for modes that are limited in frequency range
##as I shrink the range of frequencies, the number of non-zero eigen values decreases
eigen_values = np.zeros((n_f, n_principal))
for f_end in range(n_f):
    partial_w_nf = w_nf[:, :f_end+1]
    eigen_values[f_end], evector = la.eigh(partial_w_nf.dot(np.transpose(partial_w_nf)))
plt.subplot(1, 2, 1)
plt.imshow(eigen_values, interpolation='none')

eigen_values = np.zeros((n_f, n_principal))
for f_start in range(n_f):
    partial_w_nf = w_nf[:, f_start:]
    eigen_values[f_start], evector = la.eigh(partial_w_nf.dot(np.transpose(partial_w_nf)))
plt.subplot(1, 2, 2)
plt.imshow(eigen_values, interpolation='none')
plt.show()
