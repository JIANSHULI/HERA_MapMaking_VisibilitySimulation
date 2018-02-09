import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, datetime, warnings
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob

PI = np.pi
TPI = PI * 2


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
        raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i" % (str(A.ndim), str(C.ndim), str(Ni.ndim)))

    print "Estimated time", (A.shape[0] / 2000.) * (A.shape[1] / 9000.)**2, "minutes"
    sys.stdout.flush()

    chunk = len(C) / nchunk
    for i in range(nchunk):
        C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
    if chunk * nchunk < len(C):
        C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)


thresh = .5#2.#.03125#
valid_pix_thresh = 0
nside_start = 64
nside_standard = 64#256
freq = 150.
script_dir = os.path.dirname(os.path.realpath(__file__))
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
plotcoord = 'CG'
################################################
#####################GSM###########################
#############################################
pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
components = np.loadtxt(script_dir + '/../data/components.dat')
scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
w1 = si.interp1d(components[:, 0], components[:, 2])
w2 = si.interp1d(components[:, 0], components[:, 3])
w3 = si.interp1d(components[:, 0], components[:, 4])
gsm_standard = np.exp(scale_loglog(np.log(freq))) * (w1(freq) * pca1 + w2(freq) * pca2 + w3(freq) * pca3)

# rotate sky map and converts to nest
equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
print "Rotating GSM_standard and converts to nest...",
sys.stdout.flush()
equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=2013.58))
ang0, ang1 = hp.rotator.rotateDirection(equ2013_to_gal_matrix,
                                        hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
print "done."
sys.stdout.flush()


nside_distribution = np.zeros(12 * nside_standard ** 2)
final_index = np.zeros(12 * nside_standard ** 2, dtype=int)
thetas, phis, sizes = [], [], []
abs_thresh = np.mean(equatorial_GSM_standard) * thresh
pixelize(equatorial_GSM_standard, nside_distribution, nside_standard, nside_start, abs_thresh,
         final_index, thetas, phis, sizes)
npix = len(thetas)
valid_pix_mask = hpf.get_interp_val(equatorial_GSM_standard, thetas, phis, nest=True) > valid_pix_thresh * max(equatorial_GSM_standard)
valid_npix = np.sum(valid_pix_mask)
print '>>>>>>VALID NPIX =', valid_npix

fake_solution_map = np.zeros_like(thetas)
for i in range(len(fake_solution_map)):
    fake_solution_map[i] = np.sum(equatorial_GSM_standard[final_index == i])
fake_solution_map = fake_solution_map[valid_pix_mask]
sizes = np.array(sizes)[valid_pix_mask]
thetas = np.array(thetas)[valid_pix_mask]
phis = np.array(phis)[valid_pix_mask]

instruments = ['MITEoR', 'MWA', 'PAPER']
lats = [45.2977, -26.703319, -30.72153]#['-30:43:17.5', '21:25:41.9']

np.savez(datadir + 'pixel_scheme_%i.npz'%valid_npix, gsm=fake_solution_map, thetas=thetas, phis=phis, sizes=sizes, nside_distribution=nside_distribution, final_index=final_index, n_fullsky_pix=npix, valid_pix_mask=valid_pix_mask, thresh=thresh)#thresh is in there for idiotic reason  due to unneccessary inclusion of thresh in A filename

fake_solution = np.copy(fake_solution_map)

def sol2map(sol):
    solx = sol[:valid_npix]
    full_sol = np.zeros(npix)
    full_sol[valid_pix_mask] = solx / sizes
    return full_sol[final_index]

##################################################################
####################################sanity check########################
###############################################################
# npix = 0
# for i in nside_distribution:
# npix += i**2/nside_standard**2
# print npix, len(thetas)

stds = np.std((equatorial_GSM_standard).reshape(12 * nside_standard ** 2 / 4, 4), axis=1)

##################################################################
####################################plotting########################
###############################################################
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    hpv.mollview(np.log10(equatorial_GSM_standard), min=0, max=4, coord=plotcoord, title='GSM', nest=True)
    hpv.mollview(np.log10(sol2map(fake_solution)), min=0, max=4, coord=plotcoord,
                 title='GSM gridded', nest=True)
    hpv.mollview(np.log10(stds / abs_thresh), min=np.log10(thresh) - 3, max=3, coord=plotcoord, title='std',
                 nest=True)
    hpv.mollview(np.log2(nside_distribution), min=np.log2(nside_start), max=np.log2(nside_standard),
                 coord=plotcoord,
                 title='count %i %.3f' % (len(thetas), float(len(thetas)) / (12 * nside_standard ** 2)), nest=True)


for instrument, lat in zip(instruments, lats):
    above_horizon_mask = np.abs(thetas - (PI / 2 - lat * PI / 180.)) < PI / 2
    sub_valid_npix = np.sum(above_horizon_mask)
    sub_valid_pix_mask = np.copy(valid_pix_mask)
    sub_valid_pix_mask[sub_valid_pix_mask] = above_horizon_mask
    np.savez(datadir + 'pixel_scheme_%i.npz'%sub_valid_npix, gsm=fake_solution_map[above_horizon_mask], thetas=thetas[above_horizon_mask], phis=phis[above_horizon_mask], sizes=sizes[above_horizon_mask], nside_distribution=nside_distribution, final_index=final_index, n_fullsky_pix=npix, valid_pix_mask=sub_valid_pix_mask, thresh=thresh, parent_valid_npix = valid_npix, child_mask=above_horizon_mask)#thresh is in there for idiotic reason  due to unneccessary inclusion of thresh in A filename

    def sol2map(sol):
        solx = sol[:sub_valid_npix]
        full_sol = np.zeros(npix)
        full_sol[sub_valid_pix_mask] = solx / sizes[sub_valid_pix_mask]
        return full_sol[final_index]

    hpv.mollview(np.log10(sol2map(fake_solution_map[above_horizon_mask])), min=0, max=4, coord=plotcoord,
                 title='GSM gridded %s NPIX%i'%(instrument, sub_valid_npix), nest=True)
plt.show()