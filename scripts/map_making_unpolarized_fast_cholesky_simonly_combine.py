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

#112, 102
mit_pix_file = np.load('/home/omniscope/data/GSM_data/absolute_calibrated_data/pixel_scheme_9725.npz')
mit_mask = mit_pix_file['valid_pix_mask']
mit_gsm = mit_pix_file['gsm']
mit_AtNisd = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/miteor_150.00MHzAtNisd_N3.00e-02_noaddA_dI_u102_t300_p9725_n32_128_b256_1000000000.000_v1.0', dtype='float64')
mit_AtNicsd = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/miteor_150.00MHzAtNicsd_N3.00e-02_noaddA_dI_u102_t300_p9725_n32_128_b256_1000000000.000_v1.0', dtype='float64')
mit_AtNiA = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/miteor_150.00MHzAtNiA_N3.00e-02_noaddA_dI_u102_t300_p9725_n32_128_b256_1000000000.000_v1.0', dtype='float64').reshape((np.sum(mit_mask), np.sum(mit_mask)))

#339, 195
mwa_pix_file = np.load('/home/omniscope/data/GSM_data/absolute_calibrated_data/pixel_scheme_9785.npz')
mwa_mask = mwa_pix_file['valid_pix_mask']
mwa_gsm = mwa_pix_file['gsm']
mwa_AtNisd = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_150.00MHzAtNisd_N2.56e-02_noaddA_dI_u195_t300_p9785_n32_128_b256_1000000000.000_v1.0', dtype='float64')
mwa_AtNicsd = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_150.00MHzAtNicsd_N2.56e-02_noaddA_dI_u195_t300_p9785_n32_128_b256_1000000000.000_v1.0', dtype='float64')
mwa_AtNiA = np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_150.00MHzAtNiA_N2.56e-02_noaddA_dI_u195_t300_p9785_n32_128_b256_1000000000.000_v1.0', dtype='float64').reshape((np.sum(mwa_mask), np.sum(mwa_mask)))

pix_scale = np.median(mwa_pix_file['sizes'])
precision = 'float64'

npix = Ashape1 = len(mit_mask)
nside = hpf.npix2nside(npix)
AtNisd = np.zeros(npix)
AtNicsd = np.zeros(npix)
AtNiA = np.zeros((npix, npix))
fake_solution = np.zeros(npix)

AtNisd[mit_mask] += mit_AtNisd
AtNicsd[mit_mask] += mit_AtNicsd
fake_solution[mit_mask] = mit_gsm
AtNiA[np.ix_(mit_mask, mit_mask)] += mit_AtNiA

AtNisd[mwa_mask] += mwa_AtNisd
AtNicsd[mwa_mask] += mwa_AtNicsd
fake_solution[mwa_mask] = mwa_gsm
AtNiA[np.ix_(mwa_mask, mwa_mask)] += mwa_AtNiA

###############
##look for best rcond
################
# def logistic(x):
#     return (1 + np.exp(-x))**-1
# start_try = -10.
# for i, p in enumerate(np.arange(start_try, start_try + 7, 1)):
#     th, ph = hpf.pix2ang(nside, range(npix), nest=True)
#     rcond = 10**p * np.ones_like(th)
#     # rcond = 10.**(-3. + p * logistic(10 * (th - PI/2)))
#     # rcond = 10.**(-3. + p * ((th - PI/2) / (PI/2)) * (th > PI/2))
#     #BEST start_th = - PI + PI * i / 7.
#     #BEST rcond = 10.**(-3. - 4. * ((th - start_th) / (PI - start_th)) * (th > start_th))
#     # start_th = PI * i / 7.
#     # rcond = 10.**(-4. - 4. * (th > start_th))
#     # rcond = 10.**(-3. + p * th / PI)
#     maxAtNiA = np.max(AtNiA)
#     AtNiA.shape = (len(AtNiA) ** 2)
#     # AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
#     # AtNiAi_path = datadir + tag + AtNiAi_filename
#     AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
#     AtNiA.shape = (Ashape1, Ashape1)
#     AtNiAi = la.inv(AtNiA)
#     # AtNiAi = sv.InverseCholeskyMatrix(AtNiA).astype(precision)
#     # del(AtNiA)
#     # AtNiAi.tofile(AtNiAi_path, overwrite=True)
#     print "regularization stength", (maxAtNiA * rcond)**-.5 / pix_scale, "median GSM", np.median(fake_solution) / pix_scale
#     print  '###########check they are 1#################'
#     print AtNiAi[0].dot(AtNiA[:, 0]), AtNiAi[-1].dot(AtNiA[:, -1])
#     print  '###################################'
#     #####apply wiener filter##############
#     print "Applying Regularized AtNiAi...",
#     sys.stdout.flush()
#     # w_solution = AtNiAi.dotv(AtNi_data)
#     w_GSM = AtNiAi.dot(AtNicsd)
#     w_sim_sol = AtNiAi.dot(AtNisd)
#     print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
#     sys.stdout.flush()
#     hpv.mollview(np.log10(rcond), nest=True, title='rcond', sub=(3, 7, i+1), coord='cg')
#     hpv.mollview(np.log10(w_GSM / pix_scale), nest=True, title='wienered GSM', sub=(3, 7, i+8), coord='cg', min=0, max=4)
#     hpv.mollview(np.log10(w_sim_sol / pix_scale), nest=True, title='wienered simulated solution', sub=(3, 7, i+15), coord='cg', min=0, max=4)
#
#     AtNiA.shape = (len(AtNiA) ** 2)
#     AtNiA[::Ashape1 + 1] -= maxAtNiA * rcond
#     AtNiA.shape = (Ashape1, Ashape1)
# plt.show()
#
#
# sys.exit(0)
def plot_IQU(solution, title, col, shape=(2,3), coord='CG', min=0, max=4, log=True, resize=True):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    if resize:
        I = solution / pix_scale
    else:
        I = solution
    plotcoordtmp = coord
    if log:
        hpv.mollview(np.log10(I), min=min, max=max, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))
    else:
        hpv.mollview(I, min=min, max=max, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))

    if col == shape[0] * shape[1]:
        plt.show()

import matplotlib
matplotlib.rcParams.update({'font.size':22})

################
###full inverse
################

rcond = 10.**-7
maxAtNiA = np.max(AtNiA)
AtNiA.shape = (len(AtNiA) ** 2)
AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
AtNiA.shape = (Ashape1, Ashape1)
timer = time.time()
AtNiAi = la.inv(AtNiA)
print "%f minutes used" % (float(time.time() - timer) / 60.)
print "regularization stength ranges between", (maxAtNiA * np.max(rcond))**-.5 / pix_scale, (maxAtNiA * np.min(rcond))**-.5 / pix_scale, "median GSM", np.median(fake_solution) / pix_scale
print  '###########check they are 1#################'
print AtNiAi[0].dot(AtNiA[:, 0]), AtNiAi[-1].dot(AtNiA[:, -1])
print  '###################################'
AtNiA.shape = (len(AtNiA) ** 2)
AtNiA[::Ashape1 + 1] -= maxAtNiA * rcond
AtNiA.shape = (Ashape1, Ashape1)

#####apply wiener filter##############
print "Applying Regularized AtNiAi...",
sys.stdout.flush()
# w_solution = AtNiAi.dotv(AtNi_data)
w_GSM = AtNiAi.dot(AtNicsd)
w_sim_sol = AtNiAi.dot(AtNisd)
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

psf = np.einsum('ij,jk->ik', AtNiAi, AtNiA)
def fwhm2(psf, verbose=False):
    spreaded = np.abs(psf) / np.max(np.abs(psf))
    fwhm_mask = spreaded >= .5
    return (np.sum(fwhm_mask) * 4 * PI / hpf.nside2npix(nside) / PI)**.5
resolution = np.array([fwhm2(pf) for pf in psf.transpose()])

rescale = np.sum(psf, axis=-1)
noise = np.sum(psf * np.transpose(AtNiAi), axis=-1)**.5 / np.abs(rescale)
result = w_sim_sol / rescale
# plot_IQU(resolution * 180/PI, 'Resolution (degree)', 1, shape=(1, 1), coord='cg', log=False, resize=False, min=2, max=5)
plot_IQU(result, 'Simulated Dirty Map (Log(K))', 1, shape=(1, 1), coord='cg', min=2)
plot_IQU(w_GSM / rescale, 'regulated GSM (Log(K))', 1, shape=(1, 1), coord='cg', min=2)
plot_IQU(noise, 'Uncertainty (Log(K))', 1, shape=(1, 1), coord='cg', min=1, max=3)
plot_IQU(np.abs(w_GSM - w_sim_sol) / np.sum(psf * np.transpose(AtNiAi), axis=-1)**.5, 'chi', 1, shape=(1, 1), coord='cg', resize=False, min=0, max=2, log=False)

###traverse rcond
results = []
noises = []
resolutions = []
for p in np.arange(-9., -4.5, 1.):
    if p != -7.:
        rcond = 10.**p
        maxAtNiA = np.max(AtNiA)
        AtNiA.shape = (len(AtNiA) ** 2)
        AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
        AtNiA.shape = (Ashape1, Ashape1)
        timer = time.time()
        AtNiAi = la.inv(AtNiA)
        print "%f minutes used" % (float(time.time() - timer) / 60.)
        print "regularization stength ranges between", (maxAtNiA * np.max(rcond))**-.5 / pix_scale, (maxAtNiA * np.min(rcond))**-.5 / pix_scale, "median GSM", np.median(fake_solution) / pix_scale
        print  '###########check they are 1#################'
        print AtNiAi[0].dot(AtNiA[:, 0]), AtNiAi[-1].dot(AtNiA[:, -1])
        print  '###################################'
        AtNiA.shape = (len(AtNiA) ** 2)
        AtNiA[::Ashape1 + 1] -= maxAtNiA * rcond
        AtNiA.shape = (Ashape1, Ashape1)


        #####apply wiener filter##############
        print "Applying Regularized AtNiAi...",
        sys.stdout.flush()
        # w_solution = AtNiAi.dotv(AtNi_data)
        w_GSM = AtNiAi.dot(AtNicsd)
        w_sim_sol = AtNiAi.dot(AtNisd)
        print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        sys.stdout.flush()

        psf = np.einsum('ij,jk->ik', AtNiAi, AtNiA)
        def fwhm2(psf, verbose=False):
            spreaded = np.abs(psf) / np.max(np.abs(psf))
            fwhm_mask = spreaded >= .5
            return (np.sum(fwhm_mask) * 4 * PI / hpf.nside2npix(nside) / PI)**.5
        resolution = np.array([fwhm2(pf) for pf in psf.transpose()])

        rescale = np.sum(psf, axis=-1)
        noise = np.sum(psf * np.transpose(AtNiAi), axis=-1)**.5 / np.abs(rescale)
        result = w_sim_sol / rescale

        results.append(result)
        noises.append(noise)
        resolutions.append(resolution)


for i, result in enumerate(np.array(results)):
    plot_IQU(result, 'Simulated Dirty Map (Log(K))', i+1, shape=(2, 4), coord='cg', min=2)
for i, noise in enumerate(np.array(noises)):
    plot_IQU(noise, 'Uncertainty (Log(K))', i+5, shape=(2, 4), coord='cg', min=1, max=3)

# #investigate smoothing gsm
# nside_standard = 128
# freq = 150.
# script_dir = os.path.dirname(os.path.realpath(__file__))
# pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
# pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
# pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
# components = np.loadtxt(script_dir + '/../data/components.dat')
# scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
# w1 = si.interp1d(components[:, 0], components[:, 2])
# w2 = si.interp1d(components[:, 0], components[:, 3])
# w3 = si.interp1d(components[:, 0], components[:, 4])
# gsm_standard = np.exp(scale_loglog(np.log(freq))) * (w1(freq) * pca1 + w2(freq) * pca2 + w3(freq) * pca3)
#
# # rotate sky map and converts to nest
# equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
# print "Rotating GSM_standard and converts to nest...",
# sys.stdout.flush()
# equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=2013.58))
# ang0, ang1 = hp.rotator.rotateDirection(equ2013_to_gal_matrix,
#                                         hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
# equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)

sys.exit(0)
#clean
bright_points = {'cyg':{'ra': '19:59:28.3', 'dec': '40:44:02'}, 'cas':{'ra': '23:23:26', 'dec': '58:48:00'}}
pt_source_range = PI / 40
smooth_scale = PI / 30
pt_source_neighborhood_range = [smooth_scale, PI / 9]
bright_pt_mask = np.zeros(Ashape1, dtype=bool)
bright_pt_neighborhood_mask = np.zeros(Ashape1, dtype=bool)
for source in bright_points.keys():
    bright_points[source]['body'] = ephem.FixedBody()
    bright_points[source]['body']._ra = bright_points[source]['ra']
    bright_points[source]['body']._dec = bright_points[source]['dec']
    theta = PI / 2 - bright_points[source]['body']._dec
    phi = bright_points[source]['body']._ra
    pt_coord = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    sky_vecs = np.array(hpf.pix2vec(nside_start, np.arange(hpf.nside2npix(nside_start)), nest=True))[:, valid_pix_mask]
    bright_pt_mask = bright_pt_mask | (la.norm(sky_vecs - pt_coord[:, None], axis=0) < pt_source_range)
    bright_pt_neighborhood_mask = bright_pt_neighborhood_mask | (la.norm(sky_vecs - pt_coord[:, None], axis=0) >= pt_source_neighborhood_range[0])
    bright_pt_neighborhood_mask = bright_pt_neighborhood_mask | (la.norm(sky_vecs - pt_coord[:, None], axis=0) <= pt_source_neighborhood_range[1])

raw_psf = psf[:, bright_pt_mask]
cleaned_result2 = np.copy(result[bright_pt_mask])
cleaned_accumulate = np.zeros_like(cleaned_result2)
# clean_stop = 10**3.5 * np.median(sizes)#2 * np.min(np.abs(cleaned_result2))
step_size = 0.02
while np.max(np.abs(cleaned_result2)) > np.median(np.abs(cleaned_result2)) * 1.1:
    clean_pix = np.argmax(np.abs(cleaned_result2))
    cleaned_accumulate[clean_pix] += step_size * cleaned_result2[clean_pix]
    cleaned_result2 -= step_size * cleaned_result2[clean_pix] * raw_psf[bright_pt_mask, clean_pix]
cleaned_result2 = result - raw_psf.dot(cleaned_accumulate)
plot_IQU(result, 'Simulated Dirty Map (Log(K))', 1, shape=(2, 1), coord='cg')
plot_IQU(cleaned_result2, 'Simulated CLEANed Map (Log(K))', 2, shape=(1, 1), coord='cg')
plot_IQU(cleaned_result2, 'Simulated CLEANed Map (Log(K))', 1, shape=(1, 1), coord='CG')