import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, datetime, warnings
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob

PI = np.pi
TPI = np.pi * 2


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



nside_start = 32
nside_beamweight = 16
nside_standard = 128
bnside = 256
plotcoord = 'CG'
thresh = .5
valid_pix_thresh = 1e-3
# S_scale = 2
# S_thresh = 1000#Kelvin
# S_type = 'gsm%irm%i'%(S_scale,S_thresh)
S_type = 'dySP_lowaddlimit_lowpol'  # dynamic S polarized [[.25,0,0,.25], [0,p,0,0], [0,0,p,0], [.25,0,0,.25]]. lowpol:polfrac=0.01; addlimit:additive same level as max data; lowaddlimit: 10% of max data
remove_additive = False

lat_degree = 45.2977
C = 299.792458
kB = 1.3806488 * 1.e-23
script_dir = os.path.dirname(os.path.realpath(__file__))

plot_pixelization = True
plot_projection = True
plot_data_error = True

force_recompute = False
force_recompute_AtNiAi_eig = False
force_recompute_AtNiAi = False
force_recompute_S = False
force_recompute_SEi = False

####################################################
################data file and load beam##############
####################################################
tag = "q1AL_10_abscal"#"q3AL_5_abscal"  # L stands for lenient in flagging
datatag = '_2016_01_20_avg'
vartag = '_2016_01_20_avg'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
A_version = 1.0
nf = 1
data_filename = glob.glob(datadir + tag + '_xx_*_*' + datatag)[0]
nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('xx_')[-1]
nt = int(nt_nUBL.split('_')[0])
nUBL = int(nt_nUBL.split('_')[1])


# deal with beam: create a callable function of the form y(freq) in MHz and returns npix by 4
freqs = range(110, 200, 10)
local_beam = si.interp1d(freqs, np.concatenate([np.fromfile(
    '/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
    (len(freqs), 12 * bnside ** 2, 2)) for p in ['x', 'y']], axis=-1).transpose(0, 2, 1), axis=0)

A = {}

for p in ['x', 'y']:
    pol = p + p

    # tf file, t in lst hours
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf' % (p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt, nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]

    # tf mask file, 0 means flagged bad data
    try:
        tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm' % (p, p, nt, nf)
        tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt, nf))
        tmask = np.array(tfmlist[:, 0].astype('bool'))
        # print tmask
    except:
        print "No mask file found"
        tmask = np.ones_like(tlist).astype(bool)
    # print freq, tlist

    # ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include, longest baseline is %i wavelengths" % (
    len(ubls), np.max(np.linalg.norm(ubls, axis=1)) / (C / freq))

    A_path = datadir + tag + '_%s%s_%i_%i.A' % (p, p, len(tlist) * len(ubls), 12 * nside_beamweight ** 2)

    if os.path.isfile(A_path) and not force_recompute:
        print "Reading A matrix from %s" % A_path
        sys.stdout.flush()
        A[p] = np.fromfile(A_path, dtype='complex64').reshape((len(ubls), len(tlist), 12 * nside_beamweight ** 2))[
               :, tmask].reshape((len(ubls) * len(tlist[tmask]), 12 * nside_beamweight ** 2))
    else:
        # beam
        if p == 'x':
            beam_healpix = abs(local_beam(freq)[0]) ** 2 + abs(local_beam(freq)[1]) ** 2
        elif p == 'y':
            beam_healpix = abs(local_beam(freq)[2]) ** 2 + abs(local_beam(freq)[3]) ** 2
        # hpv.mollview(beam_healpix, title='beam %s'%p)
        # plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
        beam_heal_equ = np.array(
            sv.rotate_healpixmap(beam_healpix, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0]))
        print "Computing A matrix for %s pol..." % p
        sys.stdout.flush()
        timer = time.time()
        A[p] = np.empty((len(tlist) * len(ubls), 12 * nside_beamweight ** 2), dtype='complex64')
        for i in range(12 * nside_beamweight ** 2):
            dec, ra = hpf.pix2ang(nside_beamweight, i)  # gives theta phi
            dec = np.pi / 2 - dec
            print "\r%.1f%% completed, %f minutes left" % (100. * float(i) / (12. * nside_beamweight ** 2),
                                                           (12. * nside_beamweight ** 2 - i) / (i + 1) * (
                                                           float(time.time() - timer) / 60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array(
                [vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ=beam_heal_equ, tlist=tlist) for d
                 in ubls]).flatten()

        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A[p].tofile(A_path)
        A[p] = A[p].reshape((len(ubls), len(tlist), 12 * nside_beamweight ** 2))[:, tmask].reshape(
            (len(ubls) * len(tlist[tmask]), 12 * nside_beamweight ** 2))

####################################################
###beam weights using an equal pixel A matrix######
#################################################
print "Computing beam weight...",
sys.stdout.flush()
beam_weight = ((la.norm(A['x'], axis=0) ** 2 + la.norm(A['y'], axis=0) ** 2) ** .5)[
    hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
beam_weight = beam_weight / np.mean(beam_weight)
beam_weight = np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
print "done."
sys.stdout.flush()

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


###UBL####

ubls = {}
for p in ['x', 'y']:
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
#manually filter UBLs
used_common_ubls = common_ubls#[np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10

ubl_index = {}  # stored index in each pol's ubl for the common ubls
for p in ['x', 'y']:
    ubl_index[p] = np.zeros(len(used_common_ubls), dtype='int')
    for i, u in enumerate(used_common_ubls):
        if u in ubls[p]:
            ubl_index[p][i] = np.argmin(la.norm(ubls[p] - u, axis=-1)) + 1
        elif -u in ubls[p]:
            ubl_index[p][i] = - np.argmin(la.norm(ubls[p] + u, axis=-1)) - 1
        else:
            raise Exception('Logical Error')

########################################################################
########################processing dynamic pixelization######################
########################################################################

nside_distribution = np.zeros(12 * nside_standard ** 2)
final_index = np.zeros(12 * nside_standard ** 2)
thetas, phis, sizes = [], [], []
abs_thresh = np.mean(equatorial_GSM_standard * beam_weight) * thresh
pixelize(equatorial_GSM_standard * beam_weight, nside_distribution, nside_standard, nside_start, abs_thresh,
         final_index, thetas, phis, sizes)
npix = len(thetas)
valid_pix_mask = hpf.get_interp_val(beam_weight, thetas, phis, nest=True) > valid_pix_thresh * max(beam_weight)
valid_npix = np.sum(valid_pix_mask)
fake_solution_map = (hpf.get_interp_val(equatorial_GSM_standard, thetas, phis, nest=True) * sizes)[valid_pix_mask]
fake_solution = np.concatenate((fake_solution_map, np.zeros(2 * len(used_common_ubls)),
                                np.zeros_like(fake_solution_map), np.zeros(2 * len(used_common_ubls)),
                                np.zeros_like(fake_solution_map), np.zeros(2 * len(used_common_ubls)),
                                np.zeros_like(fake_solution_map), np.zeros(2 * len(used_common_ubls))))

sizes = np.concatenate((np.array(sizes)[valid_pix_mask], np.array(sizes)[valid_pix_mask],
                        np.array(sizes)[valid_pix_mask], np.array(sizes)[valid_pix_mask]))


def sol2map(sol):
    solx = sol.reshape((4, valid_npix + 2 * len(used_common_ubls)))[:, :valid_npix].flatten()
    final_index4 = np.concatenate(
        (final_index, final_index + npix, final_index + npix * 2, final_index + npix * 3)).astype(int)
    full_sol = np.zeros(4 * npix)
    full_sol[np.concatenate((valid_pix_mask, valid_pix_mask, valid_pix_mask, valid_pix_mask))] = solx / sizes
    return full_sol[final_index4]

def sol2additive(sol):
    return sol.reshape((4, valid_npix + 2 * len(used_common_ubls)))[:, valid_npix:].reshape(4, len(used_common_ubls), 2)


# final_index_filename = datadir + tag + '_%i.dyind%i_%.3f'%(nside_standard, npix, thresh)
# final_index.astype('float32').tofile(final_index_filename)
# sizes_filename = final_index_filename.replace('dyind', "dysiz")
# np.array(sizes).astype('float32').tofile(sizes_filename)
if plot_pixelization:
    ##################################################################
    ####################################sanity check########################
    ###############################################################
    # npix = 0
    # for i in nside_distribution:
    # npix += i**2/nside_standard**2
    # print npix, len(thetas)

    stds = np.std((equatorial_GSM_standard * beam_weight).reshape(12 * nside_standard ** 2 / 4, 4), axis=1)

    ##################################################################
    ####################################plotting########################
    ###############################################################
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        hpv.mollview(beam_weight, min=0, max=4, coord=plotcoord, title='beam', nest=True)
        hpv.mollview(np.log10(equatorial_GSM_standard), min=0, max=4, coord=plotcoord, title='GSM', nest=True)
        hpv.mollview(np.log10(sol2map(fake_solution)[:len(equatorial_GSM_standard)]), min=0, max=4, coord=plotcoord,
                     title='GSM gridded', nest=True)
        hpv.mollview(np.log10(stds / abs_thresh), min=np.log10(thresh) - 3, max=3, coord=plotcoord, title='std',
                     nest=True)
        hpv.mollview(np.log2(nside_distribution), min=np.log2(nside_start), max=np.log2(nside_standard),
                     coord=plotcoord,
                     title='count %i %.3f' % (len(thetas), float(len(thetas)) / (12 * nside_standard ** 2)), nest=True)
    plt.show()


##################################################################
####################compute dynamic A matrix########################
###############################################################


A_tag = 'A_dpcIQUV'
A_filename = A_tag + '_u%i_t%i_p%i_n%i_%i_b%i_%.3f_v%.1f' % (len(used_common_ubls), len(tlist), valid_npix, nside_start, nside_standard, bnside, thresh, A_version)
A_path = datadir + tag + A_filename


def get_A():
    if os.path.isfile(A_path) and not force_recompute:
        print "Reading A matrix from %s" % A_path
        sys.stdout.flush()
        A = np.fromfile(A_path, dtype='complex64').reshape((len(used_common_ubls), 4, len(tlist), 4, valid_npix + 2 * len(used_common_ubls)))
    else:
        # beam
        beam_healpix = local_beam(freq)
        # hpv.mollview(beam_healpix, title='beam %s'%p)
        # plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
        beam_heal_equ = np.array(
            [sv.rotate_healpixmap(beam_healpixi, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for
             beam_healpixi in local_beam(freq)])
        print "Computing A matrix..."
        sys.stdout.flush()
        A = np.empty((len(used_common_ubls), 4 * len(tlist), 4, valid_npix + 2 * len(used_common_ubls)), dtype='complex64')
        timer = time.time()
        for n, i in enumerate(np.arange(npix)[valid_pix_mask]):
            ra = phis[i]
            dec = np.pi / 2 - thetas[i]
            print "\r%.1f%% completed, %f minutes left" % (
            100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
            sys.stdout.flush()

            A[..., n] = vs.calculate_pol_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ,
                                                                tlist=tlist).dot(
                [[.5, .5, 0, 0], [0, 0, .5, .5j], [0, 0, .5, -.5j], [.5, -.5, 0, 0]])

        for i in range(len(used_common_ubls)):
            for p in range(4):
                A[i, p * len(tlist): (p + 1) * len(tlist), p, valid_npix + 2 * i] = 1.
                A[i, p * len(tlist): (p + 1) * len(tlist), p, valid_npix + 2 * i + 1] = 1.j

        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A.tofile(A_path)
        A.shape = (len(used_common_ubls), 4, len(tlist), 4, valid_npix + 2 * len(used_common_ubls))
    tmask = np.ones_like(tlist).astype(bool)
    for p in ['x', 'y']:
        # tf mask file, 0 means flagged bad data
        try:
            tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm' % (p, p, nt, nf)
            tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt, nf))
            tmask = tmask & np.array(tfmlist[:, 0].astype('bool'))
            # print tmask
        except:
            print "No mask file found"
            # print freq, tlist
    # Merge A
    A.shape = (len(used_common_ubls) * 4 * len(tlist[tmask]), 4 * A.shape[-1])
    try:
        return np.concatenate((np.real(A), np.imag(A)))
    except MemoryError:
        print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
        sys.stdout.flush()
        Ashape = list(A.shape)
        Ashape[0] = Ashape[0] * 2
        np.real(A).tofile(A_path + 'tmpre')
        np.imag(A).tofile(A_path + 'tmpim')
        del (A)
        os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))

        os.system("rm %s" % (A_path + 'tmpim'))
        A = np.fromfile(A_path + 'tmpre', dtype='float32').reshape(Ashape)
        os.system("rm %s" % (A_path + 'tmpre'))
        print "done."
        sys.stdout.flush()
        return A


A = get_A()
Ashape0, Ashape1 = A.shape
# Compute autocorr
if remove_additive:
    # beam_healpix = local_beam(freq)
    # vs = sv.Visibility_Simulator()
    # vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
    # beam_heal_equ = np.array(
    #     [sv.rotate_healpixmap(beam_healpixi, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for beam_healpixi in
    #      local_beam(freq)])
    # print "Computing autocorr..."
    # sys.stdout.flush()
    # timer = time.time()
    # autocorr = np.empty((4 * len(tlist), 4, valid_npix), dtype='complex64')

    # for n, i in enumerate(np.arange(npix)[valid_pix_mask]):
    #     ra = phis[i]
    #     dec = np.pi / 2 - thetas[i]
    #     print "\r%.1f%% completed, %f minutes left" % (
    #     100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
    #     sys.stdout.flush()
    #
    #     autocorr[..., n] = \
    #     vs.calculate_pol_pointsource_visibility(ra, dec, [[0, 0, 0]], freq, beam_heal_equ=beam_heal_equ, tlist=tlist)[
    #         0].dot([[.5, .5, 0, 0], [0, 0, .5, .5j], [0, 0, .5, -.5j], [.5, -.5, 0, 0]])
    #
    # print "%f minutes used" % (float(time.time() - timer) / 60.)
    # sys.stdout.flush()
    # todo use autocorr rather than constant as removal term
    # autocorr_vis = autocorr.reshape(4 * len(tlist), 4 * valid_npix).dot(fake_solution_map).reshape(4, len(tlist))
    autocorr_vis = np.ones((4, len(tlist)))

data = {}
Ni = {}
data_shape = {}
ubl_sort = {}
for p in ['x', 'y']:
    for p2 in ['x', 'y']:
        pol = p + p2
        # tf file
        tf_filename = datadir + tag + '_%s%s_%i_%i.tf' % (p, p2, nt, nf)
        tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt, nf))
        tlist = np.real(tflist[:, 0])

        # ubl file
        ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p2, nUBL, 3)
        ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
        print "%i UBLs to include, longest baseline is %i wavelengths" % (
        len(used_common_ubls), np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / freq))


        # get Ni (1/variance) and data
        var_filename = datadir + tag + '_%s%s_%i_%i' % (p, p2, nt, nUBL) + vartag + '.var'
        Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[
                            abs(ubl_index[p]) - 1].flatten() * (
                        1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * np.pi / (12 * nside_standard ** 2))) ** 2)
        data_filename = datadir + tag + '_%s%s_%i_%i' % (p, p2, nt, nUBL) + datatag
        data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose()[
            abs(ubl_index[p]) - 1]
        data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
        data[pol] = (data[pol].flatten() * 1.e-26 * (C / freq) ** 2 / 2 / kB / (
        4 * np.pi / (12 * nside_standard ** 2))).conjugate()  # there's a conjugate convention difference
        data_shape[pol] = (len(used_common_ubls), np.sum(tmask))
        ubl_sort[p] = np.argsort(la.norm(used_common_ubls, axis=1))
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

# Merge data
original_data = np.array([data['xx'], data['xy'], data['yx'], data['yy']]).reshape(
    [4] + list(data_shape['xx'])).transpose((1, 0, 2))
data = np.array([data['xx'], data['xy'], data['yx'], data['yy']]).reshape([4] + list(data_shape['xx'])).transpose(
    (1, 0, 2)).flatten()
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
Ni = np.concatenate((Ni['xx'], Ni['xy'], Ni['yx'], Ni['yy'])).reshape([4] + list(data_shape['xx'])).transpose(
    (1, 0, 2)).flatten()
Ni = np.concatenate((Ni * 2, Ni * 2))
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()
# simulate visibilities

# clean_sim_data = np.array([Aiter.dot(fake_solution) for Aiter in A])
clean_sim_data = A.dot(fake_solution.astype(A.dtype))

def get_complex_data(flat_real_data, nubl, nt):
    if len(flat_real_data) != 2 * nubl * 4 * nt:
        raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i."%(len(flat_real_data), nubl, nt, 2 * nubl * 4 * nt))

    flat_real_data.shape = (2, nubl, 4, nt)
    result = flat_real_data[0] + 1.j * flat_real_data[1]
    flat_real_data.shape = 2 * nubl * 4 * nt
    return result

def get_vis_normalization(data, clean_sim_data):
    a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:, :, [0, 3]], axis=0).flatten()
    b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:, :, [0, 3]],
                       axis=0).flatten()
    return a.dot(b) / b.dot(b)


vis_normalization = get_vis_normalization(data, clean_sim_data)
print "Normalization from visibilities", vis_normalization
diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
diff_data = diff_data[0] + 1j * diff_data[1]
diff_norm = {}
diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis=1)
diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis=1)
data_norm = {}
data_norm['x'] = la.norm(data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[0, :, 0] + 1.j * data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[1, :, 0], axis=-1)
data_norm['y'] = la.norm(data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[0, :, 3] + 1.j * data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[1, :, 3], axis=-1)

if plot_data_error:
    plt.subplot(7, 1, 1)
    plt.plot((diff_norm['x']/data_norm['x'])[ubl_sort['x']])
    plt.plot((diff_norm['y']/data_norm['y'])[ubl_sort['y']])


if remove_additive:
    niter = 0

    additive = 0
    raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])
    while niter == 0 or (abs(vis_normalization - get_vis_normalization(data, clean_sim_data)) > 1e-2 and niter < 20):
        niter += 1
        vis_normalization = get_vis_normalization(data, clean_sim_data)
        print "Normalization from visibilities", vis_normalization
        diff_data = (clean_sim_data * vis_normalization - data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])
        diff_data = diff_data[0] + 1j * diff_data[1]
        diff_norm = {}
        diff_norm['x'] = la.norm(diff_data[:, 0], axis=1)
        diff_norm['y'] = la.norm(diff_data[:, 3], axis=1)

        additive_inc = np.zeros_like(diff_data)
        for p in range(4):
            if p == 0 or p == 3:
                additive_inc[:, p] = np.outer(
                    autocorr_vis[p].dot(diff_data[:, p].transpose()) / np.sum(autocorr_vis[p] ** 2), autocorr_vis[p])
            else:
                additive_inc[:, p] = np.outer(
                    (autocorr_vis[0] + autocorr_vis[3]).dot(diff_data[:, p].transpose()) / np.sum(
                        (autocorr_vis[0] + autocorr_vis[3]) ** 2), (autocorr_vis[0] + autocorr_vis[3]))

        additive = additive + additive_inc
        data = data + np.concatenate((np.real(additive_inc.flatten()), np.imag(additive_inc.flatten())))

    if plot_data_error:
        vis_normalization = get_vis_normalization(data, clean_sim_data)
        print "Normalization from visibilities", vis_normalization
        diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
        diff_data = diff_data[0] + 1j * diff_data[1]
        diff_norm = {}
        diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis=1)
        diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis=1)
        plt.plot((diff_norm['x']/data_norm['x'])[ubl_sort['x']])
        plt.plot((diff_norm['y']/data_norm['y'])[ubl_sort['y']])

if plot_data_error:
    qaz_model = (clean_sim_data * vis_normalization).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])
    qaz_data = data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])
    plt.subplot(7, 1, 2)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][0], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][0], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][0], 0])
    plt.subplot(7, 1, 3)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.subplot(7, 1, 4)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][-1], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][-1], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][-1], 0])
    plt.subplot(7, 1, 5)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][0], 1])
    plt.plot(qaz_data[1, ubl_sort['x'][0], 1])
    plt.plot(qaz_model[1, ubl_sort['x'][0], 1])
    plt.subplot(7, 1, 6)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.plot(qaz_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.plot(qaz_model[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.subplot(7, 1, 7)
    if remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][-1], 1])
    plt.plot(qaz_data[1, ubl_sort['x'][-1], 1])
    plt.plot(qaz_model[1, ubl_sort['x'][-1], 1])
    plt.show()


# vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
# print "Normalization from visibilities", vis_normalization
# diff_data = (clean_sim_data * vis_normalization - data)
# diff_norm = {}
# diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
# diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)

# if plot_data_error:
# plt.plot(diff_norm['x'][ubl_sort['x']], label='original x error')
# plt.plot(diff_norm['y'][ubl_sort['y']], label='original y error')

# if remove_additive:
# niter = 0
# additive = {'x':0, 'y':0}
# additive_inc = {'x':0, 'y':0}
# while niter == 0 or (abs(vis_normalization - np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))) > 1e-2 and niter < 20):
# niter += 1
# vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
# print "Normalization from visibilities", vis_normalization
# diff_data = clean_sim_data * vis_normalization - data
# diff_norm = {}
# diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
# diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)
# additive_inc['x'] = np.repeat(np.mean(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1, keepdims = True), data_shape['x'][1], axis = 1)
# additive_inc['y'] = np.repeat(np.mean(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1, keepdims = True), data_shape['y'][1], axis = 1)
# additive['x'] = additive['x'] + additive_inc['x']
# additive['y'] = additive['y'] + additive_inc['y']
# data = data + np.concatenate((additive_inc['x'].flatten(), additive_inc['y'].flatten()))

# if plot_data_error:
# vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
# print "Normalization from visibilities", vis_normalization
# diff_data = clean_sim_data * vis_normalization - data
# diff_norm = {}
# diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
# diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)
# plt.plot(diff_norm['x'][ubl_sort['x']], label='new x error')
# plt.plot(diff_norm['y'][ubl_sort['y']], label='new y error')
# plt.legend();plt.show()

##renormalize the model
fake_solution = fake_solution * vis_normalization
clean_sim_data = clean_sim_data * vis_normalization
sim_data = clean_sim_data + np.random.randn(len(data)) / Ni ** .5


# compute AtNi.y
AtNi_data = np.transpose(A).dot((data * Ni).astype(A.dtype))
AtNi_sim_data = np.transpose(A).dot((sim_data * Ni).astype(A.dtype))
AtNi_clean_sim_data = np.transpose(A).dot((clean_sim_data * Ni).astype(A.dtype))

# compute S
print "computing S...",
sys.stdout.flush()
timer = time.time()

if 'lowpol' in S_type:
    pol_frac = .01  # assuming QQ=UU=pol_frac*II
elif 'nopol' in S_type:
    pol_frac = 1e-6  # assuming QQ=UU=pol_frac*II
else:
    pol_frac = .4  # assuming QQ=UU=pol_frac*II

v_pol_frac = 1e-6
S = np.zeros((Ashape1, Ashape1), dtype='float32')
for i in range(valid_npix):
    S[i::Ashape1/4, i::Ashape1/4] = np.array([[1, 0, 0, 0], [0, pol_frac, 0, 0], [0, 0, pol_frac, 0], [0, 0, 0, v_pol_frac]]) * \
                                      fake_solution_map[i] ** 2  # np.array([[1+pol_frac,0,0,1-pol_frac],[0,pol_frac,pol_frac,0],[0,pol_frac,pol_frac,0],[1-pol_frac,0,0,1+pol_frac]]) / 4 * (2*sim_x_clean[i])**2

data_max = np.transpose(np.percentile(np.abs(data.reshape((2, len(used_common_ubls), 4, len(tlist)))), 95, axis=-1), (2, 1, 0)).reshape((4, len(used_common_ubls) * 2))
for i in range(4):
    start = i * Ashape1 / 4 + valid_npix
    end = (i + 1) * Ashape1 / 4
    S[start:end, start:end] = np.eye(len(used_common_ubls) * 2) * data_max[i]**2 / 100.

print "Done."
print "%f minutes used" % (float(time.time() - timer) / 60.)
sys.stdout.flush()



# compute (AtNiA+Si)i eigensystems
precision = 'float64'
AtNiAi_tag = 'AtNiASii'
AtNiAi_version = 0.1
rcond = .5e-3
AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_v%.1f'%(S_type, np.log10(rcond), AtNiAi_version) + A_filename
AtNiAi_path = datadir + tag + AtNiAi_filename

if os.path.isfile(AtNiAi_path) and not force_recompute_AtNiAi and not force_recompute and not force_recompute_S:
    print "Reading Regularized AtNiAi...",
    sys.stdout.flush()
    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S), precision)
else:
    AtNiA_tag = 'AtNiA'
    AtNiA_filename = AtNiA_tag + A_filename
    AtNiA_path = datadir + tag + AtNiA_filename
    if os.path.isfile(AtNiA_path) and not force_recompute:
        print "Reading AtNiA...",
        sys.stdout.flush()
        AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
    else:
        print "Allocating AtNiA..."
        sys.stdout.flush()
        timer = time.time()
        AtNiA = np.zeros((A.shape[1], A.shape[1]), dtype=precision)
        print "Computing AtNiA...", datetime.datetime.now()
        sys.stdout.flush()
        ATNIA(A, Ni, AtNiA)
        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        AtNiA.tofile(AtNiA_path)
    del (A)

    print "Computing Regularized AtNiAi...",
    sys.stdout.flush()
    timer = time.time()
    if la.norm(S) != la.norm(np.diagonal(S)):
        raise Exception("Non-diagonal S not supported yet")
    AtNiAi = sv.InverseCholeskyMatrix(np.diag(1./np.diagonal(S)) + AtNiA + np.eye(S.shape[0]) * np.max(AtNiA) * rcond).astype(precision)
    AtNiAi.tofile(AtNiAi_path, overwrite=True)
    print "%f minutes used" % (float(time.time() - timer) / 60.)


#####apply wiener filter##############
print "Applying Regularized AtNiAi...",
sys.stdout.flush()
w_solution = AtNiAi.dotv(AtNi_data)
w_GSM = AtNiAi.dotv(AtNi_clean_sim_data)
w_sim_sol = AtNiAi.dotv(AtNi_sim_data)
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

del (AtNiAi)
A = get_A()
best_fit = A.dot(w_solution.astype(A.dtype))
best_fit_no_additive = np.sum((A * (w_solution.astype(A.dtype))).reshape((Ashape0, 4, Ashape1/4))[..., :valid_npix].reshape((Ashape0, 4 * valid_npix)), axis=-1)

if plot_data_error:
    us = ubl_sort['x'][::len(ubl_sort['x'])/6]
    best_fit.shape = (2, data_shape['xx'][0], 4, data_shape['xx'][1])
    best_fit_no_additive.shape = (2, data_shape['xx'][0], 4, data_shape['xx'][1])
    ri = 1
    for p in range(4):
        for nu, u in enumerate(us):

            plt.subplot(4, len(us), len(us) * p + nu + 1)

            plt.plot(qaz_data[ri, u, p])
            plt.plot(qaz_model[ri, u, p])
            plt.plot(best_fit[ri, u, p])
            plt.plot(best_fit_no_additive[ri, u, p])
            plt.plot(np.ones_like(qaz_data[ri, u, p]) * sol2additive(w_solution)[p, u, ri])
    plt.show()

def plot_IQU(solution, title, col, ncol=6, coord='C'):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    IQUV = sol2map(solution)
    IQUV.shape = (4, IQUV.shape[0] / 4)
    I = IQUV[0]
    Q = IQUV[1]
    U = IQUV[2]
    V = IQUV[3]
    pangle = 180 * np.arctan2(Q, U) / 2 / PI
    plotcoordtmp = coord
    hpv.mollview(np.log10(I), min=0, max=4, coord=plotcoordtmp, title=title, nest=True, sub=(4, ncol, col))

    hpv.mollview((Q ** 2 + U ** 2) ** .5 / I, min=0, max=1, coord=plotcoordtmp, title=title, nest=True,
                 sub=(4, ncol, ncol + col))
    from matplotlib import cm
    cool_cmap = cm.hsv
    cool_cmap.set_under("w")  # sets background to white
    hpv.mollview(pangle, min=-90, max=90, coord=plotcoordtmp, title=title, nest=True, sub=(4, ncol, 2 * ncol + col),
                 cmap=cool_cmap)

    hpv.mollview(np.arcsinh(V) / np.log(10), min=-np.arcsinh(10. ** 4) / np.log(10),
                 max=np.arcsinh(10. ** 4) / np.log(10), coord=plotcoordtmp, title=title, nest=True,
                 sub=(4, ncol, 3 * ncol + col))
    if col == ncol:
        plt.show()

for coord in ['C', 'CG']:
    plot_IQU(fake_solution, 'GSM gridded', 1, coord=coord)
    plot_IQU(w_GSM, 'wienered GSM', 2, coord=coord)
    plot_IQU(w_sim_sol, 'wienered simulated solution', 3, coord=coord)
    plot_IQU(w_solution, 'wienered solution', 4, coord=coord)
    sol_iquv = sol2map(w_solution).reshape((4, hpf.nside2npix(nside_standard)))
    hpv.mollview(np.arcsinh(sol_iquv[1]/2.) * np.log10(np.e), nest=True, sub=(4, 6, 12), min=-4, max=4, coord=coord)
    hpv.mollview(np.arcsinh(sol_iquv[2]/2.) * np.log10(np.e), nest=True, sub=(4, 6, 18), min=-4, max=4, coord=coord)
    hpv.mollview(np.arcsinh(sol_iquv[3]/2.) * np.log10(np.e), nest=True, sub=(4, 6, 24), min=-4, max=4, coord=coord)
    plt.show()


# hpv.mollview(np.log10(fake_solution[np.array(final_index).tolist()]), min= 0, max =4, coord=plotcoord, title='GSM gridded', nest=True)
# hpv.mollview(np.log10((x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw solution, chi^2=%.2f'%chisq, nest=True)
# hpv.mollview(np.log10((sim_x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw simulated solution, chi^2=%.2f'%chisq_sim, nest=True)
# hpv.mollview(np.log10((w_GSM/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered GSM', nest=True)
# hpv.mollview(np.log10((w_solution/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered solution', nest=True)
# hpv.mollview(np.log10((w_sim_sol/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered simulated solution', nest=True)
# plt.show()
