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
nside_standard = 128

plotcoord = 'CG'


baseline_safety_factor = 10.#max_ubl = 1.4*lambda/baseline_safety_factor
nside_beamweight = 16
crosstalk_type = 'autocorr'
# S_scale = 2
# S_thresh = 1000#Kelvin
# S_type = 'gsm%irm%i'%(S_scale,S_thresh)
S_type = 'dyS_lowadduniform_Iuniform'  # dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
pre_remove_additive = False


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
INSTRUMENT = sys.argv[1]#'miteor'#'mwa'#
if INSTRUMENT == 'miteor':

    nside_beamweight = 16
    lat_degree = 45.2977
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    tag = "q1AL_10_abscal"#"q3AL_5_abscal"  #'q3_abscalibrated'#"q4AL_3_abscal"#"q1AL_10_abscal"# L stands for lenient in flagging
    datatag = '_2016_01_20_avg_unpol'#'_seccasa.rad'#
    vartag = '_2016_01_20_avg_unpolx100'#''#
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    freqs = range(110, 200, 10)
    local_beam_unpol = si.interp1d(freqs, np.array([la.norm(np.fromfile(
        '/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
        (len(freqs), 12 * bnside ** 2, 2)), axis=-1)**2 for p in ['x', 'y']]).transpose(1, 0, 2), axis=0)
else:
    nside_beamweight = 256
    lat_degree = -26.703319
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    tag = "mwa_aug23_eor0" #
    datatag = '.datt4'#
    vartag = ''#''#
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/'

    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    freqs = range(110, 200, 10)
    local_beam_unpol = si.interp1d(freqs, np.array([[np.fromfile(
        '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_curtin_beam_%s_nside%i_freq167.275_zenith_float32.dat'%(P, bnside), dtype='float32') for P in ['XX', 'YY']] for i in range(len(freqs))]), axis=0)

A_version = 1.0
nf = 1
data_filename = glob.glob(datadir + tag + '_xx_*_*' + datatag)[0]
nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('xx_')[-1]
nt = int(nt_nUBL.split('_')[0])
nUBL = int(nt_nUBL.split('_')[1])




###t####
tmasks = {}
for p in ['x', 'y']:
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
        tmasks[p] = np.array(tfmlist[:, 0].astype('bool'))
        # print tmask
    except:
        print "No mask file found"
        tmasks[p] = np.ones_like(tlist).astype(bool)
tmask = tmasks['x']&tmasks['y']
tlist = tlist[tmask]
nt_used = len(tlist)

###UBL####
ubls = {}
for p in ['x', 'y']:
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
#manually filter UBLs
used_common_ubls = common_ubls[la.norm(common_ubls, axis=-1) / (300./freq) <= 1.4 * nside_standard / baseline_safety_factor]#[np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
nUBL_used = len(used_common_ubls)
ubl_index = {}  # stored index in each pol's ubl for the common ubls
for p in ['x', 'y']:
    ubl_index[p] = np.zeros(nUBL_used, dtype='int')
    for i, u in enumerate(used_common_ubls):
        if u in ubls[p]:
            ubl_index[p][i] = np.argmin(la.norm(ubls[p] - u, axis=-1)) + 1
        elif -u in ubls[p]:
            ubl_index[p][i] = - np.argmin(la.norm(ubls[p] + u, axis=-1)) - 1
        else:
            raise Exception('Logical Error')

print '>>>>>>Used nUBL = %i, nt = %i.'%(nUBL_used, nt_used)

####set up vs and beam
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
beam_heal_hor_x = local_beam_unpol(freq)[0]
beam_heal_hor_y = local_beam_unpol(freq)[1]
beam_heal_equ_x = sv.rotate_healpixmap(beam_heal_hor_x, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])
beam_heal_equ_y = sv.rotate_healpixmap(beam_heal_hor_y, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])

####initial A
A = {}
for p in ['x', 'y']:
    pol = p + p
    # ubl file
    #// ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls = np.array([[0,0,0]])#//np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    #// print "%i UBLs to include, longest baseline is %i wavelengths" % (len(ubls), np.max(np.linalg.norm(ubls, axis=1)) / (C / freq))

    A_path = datadir + tag + '_%s%s_%i_%i.A' % (p, p, nt_used * len(ubls), 12 * nside_beamweight ** 2)

    if os.path.isfile(A_path) and not force_recompute:
        print "Reading A matrix from %s" % A_path
        sys.stdout.flush()
        A[p] = np.fromfile(A_path, dtype='complex64').reshape((len(ubls) * nt_used, 12 * nside_beamweight ** 2))
    else:
        # beam
        if p == 'x':
            beam_heal_equ = beam_heal_equ_x
        elif p == 'y':
            beam_heal_equ = beam_heal_equ_x
        print "Computing sky weighting A matrix for %s pol..." % p
        sys.stdout.flush()

        A[p] = np.zeros((nt_used * len(ubls), 12 * nside_beamweight ** 2), dtype='complex64')

        timer = time.time()
        for i in np.arange(12 * nside_beamweight ** 2):
            dec, ra = hpf.pix2ang(nside_beamweight, i)  # gives theta phi
            dec = np.pi / 2 - dec
            print "\r%.1f%% completed" % (100. * float(i) / (12. * nside_beamweight ** 2)),
            sys.stdout.flush()
            if abs(dec - lat_degree * np.pi / 180) <= np.pi / 2:
                A[p][:, i] = vs.calculate_pointsource_visibility(ra, dec, ubls, freq, beam_heal_equ=beam_heal_equ, tlist=tlist).flatten()

        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A[p].tofile(A_path)

####################################################
###beam weights using an equal pixel A matrix######
#################################################
print "Computing beam weight...",
sys.stdout.flush()
beam_weight = ((la.norm(A['x'], axis=0) ** 2 + la.norm(A['y'], axis=0) ** 2) ** .5)[hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
beam_weight = beam_weight / np.mean(beam_weight)
thetas_standard, phis_standard = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
beam_weight = hpf.get_interp_val(beam_weight, thetas_standard, phis_standard, nest=True) #np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
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

parameters = []
err_norm = []
n_pix = []
for thresh in 2.**np.arange(-7., 2., 2.):
    for valid_pix_thresh in 10.**np.arange(-5., -2., .5):# 1e-4#1e-2
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print (thresh, valid_pix_thresh)
        ########################################################################
        ########################processing dynamic pixelization######################
        ########################################################################
        gsm_beamweighted = equatorial_GSM_standard * beam_weight
        nside_distribution = np.zeros(12 * nside_standard ** 2)
        final_index = np.zeros(12 * nside_standard ** 2, dtype=int)
        thetas, phis, sizes = [], [], []
        abs_thresh = np.mean(gsm_beamweighted) * thresh
        pixelize(gsm_beamweighted, nside_distribution, nside_standard, nside_start, abs_thresh,
                 final_index, thetas, phis, sizes)
        npix = len(thetas)
        valid_pix_mask = hpf.get_interp_val(gsm_beamweighted, thetas, phis, nest=True) > valid_pix_thresh * max(gsm_beamweighted)
        valid_npix = np.sum(valid_pix_mask)
        print '>>>>>>VALID NPIX =', valid_npix

        fake_solution_map = np.zeros_like(thetas)
        for i in range(len(fake_solution_map)):
            fake_solution_map[i] = np.sum(equatorial_GSM_standard[final_index == i])
        fake_solution_map = fake_solution_map[valid_pix_mask]

        fake_solution = np.concatenate((fake_solution_map, np.zeros(4 * nUBL_used)))

        sizes = np.array(sizes)[valid_pix_mask]


        def sol2map(sol):
            solx = sol[:valid_npix]
            full_sol = np.zeros(npix)
            full_sol[valid_pix_mask] = solx / sizes
            return full_sol[final_index]

        def sol2additive(sol):
            return np.transpose(sol[valid_npix:].reshape(nUBL_used, 2, 2), (1, 0, 2))#ubl by pol by re/im before transpose


        ###########################################################
        ####simulate visibilities using non dynamic pixelization###
        ##########################################


        full_sim_filename = datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis'%(nUBL_used+1, nt_used, nside_standard, bnside)

        if os.path.isfile(full_sim_filename):
            fullsim_vis = np.fromfile(full_sim_filename, dtype='complex64').reshape((2, nUBL_used+1, nt_used))
        else:

            fullsim_vis = np.zeros((2, nUBL_used + 1, nt_used), dtype='complex128')#since its going to accumulate along the pixels it needs to start with complex128. significant error if start with complex64
            full_sim_ubls = np.concatenate((used_common_ubls, [[0, 0, 0]]), axis=0)#tag along auto corr
            full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
            full_decs = np.pi / 2 - full_thetas
            full_ras = full_phis
            full_sim_mask = hpf.get_interp_val(beam_weight, full_thetas, full_phis, nest=True) > 0
            # fullsim_vis_DBG = np.zeros((2, len(used_common_ubls), nt_used, np.sum(full_sim_mask)), dtype='complex128')

            print "Simulating visibilities, %s, expected time %.1f min"%(datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (np.sum(full_sim_mask) / 1.4e5)),
            sys.stdout.flush()
            masked_equ_GSM = equatorial_GSM_standard[full_sim_mask]
            timer = time.time()
            for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
                for i, (ra, dec) in enumerate(zip(full_ras[full_sim_mask], full_decs[full_sim_mask])):
                    res = vs.calculate_pointsource_visibility(ra, dec, full_sim_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=tlist) / 2
                    fullsim_vis[p] += masked_equ_GSM[i] * res
                    # fullsim_vis_DBG[p, ..., i] = res[:-1]
            print "simulated visibilities in %.1f minutes."%((time.time() - timer) / 60.)
            fullsim_vis.astype('complex64').tofile(full_sim_filename)
        autocorr_vis = np.real(fullsim_vis[:, -1])
        if crosstalk_type == 'autocorr':
            autocorr_vis_normalized = np.array([autocorr_vis[p] / (la.norm(autocorr_vis[p]) / la.norm(np.ones_like(autocorr_vis[p]))) for p in range(2)])
        else:
            autocorr_vis_normalized = np.ones((2, nt_used))
        fullsim_vis = fullsim_vis[:, :-1].transpose((1, 0, 2))


        ##################################################################
        ####################compute dynamic A matrix########################
        ###############################################################
        A_tag = 'A_dI'
        A_filename = A_tag + '_u%i_t%i_p%i_n%i_%i_b%i_%.3f_v%.1f' % (nUBL_used, nt_used, valid_npix, nside_start, nside_standard, bnside, thresh, A_version)
        A_path = datadir + tag + A_filename

        def get_A():
            if os.path.isfile(A_path) and not force_recompute:
                print "Reading A matrix from %s" % A_path
                sys.stdout.flush()
                A = np.fromfile(A_path, dtype='complex64').reshape((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used))
            else:

                print "Computing A matrix..."
                sys.stdout.flush()
                A = np.empty((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used), dtype='complex64')
                timer = time.time()
                for n, i in enumerate(np.arange(npix)[valid_pix_mask]):
                    ra = phis[i]
                    dec = np.pi / 2 - thetas[i]
                    print "\r%.1f%% completed, %f minutes left" % (
                    100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
                    sys.stdout.flush()

                    A[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_x, tlist=tlist) / 2 #xx and yy are each half of I
                    A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=tlist) / 2



                print "%f minutes used" % (float(time.time() - timer) / 60.)
                sys.stdout.flush()
                # A.tofile(A_path)

            #put in autocorr regardless of whats saved on disk
            for i in range(nUBL_used):
                for p in range(2):
                    A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
                    A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]

            A.shape = (nUBL_used * 2 * nt_used, A.shape[-1])
            # Merge A
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

        # for ipix in hpf.ang2pix(nside_standard, thetas, phis, nest=True):
        #     if

        data = {}
        Ni = {}
        data_shape = {}
        ubl_sort = {}
        for p in ['x', 'y']:
            pol = p + p
            print "%i UBLs to include, longest baseline is %i wavelengths" % (
            nUBL_used, np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / freq))


            # get Ni (1/variance) and data
            var_filename = datadir + tag + '_%s%s_%i_%i' % (p, p, nt, nUBL) + vartag + '.var'
            data_filename = datadir + tag + '_%s%s_%i_%i' % (p, p, nt, nUBL) + datatag
            if INSTRUMENT == 'mwa':
                Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nUBL, nt))[:, tmask][abs(ubl_index[p]) - 1].flatten() * (1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * np.pi / (12 * nside_standard ** 2))) ** 2)
                data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nUBL, nt))[:, tmask][abs(ubl_index[p]) - 1]
                data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
                data[pol] = (data[pol].flatten() * 1.e-26 * (C / freq) ** 2 / 2 / kB / (
                4 * np.pi / (12 * nside_standard ** 2))).conjugate()  # there's a conjugate convention difference
                data_shape[pol] = (nUBL_used, nt_used)
            else:
                Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[
                                    abs(ubl_index[p]) - 1].flatten() * (
                                1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * np.pi / (12 * nside_standard ** 2))) ** 2)


                data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose()[
                    abs(ubl_index[p]) - 1]
                data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
                data[pol] = (data[pol].flatten() * 1.e-26 * (C / freq) ** 2 / 2 / kB / (
                4 * np.pi / (12 * nside_standard ** 2))).conjugate()  # there's a conjugate convention difference
                data_shape[pol] = (nUBL_used, nt_used)
            ubl_sort[p] = np.argsort(la.norm(used_common_ubls, axis=1))
        print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        sys.stdout.flush()

        # Merge data
        data = np.array([data['xx'], data['yy']]).reshape([2] + list(data_shape['xx'])).transpose(
            (1, 0, 2)).flatten()
        data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
        Ni = np.concatenate((Ni['xx'], Ni['yy'])).reshape([2] + list(data_shape['xx'])).transpose(
            (1, 0, 2)).flatten()
        Ni = np.concatenate((Ni * 2, Ni * 2))
        print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        sys.stdout.flush()

        print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
        sys.stdout.flush()


        def get_complex_data(real_data, nubl=nUBL_used, nt=nt_used):
            if len(real_data.flatten()) != 2 * nubl * 2 * nt:
                raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i."%(len(real_data), nubl, nt, 2 * nubl * 4 * nt))
            input_shape = real_data.shape
            real_data.shape = (2, nubl, 2, nt)
            result = real_data[0] + 1.j * real_data[1]
            real_data.shape = input_shape
            return result

        def get_vis_normalization(data, clean_sim_data):
            a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
            b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
            return a.dot(b) / b.dot(b)

        ##############
        # simulate visibilities according to the pixelized A matrix
        ##############
        clean_sim_data = A.dot(fake_solution.astype(A.dtype))

        us = ubl_sort['x'][::len(ubl_sort['x'])/10]
        cdata = get_complex_data(data)
        cdynamicmodel = get_complex_data(clean_sim_data)
        print "total deviation between dynamic and full sim compared to sim", la.norm(fullsim_vis - cdynamicmodel) / la.norm(fullsim_vis)
        print "total deviation between dynamic and full sim compared to data noise", la.norm(fullsim_vis - cdynamicmodel) / np.sum(Ni**-1)**.5
        parameters.append((thresh, valid_pix_thresh))
        err_norm.append(la.norm(fullsim_vis - cdynamicmodel))
        n_pix.append(valid_npix)
par_result_filename = full_sim_filename.replace('.simvis', '_par_search.npz')
np.savez(par_result_filename, parameters=parameters, err_norm=err_norm, n_pix=n_pix)

