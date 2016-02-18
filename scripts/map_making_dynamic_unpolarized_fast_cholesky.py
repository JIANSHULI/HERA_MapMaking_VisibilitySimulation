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
bnside = 256
plotcoord = 'CG'
thresh = .5
valid_pix_thresh = 1e-2
nside_beamweight = 16
# S_scale = 2
# S_thresh = 1000#Kelvin
# S_type = 'gsm%irm%i'%(S_scale,S_thresh)
S_type = 'dyS_lowadduniform_Iuniform'  # dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
pre_remove_additive = False

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
tag = "q3AL_5_abscal"  #'q3_abscalibrated'#"q4AL_3_abscal"#"q1AL_10_abscal"# L stands for lenient in flagging
datatag = '_2016_01_20_avg_unpol'#'_seccasa.rad'#
vartag = '_2016_01_20_avg_unpol'#''#
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

# deal with beam: create a callable function of the form y(freq) in MHz and returns npix by 4
freqs = range(110, 200, 10)
local_beam_unpol = si.interp1d(freqs, np.array([la.norm(np.fromfile(
    '/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
    (len(freqs), 12 * bnside ** 2, 2)), axis=-1)**2 for p in ['x', 'y']]).transpose(1, 0, 2), axis=0)


A_version = 1.0
nf = 1
data_filename = glob.glob(datadir + tag + '_xx_*_*' + datatag)[0]
nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('xx_')[-1]
nt = int(nt_nUBL.split('_')[0])
nUBL = int(nt_nUBL.split('_')[1])



A = {}
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
for p in ['x', 'y']:
    pol = p + p
    # ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include, longest baseline is %i wavelengths" % (
    len(ubls), np.max(np.linalg.norm(ubls, axis=1)) / (C / freq))

    A_path = datadir + tag + '_%s%s_%i_%i.A' % (p, p, nt_used * len(ubls), 12 * nside_beamweight ** 2)

    if os.path.isfile(A_path) and not force_recompute:
        print "Reading A matrix from %s" % A_path
        sys.stdout.flush()
        A[p] = np.fromfile(A_path, dtype='complex64').reshape((len(ubls) * nt_used, 12 * nside_beamweight ** 2))
    else:
        # beam
        if p == 'x':
            beam_healpix = local_beam_unpol(freq)[0]
        elif p == 'y':
            beam_healpix = local_beam_unpol(freq)[1]
        # hpv.mollview(beam_healpix, title='beam %s'%p)
        # plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
        beam_heal_equ = np.array(
            sv.rotate_healpixmap(beam_healpix, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0]))
        print "Computing A matrix for %s pol..." % p
        sys.stdout.flush()
        timer = time.time()
        A[p] = np.zeros((nt_used * len(ubls), 12 * nside_beamweight ** 2), dtype='complex64')
        for i in range(12 * nside_beamweight ** 2):
            dec, ra = hpf.pix2ang(nside_beamweight, i)  # gives theta phi
            dec = np.pi / 2 - dec
            print "\r%.1f%% completed, %f minutes left" % (100. * float(i) / (12. * nside_beamweight ** 2),
                                                           (12. * nside_beamweight ** 2 - i) / (i + 1) * (
                                                           float(time.time() - timer) / 60.)),
            sys.stdout.flush()
            if abs(dec - lat_degree * np.pi / 180) < np.pi / 2:
                A[p][:, i] = np.array(
                    [vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ=beam_heal_equ, tlist=tlist) for d
                     in ubls]).flatten()

        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A[p].tofile(A_path)

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

########################################################################
########################processing dynamic pixelization######################
########################################################################

nside_distribution = np.zeros(12 * nside_standard ** 2)
final_index = np.zeros(12 * nside_standard ** 2, dtype=int)
thetas, phis, sizes = [], [], []
abs_thresh = np.mean(equatorial_GSM_standard * beam_weight) * thresh
pixelize(equatorial_GSM_standard * beam_weight, nside_distribution, nside_standard, nside_start, abs_thresh,
         final_index, thetas, phis, sizes)
npix = len(thetas)
valid_pix_mask = hpf.get_interp_val(beam_weight, thetas, phis, nest=True) > valid_pix_thresh * max(beam_weight)
valid_npix = np.sum(valid_pix_mask)


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


###########################################################
####simulate visibilities using non dynamic pixelization###
##########################################

vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, lat_degree * np.pi / 180])  # self.zenithequ
beam_heal_hor_x = local_beam_unpol(freq)[0]
beam_heal_hor_y = local_beam_unpol(freq)[1]
beam_heal_equ_x = sv.rotate_healpixmap(beam_heal_hor_x, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])
beam_heal_equ_y = sv.rotate_healpixmap(beam_heal_hor_y, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])

full_sim_filename = datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis'%(nUBL_used+1, nt_used, nside_standard, bnside)

if os.path.isfile(full_sim_filename):
    fullsim_vis = np.fromfile(full_sim_filename, dtype='complex64').reshape((2, nUBL_used+1, nt_used))
else:
    print "Simulating visibilities, %s, expected time %.1f min"%(datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (nside_standard / 128.)**2),
    sys.stdout.flush()
    fullsim_vis = np.zeros((2, len(common_ubls) + 1, nt_used), dtype='complex128')#since its going to accumulate along the pixels it needs to start with complex128. significant error if start with complex64
    full_sim_ubls = np.concatenate((common_ubls, [[0, 0, 0]]), axis=0)#tag along auto corr
    full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
    full_decs = np.pi / 2 - full_thetas
    full_ras = full_phis
    full_sim_mask = hpf.get_interp_val(beam_weight, full_thetas, full_phis, nest=True) > 0
    # fullsim_vis_DBG = np.zeros((2, len(common_ubls), nt_used, np.sum(full_sim_mask)), dtype='complex128')

    masked_equ_GSM = equatorial_GSM_standard[full_sim_mask]
    timer = time.time()
    for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
        for i, (ra, dec) in enumerate(zip(full_ras[full_sim_mask], full_decs[full_sim_mask])):
            res = vs.calculate_pointsource_visibility(ra, dec, full_sim_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=tlist) / 2
            fullsim_vis[p] += masked_equ_GSM[i] * res
            # fullsim_vis_DBG[p, ..., i] = res[:-1]
    print "simulated visibilities in %.1f minutes."%((time.time() - timer) / 60.)
    fullsim_vis.astype('complex64').tofile(full_sim_filename)
autocorr_vis = np.real(fullsim_vis[:, -1])#TODO use this in cross talk removal
autocorr_vis_normalized = np.array([autocorr_vis[p] / (la.norm(autocorr_vis[p]) / la.norm(np.ones_like(autocorr_vis[p]))) for p in range(2)])
fullsim_vis = fullsim_vis[:, :-1].transpose((1, 0, 2))

plt.plot(autocorr_vis_normalized.transpose())
plt.title("autocorr_vis_normalized")
plt.ylim([0, 2])
plt.show()
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
        A = np.fromfile(A_path, dtype='complex64').reshape((nUBL_used * 2 * nt_used, valid_npix + 4 * nUBL_used))
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

        for i in range(nUBL_used):
            for p in range(2):
                A[i, p, :, valid_npix + 4 * i + 2 * p] = 1.
                A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j

        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A.tofile(A_path)

    # Merge A
    A.shape = (nUBL_used * 2 * nt_used, A.shape[-1])
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
    Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[
                        abs(ubl_index[p]) - 1].flatten() * (
                    1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * np.pi / (12 * nside_standard ** 2))) ** 2)
    data_filename = datadir + tag + '_%s%s_%i_%i' % (p, p, nt, nUBL) + datatag
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


def get_complex_data(flat_real_data, nubl=nUBL_used, nt=nt_used):
    if len(flat_real_data) != 2 * nubl * 2 * nt:
        raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i."%(len(flat_real_data), nubl, nt, 2 * nubl * 4 * nt))

    flat_real_data.shape = (2, nubl, 2, nt)
    result = flat_real_data[0] + 1.j * flat_real_data[1]
    flat_real_data.shape = 2 * nubl * 2 * nt
    return result

def get_vis_normalization(data, clean_sim_data):
    a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    return a.dot(b) / b.dot(b)

##############
# simulate visibilities according to the pixelized A matrix
##############
clean_sim_data = A.dot(fake_solution.astype(A.dtype))

if plot_data_error:
    us = ubl_sort['x'][::len(ubl_sort['x'])/6]
    cdata = get_complex_data(data)
    cdynamicmodel = get_complex_data(clean_sim_data)
    fun = np.imag
    for p in range(2):
        for nu, u in enumerate(us):

            plt.subplot(2, len(us), len(us) * p + nu + 1)

            plt.plot(fun(cdata[u, p]))
            plt.plot(fun(fullsim_vis[u, p]))
            plt.plot(fun(cdynamicmodel[u, p]))
            data_range = np.max(np.abs(fun(cdata[u, p])))
            plt.ylim([-1.05*data_range, 1.05*data_range])
    plt.show()




vis_normalization = get_vis_normalization(data, clean_sim_data)
print "Normalization from visibilities", vis_normalization
diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
diff_data = diff_data[0] + 1j * diff_data[1]
diff_norm = {}
diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 2, data_shape['xx'][1])[:, 0], axis=1)
diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 2, data_shape['yy'][1])[:, -1], axis=1)
data_norm = {}
data_norm['x'] = la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[0, :, 0] + 1.j * data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[1, :, 0], axis=-1)
data_norm['y'] = la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[0, :, -1] + 1.j * data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[1, :, -1], axis=-1)

if plot_data_error:
    plt.subplot(7, 1, 1)
    plt.plot((diff_norm['x']/data_norm['x'])[ubl_sort['x']])
    plt.plot((diff_norm['y']/data_norm['y'])[ubl_sort['y']])


if pre_remove_additive:
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
        for p in range(2):
                additive_inc[:, p] = np.outer(
                    autocorr_vis[p].dot(diff_data[:, p].transpose()) / np.sum(autocorr_vis[p] ** 2), autocorr_vis[p])

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
    qaz_model = (clean_sim_data * vis_normalization).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    qaz_data = data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    plt.subplot(7, 1, 2)
    if pre_remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][0], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][0], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][0], 0])
    plt.subplot(7, 1, 3)
    if pre_remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][data_shape['xx'][0]/2], 0])
    plt.subplot(7, 1, 4)
    if pre_remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][-1], 0])
    plt.plot(qaz_data[1, ubl_sort['x'][-1], 0])
    plt.plot(qaz_model[1, ubl_sort['x'][-1], 0])
    plt.subplot(7, 1, 5)
    if pre_remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][0], 1])
    plt.plot(qaz_data[1, ubl_sort['x'][0], 1])
    plt.plot(qaz_model[1, ubl_sort['x'][0], 1])
    plt.subplot(7, 1, 6)
    if pre_remove_additive:
        plt.plot(raw_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.plot(qaz_data[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.plot(qaz_model[1, ubl_sort['x'][data_shape['xx'][0]/2], 1])
    plt.subplot(7, 1, 7)
    if pre_remove_additive:
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
fake_solution *= vis_normalization
clean_sim_data *= vis_normalization
fullsim_vis *= vis_normalization
sim_data = np.concatenate((np.real(fullsim_vis.flatten()), np.imag(fullsim_vis.flatten()))) + np.random.randn(len(data)) / Ni ** .5
#add additive term
sim_data.shape = (2, nUBL_used, 2, nt_used)
sim_additive = np.random.randn(2, nUBL_used, 2) * np.median(np.abs(data)) / 2.
sim_data = sim_data + np.array([np.outer(sim_additive[..., p], autocorr_vis_normalized[p]).reshape((2, nUBL_used, nt_used)) for p in range(2)]).transpose((1, 2, 0, 3))#sim_additive[..., None]
sim_data = sim_data.flatten()

# compute AtNi.y
AtNi_data = np.transpose(A).dot((data * Ni).astype(A.dtype))
AtNi_sim_data = np.transpose(A).dot((sim_data * Ni).astype(A.dtype))
AtNi_clean_sim_data = np.transpose(A).dot((clean_sim_data * Ni).astype(A.dtype))

# compute S
print "computing S...",
sys.stdout.flush()
timer = time.time()

#diagonal of S consists of S_diag_I and S-diag_add
if 'Iuniform' in S_type:
    S_diag_I = (np.median(equatorial_GSM_standard) * sizes)**2
else:
    S_diag_I = fake_solution_map ** 2  # np.array([[1+pol_frac,0,0,1-pol_frac],[0,pol_frac,pol_frac,0],[0,pol_frac,pol_frac,0],[1-pol_frac,0,0,1+pol_frac]]) / 4 * (2*sim_x_clean[i])**2

data_max = np.transpose(np.percentile(np.abs(data.reshape((2, nUBL_used, 2, nt_used))), 95, axis=-1), (1, 2, 0)).flatten()
if 'lowadd' in S_type:
    add_supress = 100.
else:
    add_supress = 1

if 'adduniform' in S_type:
    S_diag_add = np.ones(nUBL_used * 4) * np.median(data_max)**2 / add_supress
else:
    S_diag_add = data_max**2 / add_supress

S = np.diag(np.concatenate((S_diag_I, S_diag_add))).astype('float64')
print "Done."
print "%f minutes used" % (float(time.time() - timer) / 60.)
sys.stdout.flush()



# compute (AtNiA+Si)i eigensystems
precision = 'float64'
AtNiAi_tag = 'AtNiASii'
AtNiAi_version = 0.1
rcond = 1e-4
AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
AtNiAi_path = datadir + tag + AtNiAi_filename

if os.path.isfile(AtNiAi_path) and not force_recompute_AtNiAi and not force_recompute and not force_recompute_S:
    print "Reading Regularized AtNiAi...",
    sys.stdout.flush()
    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S), precision)
else:
    AtNiA_tag = 'AtNiA_N%s'%vartag
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
best_fit_no_additive = np.sum((A * (w_solution.astype(A.dtype))).reshape((Ashape0, Ashape1))[..., :valid_npix], axis=-1)

sim_best_fit = A.dot(w_sim_sol.astype(A.dtype))
sim_best_fit_no_additive = np.sum((A * (w_sim_sol.astype(A.dtype))).reshape((Ashape0, Ashape1))[..., :valid_npix], axis=-1)

if plot_data_error:
    us = ubl_sort['x'][::len(ubl_sort['x'])/10]
    best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    ri = 1
    for p in range(2):
        for nu, u in enumerate(us):

            plt.subplot(4, len(us), len(us) * p + nu + 1)

            plt.errorbar(range(nt_used), qaz_data[ri, u, p], yerr=Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
            plt.plot(qaz_model[ri, u, p])
            plt.plot(best_fit[ri, u, p])
            plt.plot(best_fit_no_additive[ri, u, p])
            plt.plot(np.ones_like(qaz_data[ri, u, p]) * sol2additive(w_solution)[p, u, ri])
            plt.plot(best_fit[ri, u, p] - qaz_data[ri, u, p])
            plt.plot(Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
            data_range = np.max(np.abs(qaz_data[ri, u, p]))
            plt.ylim([-1.05*data_range, 1.05*data_range])
    plt.show()

    # sim_best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    # sim_best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    # ri = 1
    # for p in range(2):
    #     for nu, u in enumerate(us):
    #
    #         plt.subplot(4, len(us), len(us) * p + nu + 1)
    #         sim_qazdata = sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    #         plt.plot(sim_qazdata[ri, u, p])
    #         plt.plot(qaz_model[ri, u, p])
    #         plt.plot(sim_best_fit[ri, u, p])
    #         plt.plot(sim_best_fit_no_additive[ri, u, p])
    #         plt.plot(np.ones_like(sim_qazdata[ri, u, p]) * sol2additive(w_sim_sol)[p, u, ri])
    #         plt.plot(sim_best_fit[ri, u, p] - sim_qazdata[ri, u, p])
    #         data_range = np.max(np.abs(sim_qazdata[ri, u, p]))
    #         plt.ylim([-1.05*data_range, 1.05*data_range])
    # plt.show()

def plot_IQU(solution, title, col, ncol=4, coord='C'):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    I = sol2map(solution)
    plotcoordtmp = coord
    hpv.mollview(np.log10(I), min=0, max=4, coord=plotcoordtmp, title=title, nest=True, sub=(1, ncol, col))
    if col == ncol:
        plt.show()

for coord in ['C', 'CG']:
    plot_IQU(fake_solution, 'GSM gridded', 1, coord=coord)
    plot_IQU(w_GSM, 'wienered GSM', 2, coord=coord)
    plot_IQU(w_sim_sol, 'wienered simulated solution', 3, coord=coord)
    plot_IQU(w_solution, 'wienered solution', 4, coord=coord)
    plt.show()



error = data.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) - best_fit
chi = error * (Ni.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])))**.5
print "chi^2 = %.3e, data points %i, pixels %i"%(la.norm(chi)**2, len(data), valid_npix)
print "re/im chi2 %.3e, %.3e"%(la.norm(chi[0])**2, la.norm(chi[1])**2)
print "xx/yy chi2 %.3e, %.3e"%(la.norm(chi[:, :, 0])**2, la.norm(chi[:, :, 1])**2)
plt.subplot(2, 2, 1)
plt.plot([la.norm(error[:, u]) for u in ubl_sort['x']])
plt.subplot(2, 2, 2)
plt.plot([la.norm(chi[:, u]) for u in ubl_sort['x']])
plt.subplot(2, 2, 3)
plt.plot(tlist, [la.norm(error[..., t]) for t in range(error.shape[-1])])
plt.subplot(2, 2, 4)
plt.plot(tlist, [la.norm(chi[..., t]) for t in range(error.shape[-1])])
plt.show()

cheat_cal = False
if cheat_cal:
    data_noadditive = data.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) - (best_fit - best_fit_no_additive)
    cdata_noadditive = data_noadditive[0] + 1.j * data_noadditive[1]
    cbestfit_noadditive = best_fit_no_additive[0] + 1.j * best_fit_no_additive[1]
    cclean_sim_data = get_complex_data(clean_sim_data)
    cheat_ratio = np.sum(cdata_noadditive * np.conjugate(cclean_sim_data), axis=-1) / np.sum(cclean_sim_data * np.conjugate(cclean_sim_data), axis=-1)
    cheat_fit = (cclean_sim_data * cheat_ratio[..., None]).flatten()
    cheat_error = (data_noadditive.flatten() - np.concatenate((np.real(cheat_fit), np.imag(cheat_fit)))).reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1]))
    cheat_chi = cheat_error * (Ni.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])))**.5
    print "chi^2 = %.3e, data points %i, pixels %i"%(la.norm(cheat_chi)**2, len(data), valid_npix)
    print "re/im chi2 %.3e, %.3e"%(la.norm(cheat_chi[0])**2, la.norm(cheat_chi[1])**2)
    print "xx/yy chi2 %.3e, %.3e"%(la.norm(cheat_chi[:, :, 0])**2, la.norm(cheat_chi[:, :, 1])**2)
    plt.subplot(2, 2, 1)
    plt.plot([la.norm(cheat_error[:, u]) for u in ubl_sort['x']])
    plt.subplot(2, 2, 2)
    plt.plot([la.norm(cheat_chi[:, u]) for u in ubl_sort['x']])
    plt.subplot(2, 2, 3)
    plt.plot(tlist, [la.norm(cheat_error[..., t]) for t in range(error.shape[-1])])
    plt.subplot(2, 2, 4)
    plt.plot(tlist, [la.norm(cheat_chi[..., t]) for t in range(error.shape[-1])])
    plt.show()

    ccheat_data = (cdata_noadditive / cheat_ratio[..., None]).flatten()
    cheat_data = np.concatenate((np.real(ccheat_data), np.imag(ccheat_data)))
    cheat_w_solution = AtNiAi.dotv(np.transpose(A).dot((cheat_data * Ni).astype(A.dtype)))

    for coord in ['C', 'CG']:
        plot_IQU(fake_solution, 'GSM gridded', 1, coord=coord)
        plot_IQU(w_GSM, 'wienered GSM', 2, coord=coord)
        plot_IQU(w_sim_sol, 'wienered simulated solution', 3, coord=coord)
        plot_IQU(cheat_w_solution, 'wienered solution', 4, coord=coord)
        plt.show()

selfcal = False#currently buggy: chi2 increases over iterations
if selfcal:

    import omnical.calibration_omni as omni
    def solve_phase_degen(data_xx, data_yy, model_xx, model_yy, ubls, plot=False):#data should be time by ubl at single freq. data * phasegensolution = model
        if data_xx.shape != data_yy.shape or data_xx.shape != model_xx.shape or data_xx.shape != model_yy.shape or data_xx.shape[1] != ubls.shape[0]:
            raise ValueError("Shapes mismatch: %s %s %s %s, ubl shape %s"%(data_xx.shape, data_yy.shape, model_xx.shape, model_yy.shape, ubls.shape))
        A = np.zeros((len(ubls) * 2, 2))
        b = np.zeros(len(ubls) * 2)

        nrow = 0
        for p, (data, model) in enumerate(zip([data_xx, data_yy], [model_xx, model_yy])):
            for u, ubl in enumerate(ubls):
                amp_mask = (np.abs(data[:, u]) > (np.median(np.abs(data[:, u])) / 2.))
                A[nrow] = ubl[:2]
                b[nrow] = omni.medianAngle(np.angle(model[:, u] / data[:, u])[amp_mask])
                nrow += 1
        phase_cal = omni.solve_slope(np.array(A), np.array(b), 1)
        if plot:
            plt.hist((np.array(A).dot(phase_cal)-b + PI)%TPI-PI)
            plt.title('phase fitting error')
            plt.show()

        #sooolve
        return phase_cal

    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S), precision)
    new_data = np.copy(data)
    new_best_fit = np.copy(best_fit)
    new_best_fit_no_additive = np.copy(best_fit_no_additive)

    for i in range(5):
        data_noadditive = new_data.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) - (new_best_fit - new_best_fit_no_additive)
        complex_data = np.sum(data_noadditive.transpose((2, 3, 1, 0)) * [1, 1.j], axis=-1)
        complex_best_fit = np.sum(new_best_fit_no_additive.transpose((2, 3, 1, 0)) * [1, 1.j], axis=-1)
        for p in range(2):
            amp = 1#data_noadditive.flatten().dot(new_best_fit_no_additive.flatten()) / new_best_fit_no_additive.flatten().dot(new_best_fit_no_additive.flatten())
            print amp
            degen = solve_phase_degen(complex_data[p], complex_data[p], complex_best_fit[p], complex_best_fit[p], used_common_ubls)
            print degen,
            complex_data[p] *= np.exp(1.j * common_ubls[:, :2].dot(degen)) / amp
            print solve_phase_degen(complex_data[p], complex_data[p], complex_best_fit[p], complex_best_fit[p], used_common_ubls)

        new_data = np.concatenate((np.real(complex_data.transpose((2, 0, 1)).flatten()), np.imag(complex_data.transpose((2, 0, 1)).flatten())))
        new_w_solution = AtNiAi.dotv(np.transpose(A).dot((new_data * Ni).astype(A.dtype)))
        new_best_fit = A.dot(new_w_solution.astype(A.dtype))
        new_chi = (new_data - new_best_fit).reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) * (Ni.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])))**.5
        print la.norm(chi)**2, la.norm(new_chi)**2
        print "-----"


        new_best_fit_no_additive = np.sum((A * (new_w_solution.astype(A.dtype))).reshape((Ashape0, Ashape1))[..., :valid_npix], axis=-1)
        new_best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
        new_best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])

    for coord in ['C', 'CG']:
        plot_IQU(fake_solution, 'GSM gridded', 1, coord=coord)
        plot_IQU(w_GSM, 'wienered GSM', 2, coord=coord)
        plot_IQU(w_sim_sol, 'wienered simulated solution', 3, coord=coord)
        plot_IQU(new_w_solution, 'wienered solution', 4, coord=coord)
        plt.show()