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
        raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i"%(A.ndim, C.ndim, Ni.ndim))

    expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1])**2
    print "Estimated time for A %i by %i"%(A.shape[0], A.shape[1]), expected_time, "minutes",
    sys.stdout.flush()

    chunk = len(C) / nchunk
    for i in range(nchunk):
        ltm = time.time()
        C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
        if expected_time >= 1.:
            print "%i/%i: %.1fmins"%(i, nchunk, (time.time() - ltm)/60.),
            sys.stdout.flush()
    if chunk * nchunk < len(C):
        C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)


#####commandline inputs#####
INSTRUMENT = sys.argv[1]#'miteor'#'mwa'#
freq = float(sys.argv[2])#'miteor'#'mwa'#
AtNiA_only = False
if len(sys.argv) > 3 and sys.argv[3][:5] == 'atnia':
    AtNiA_only = True
    pixel_scheme_number = int(sys.argv[3][5:])

plotcoord = 'CG'
baseline_safety_factor = 2.5#max_ubl = lambda/baseline_safety_factor
crosstalk_type = 'autocorr'
pixel_directory = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

plot_pixelization = True and not AtNiA_only
plot_projection = True and not AtNiA_only
plot_data_error = False and not AtNiA_only

force_recompute = False
force_recompute_AtNiAi_eig = False
force_recompute_AtNiAi = False
force_recompute_S = False
force_recompute_SEi = False

C = 299.792458
kB = 1.3806488 * 1.e-23
script_dir = os.path.dirname(os.path.realpath(__file__))

####################################################
################data file and load beam##############
####################################################

if INSTRUMENT == 'miteor':
    S_type = 'none'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data

    seek_optimal_threshs = False and not AtNiA_only
    dynamic_precision = .2#.1#ratio of dynamic pixelization error vs data std, in units of data, so not power
    thresh = 1.e9#2#.2#2.#.03125#
    valid_pix_thresh = 1.e-6
    nside_start = 32
    nside_standard = 128

    pre_calibrate = True
    pre_ampcal = False
    pre_phscal = True
    pre_addcal = False
    fit_for_additive = False
    nside_beamweight = 16

    lat_degree = 45.2977
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
    antpairs = None
    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    freqs = range(110, 200, 10)
    local_beam_unpol = si.interp1d(freqs, np.array([la.norm(np.fromfile(
        '/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
        (len(freqs), 12 * bnside ** 2, 2)), axis=-1)**2 for p in ['x', 'y']]).transpose(1, 0, 2), axis=0)

    # freq = 167.275#125.
    bw = 0.75#MHz
    deltat = .04#hours
    jansky2kelvin = 1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * PI / (12 * nside_standard ** 2))
    tlist = np.arange(12., 24., deltat)%24.
    nt_used = len(tlist)
    ###UBL####
    nUBL = 112
    ubls = {}
    for p in ['x', 'y']:
        ubl8 = [[3. * i, 3. * j, 0] for i in range(8) for j in range(8) if (i != 0 or j != 0)]
        ubl7 = [[3. * i, -3. * j, 0] for i in range(1, 8) for j in range(1, 8)]
        ubls[p] = np.array(ubl8 + ubl7)
    redundancy = (8. - np.abs(ubls['x'][:, 0]) / 3.) * (8. - np.abs(ubls['x'][:, 1]) / 3.)

elif INSTRUMENT == 'mwa':
    S_type = 'none'#'dyS_lowadduniform_lowI'  #'none'#'dyS_min2adduniform_Iuniform'  # dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
    seek_optimal_threshs = False
    thresh = 1e9#.2#2.#.03125#
    valid_pix_thresh = 1.e-6
    dynamic_precision = .2
    pre_calibrate = True
    pre_ampcal = False
    pre_phscal = True
    pre_addcal = False
    fit_for_additive = False
    nside_beamweight = 64
    nside_start = 32
    nside_standard = 128
    lat_degree = -26.703319
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    # tag = "mwa_aug23_eor0" #
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/'
    antpairs = np.array(zip(np.fromfile(datadir + 'ant1_int16.dat', dtype='int16'), np.fromfile(datadir + 'ant2_int16.dat', dtype='int16')))
    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    beam167 = [np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_curtin_beam_%s_nside%i_freq167.275_zenith_float32.dat'%(P, bnside), dtype='float32') for P in ['XX', 'YY']]
    beam150 = [np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_curtin_beam_%s_nside%i_freq150.0_zenith_float32.dat'%(P, bnside), dtype='float32') for P in ['XX', 'YY']]
    beam125 = [np.fromfile('/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/beam%s_125'%(P), dtype='float32') for P in ['XX', 'YY']]
    local_beam_unpol = si.interp1d(np.array([125., 150., 167.275]), np.array([beam125, beam150, beam167]), axis=0)

    # freq = 125#167.275#125.
    bw = .64
    deltat = .04
    jansky2kelvin = 1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * PI / (12 * nside_standard ** 2))
    # tlist = np.arange(12., 22., deltat)%24.
    tlist = np.arange(12., 24., deltat)%24.
    nt_used = len(tlist)
    ###UBL####
    nUBL = 1872
    ubls = {}
    for p in ['x', 'y']:
        ubl_filename = datadir + 'mwa_aug23_eor0' + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
        ubls[p] = np.fromfile('/mnt/data0/omniscope/downloads/baselines_BaselineNum_SouthEastUp_float32.dat', dtype='float32').reshape((nUBL,3))
    redundancy = np.ones(nUBL)

elif INSTRUMENT == 'paper':
    S_type = 'dyS_lowadduniform_lowI'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data

    seek_optimal_threshs = False and not AtNiA_only
    dynamic_precision = .2#.1#ratio of dynamic pixelization error vs data std, in units of data, so not power
    thresh = 2#2.#.03125#
    valid_pix_thresh = 1.e-3
    nside_start = 32
    nside_standard = 256

    pre_calibrate = True
    pre_ampcal = True
    pre_phscal = True
    pre_addcal = False
    fit_for_additive = False
    nside_beamweight = 16

    lat_degree = -30.72153
    lst_offset = 0.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    beam_freqs = np.arange(120., 185., 5.)

    datatag = '_lstbineven_avg4'
    vartag = '_lstbineven_avg4'
    datadir = '/home/omniscope/data/PAPER/lstbin_fg/even/'
    antpairs = None


    #cal file
    calfile = 'psa6622_v002'
    print "Reading calfile %s..."%calfile,
    sys.stdout.flush()
    aa = ap.cal.get_aa(calfile, beam_freqs/1000.)
    print "Done."
    sys.stdout.flush()

    bnside = 64
    beam_healpix = np.zeros((len(beam_freqs), 2, 12*bnside**2), dtype='float32')
    healpixvecs = np.array(hpf.pix2vec(bnside, range(12*bnside**2)))
    paper_healpixvecs = (healpixvecs[:, healpixvecs[2]>=0]).transpose().dot(sv.rotatez_matrix(-np.pi/2).transpose())#in paper's bm_response convention, (x,y) = (0,1) points north.
    for p, pol in enumerate(['x', 'y']):
        for i, paper_angle in enumerate(paper_healpixvecs):
            beam_healpix[:, p, i] = (aa[0].bm_response(paper_angle, pol)**2.).flatten()

    local_beam_unpol = si.interp1d(beam_freqs, beam_healpix, axis=0)
tag = INSTRUMENT + '_%.2fMHz'%freq
data_tag = 'sim'
vartag = '%.2e'%(bw * deltat)
print '#####################################################'
print '###############',tag,'###########'
print '#####################################################'
A_version = 1.0
nf = 1
# data_filename = glob.glob(datadir + tag + '_xx_*_*' + datatag)[0]
# nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('xx_')[-1]
# nUBL = int(nt_nUBL.split('_')[1])



common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])#useless since both pol have same ubls
#manually filter UBLs
cal_ubl_mask = (la.norm(common_ubls, axis=-1) / (C / freq) <= nside_start / baseline_safety_factor)&(la.norm(common_ubls, axis=-1) > 2)
# cal_ubl_mask = (la.norm(common_ubls, axis=-1) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor)&(la.norm(common_ubls, axis=-1) > 2)
used_common_ubls = common_ubls[cal_ubl_mask]#[np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
redundancy = redundancy[cal_ubl_mask]
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

################
####set up vs and beam
################
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, lat_degree * PI / 180])  # self.zenithequ
beam_heal_hor_x = local_beam_unpol(freq)[0]
beam_heal_hor_y = local_beam_unpol(freq)[1]
beam_heal_equ_x = sv.rotate_healpixmap(beam_heal_hor_x, 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0])
beam_heal_equ_y = sv.rotate_healpixmap(beam_heal_hor_y, 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0])

################
####initial A to compute beam weight
A = {}
for p in ['x', 'y']:
    pol = p + p
    # ubl file
    #// ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls = np.array([[0,0,0]])#//np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    #// print "%i UBLs to include, longest baseline is %i wavelengths" % (len(ubls), np.max(np.linalg.norm(ubls, axis=1)) / (C / freq))

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
        dec = PI / 2 - dec
        print "\r%.1f%% completed" % (100. * float(i) / (12. * nside_beamweight ** 2)),
        sys.stdout.flush()
        if abs(dec - lat_degree * PI / 180) <= PI / 2:
            A[p][:, i] = vs.calculate_pointsource_visibility(ra, dec, ubls, freq, beam_heal_equ=beam_heal_equ, tlist=tlist).flatten()

    print "%f minutes used" % (float(time.time() - timer) / 60.)
    sys.stdout.flush()

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
########################################################################
########################processing dynamic pixelization######################
########################################################################
gsm_beamweighted = equatorial_GSM_standard * beam_weight
if AtNiA_only:
    valid_npix = pixel_scheme_number
    pixel_scheme_file = np.load(pixel_directory + 'pixel_scheme_%i.npz'%valid_npix)
    fake_solution_map = pixel_scheme_file['gsm']
    thetas = pixel_scheme_file['thetas']
    phis= pixel_scheme_file['phis']
    sizes= pixel_scheme_file['sizes']
    nside_distribution= pixel_scheme_file['nside_distribution']
    final_index= pixel_scheme_file['final_index']
    npix = pixel_scheme_file['n_fullsky_pix']
    valid_pix_mask= pixel_scheme_file['valid_pix_mask']
    thresh= pixel_scheme_file['thresh']
else:
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
    sizes = np.array(sizes)[valid_pix_mask]
    thetas = np.array(thetas)[valid_pix_mask]
    phis = np.array(phis)[valid_pix_mask]
    np.savez(pixel_directory + 'pixel_scheme_%i.npz'%valid_npix, gsm=fake_solution_map, thetas=thetas, phis=phis, sizes=sizes, nside_distribution=nside_distribution, final_index=final_index, n_fullsky_pix=npix, valid_pix_mask=valid_pix_mask, thresh=thresh)#thresh is in there for idiotic reason  due to unneccessary inclusion of thresh in A filename

if not fit_for_additive:
    fake_solution = np.copy(fake_solution_map)
else:
    fake_solution = np.concatenate((fake_solution_map, np.zeros(4 * nUBL_used)))

def sol2map(sol,resize=True):
    solx = sol[:valid_npix]
    full_sol = np.zeros(npix)+ np.nan
    if resize:
        full_sol[valid_pix_mask] = solx / sizes
    else:
        full_sol[valid_pix_mask] = solx

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
full_sim_filename = datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis'%(nUBL_used+1, nt_used, nside_standard, bnside)

if os.path.isfile(full_sim_filename):
    fullsim_vis = np.fromfile(full_sim_filename, dtype='complex64').reshape((2, nUBL_used+1, nt_used))
else:

    fullsim_vis = np.zeros((2, nUBL_used + 1, nt_used), dtype='complex128')#since its going to accumulate along the pixels it needs to start with complex128. significant error if start with complex64
    full_sim_ubls = np.concatenate((used_common_ubls, [[0, 0, 0]]), axis=0)#tag along auto corr
    full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
    full_decs = PI / 2 - full_thetas
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

Ni = (np.repeat([((bw * 1.e6 * deltat * 3600.)**-.5 * autocorr_vis)**-2 * 2], nUBL_used * 2, axis=0).reshape(2, nUBL_used, 2, nt_used) * redundancy[:, None, None]).flatten()
if plot_data_error:
    plt.plot(autocorr_vis_normalized.transpose())
    plt.title("autocorr_vis_normalized")
    plt.ylim([0, 2])
    plt.show()

######################
####simulate cas and cyg
#######################
southern_points = {'hyd':{'ra': '09:18:05.7', 'dec': '-12:05:44'},
'cen':{'ra': '13:25:27.6', 'dec': '-43:01:09'},
'cyg':{'ra': '19:59:28.3', 'dec': '40:44:02'},
'pic':{'ra': '05:19:49.7', 'dec': '-45:46:44'},
'vir':{'ra': '12:30:49.4', 'dec': '12:23:28'},
'for':{'ra': '03:22:41.7', 'dec': '-37:12:30'},
'sag':{'ra': '17:45:40.045', 'dec': '-29:0:27.9'},
'cas':{'ra': '23:23:26', 'dec': '58:48:00'},
'crab':{'ra': '5:34:31.97', 'dec': '22:00:52.1'}}


for source in southern_points.keys():
    southern_points[source]['body'] = ephem.FixedBody()
    southern_points[source]['body']._ra = southern_points[source]['ra']
    southern_points[source]['body']._dec = southern_points[source]['dec']

flux_func = {}
flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])

pt_sources = ['cyg', 'cas']
pt_vis = np.zeros((len(pt_sources), 2, nUBL_used, nt_used), dtype='complex128')
if INSTRUMENT == 'miteor':
    print "Simulating cyg casvisibilities, %s, expected time %.1f min"%(datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (2. / 1.4e5)),
    sys.stdout.flush()
    timer = time.time()
    for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
        for i, source in enumerate(pt_sources):
            ra = southern_points[source]['body']._ra
            dec = southern_points[source]['body']._dec
            pt_vis[i, p] = jansky2kelvin * flux_func[source](freq) * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=tlist) / 2

################
####read data and N
################
data_shape = {}
ubl_sort = {}
for p in ['x', 'y']:
    pol = p + p
    print "%i UBLs to include, longest baseline is %i wavelengths" % (
    nUBL_used, np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / freq))

    ubl_sort[p] = np.argsort(la.norm(used_common_ubls, axis=1))
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()


def get_complex_data(real_data, nubl=nUBL_used, nt=nt_used):
    if len(real_data.flatten()) != 2 * nubl * 2 * nt:
        raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i."%(len(real_data), nubl, nt, 2 * nubl * 2 * nt))
    input_shape = real_data.shape
    real_data.shape = (2, nubl, 2, nt)
    result = real_data[0] + 1.j * real_data[1]
    real_data.shape = input_shape
    return result

def stitch_complex_data(complex_data):
    return np.concatenate((np.real(complex_data.flatten()), np.imag(complex_data.flatten())))



##################################################################
####################compute dynamic A matrix########################
###############################################################
A_tag = 'A_dI'
A_filename = A_tag + '_u%i_t%i_p%i_n%i_%i_b%i_%.3f_v%.1f' % (nUBL_used, nt_used, valid_npix, nside_start, nside_standard, bnside, thresh, A_version)
A_path = datadir + tag + A_filename
AtNiA_tag = 'AtNiA_N%s'%vartag
if not fit_for_additive:
    AtNiA_tag += "_noadd"
elif crosstalk_type == 'autocorr':
    AtNiA_tag += "_autocorr"
if pre_ampcal:
    AtNiA_tag += "_ampcal"
AtNiA_filename = AtNiA_tag + A_filename
AtNiA_path = datadir + tag + AtNiA_filename
if os.path.isfile(AtNiA_path) and AtNiA_only and not force_recompute:
    sys.exit(0)


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
        for n in range(valid_npix):
            ra = phis[n]
            dec = PI / 2 - thetas[n]
            print "\r%.1f%% completed, %f minutes left" % (
            100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
            sys.stdout.flush()

            A[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_x, tlist=tlist) / 2 #xx and yy are each half of I
            A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=tlist) / 2



        print "%f minutes used" % (float(time.time() - timer) / 60.)
        sys.stdout.flush()
        A.tofile(A_path)

    # #put in autocorr regardless of whats saved on disk
    # for i in range(nUBL_used):
    #     for p in range(2):
    #         A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
    #         A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]

    A.shape = (nUBL_used * 2 * nt_used, A.shape[-1])
    if not fit_for_additive:
        A = A[:, :valid_npix]
    else:
        A[:, valid_npix:] = additive_A[:, 1:]
    # Merge A
    try:
        return np.concatenate((np.real(A), np.imag(A))).astype('float64')
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
        return A.astype('float64')


A = get_A()
Ashape0, Ashape1 = A.shape

# for ipix in hpf.ang2pix(nside_standard, thetas, phis, nest=True):
#     if

print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()



def get_vis_normalization(data, clean_sim_data):
    a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
    return a.dot(b) / b.dot(b)

##############
# simulate visibilities according to the pixelized A matrix
##############
clean_sim_data = A.dot(fake_solution.astype('float64'))

sim_data = stitch_complex_data(fullsim_vis) + np.random.randn(len(Ni)) / Ni ** .5
#add additive term
# if fit_for_additive:
#     sim_data.shape = (2, nUBL_used, 2, nt_used)
#     sim_additive = np.random.randn(2, nUBL_used, 2) * np.median(np.abs(data)) / 2.
#     sim_data = sim_data + np.array([np.outer(sim_additive[..., p], autocorr_vis_normalized[p]).reshape((2, nUBL_used, nt_used)) for p in range(2)]).transpose((1, 2, 0, 3))#sim_additive[..., None]
#     sim_data = sim_data.flatten()

# compute AtNi.y
# AtNi_data = np.transpose(A).dot((data * Ni).astype(A.dtype))
AtNi_sim_data = np.transpose(A).dot((sim_data * Ni).astype('float64'))
AtNi_clean_sim_data = np.transpose(A).dot((clean_sim_data * Ni).astype('float64'))

for AtNid_type, AtNid in zip(['AtNisd', 'AtNicsd'], [AtNi_sim_data, AtNi_clean_sim_data]):
    AtNid_tag = AtNiA_tag.replace('AtNiA', AtNid_type)
    AtNid_filename = AtNid_tag + A_filename
    AtNid_path = datadir + tag + AtNid_filename
    AtNid.tofile(AtNid_path)

# compute S
print "computing S...",
sys.stdout.flush()
timer = time.time()

#diagonal of S consists of S_diag_I and S-diag_add
if S_type == 'none':
    S_diag = np.ones(Ashape1) * np.max(equatorial_GSM_standard)**2 * 1.e12
else:
    if 'lowI' in S_type:
        I_supress = 25.
    else:
        I_supress = 1.
    if 'Iuniform' in S_type:
        S_diag_I = (np.median(equatorial_GSM_standard) * sizes)**2 / I_supress
    else:
        S_diag_I = fake_solution_map ** 2 / I_supress  # np.array([[1+pol_frac,0,0,1-pol_frac],[0,pol_frac,pol_frac,0],[0,pol_frac,pol_frac,0],[1-pol_frac,0,0,1+pol_frac]]) / 4 * (2*sim_x_clean[i])**2


    print "Done."
    print "%f minutes used" % (float(time.time() - timer) / 60.)
    sys.stdout.flush()



# compute (AtNiA+Si)i
precision = 'float64'
AtNiAi_tag = 'AtNiASii'
if not fit_for_additive:
    AtNiAi_version = 0.3
elif crosstalk_type == 'autocorr':
    AtNiAi_version = 0.2
else:
    AtNiAi_version = 0.1
if pre_ampcal:
    AtNiAi_version += 1.
start_try = {'mwa': -10, 'miteor': -9}[INSTRUMENT]
rcond_list = 10.**np.arange(start_try + 2, -2., 1.)

AtNiAi_candidate_files = glob.glob(datadir + tag + AtNiAi_tag + '_S%s_RE*_N%s_v%.1f'%(S_type, vartag, AtNiAi_version) + A_filename)
if len(AtNiAi_candidate_files) > 0 and not force_recompute_AtNiAi and not force_recompute and not force_recompute_S and not AtNiA_only:
    rcond = 10**min([float(fn.split('_RE')[1].split('_N')[0]) for fn in AtNiAi_candidate_files])

    AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
    AtNiAi_path = datadir + tag + AtNiAi_filename

    print "Reading Regularized AtNiAi...",
    sys.stdout.flush()
    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, Ashape1, precision)
else:
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
    if AtNiA_only:
        sys.exit(0)
    del (A)
    AtNiA_diag = np.diagonal(AtNiA)
    print "Computing Regularized AtNiAi, %s, expected time %.1f min"%(datetime.datetime.now(), 88. * (Ashape1 / 4.6e4)**3.),
    sys.stdout.flush()
    timer = time.time()
    # if la.norm(S) != la.norm(np.diagonal(S)):
    #     raise Exception("Non-diagonal S not supported yet")

    for rcond in rcond_list:
        #add Si on top og AtNiA without renaming AtNiA to save memory
        maxAtNiA = np.max(AtNiA)
        AtNiA.shape = (len(AtNiA) ** 2)
        AtNiA[::Ashape1 + 1] += 1./S_diag

        print 'trying', rcond,
        sys.stdout.flush()
        try:
            AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
            AtNiAi_path = datadir + tag + AtNiAi_filename
            AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
            AtNiA.shape = (Ashape1, Ashape1)
            AtNiAi = sv.InverseCholeskyMatrix(AtNiA).astype(precision)
            del(AtNiA)
            AtNiAi.tofile(AtNiAi_path, overwrite=True)
            print "%f minutes used" % (float(time.time() - timer) / 60.)
            print "regularization stength", (maxAtNiA * rcond)**-.5, "median GSM ranges between", np.median(equatorial_GSM_standard) * min(sizes), np.median(equatorial_GSM_standard) * max(sizes)
            break
        except:
            AtNiA[::Ashape1 + 1] -= maxAtNiA * rcond
            continue

#####apply wiener filter##############
print "Applying Regularized AtNiAi...",
sys.stdout.flush()
# w_solution = AtNiAi.dotv(AtNi_data)
w_GSM = AtNiAi.dotv(AtNi_clean_sim_data)
w_sim_sol = AtNiAi.dotv(AtNi_sim_data)
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()
# del (AtNiAi)
# A = get_A()
# best_fit = A.dot(w_solution.astype(A.dtype))
# best_fit_no_additive = A[..., :valid_npix].dot(w_solution[:valid_npix].astype(A.dtype))
#
# sim_best_fit = A.dot(w_sim_sol.astype(A.dtype))
# sim_best_fit_no_additive = A[..., :valid_npix].dot(w_sim_sol[:valid_npix].astype(A.dtype))
#
# if plot_data_error:
#     qaz_model = (clean_sim_data * vis_normalization).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
#     qaz_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
#     if pre_calibrate:
#         qaz_add = np.copy(additive_term).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
#     us = ubl_sort['x'][::max(1, len(ubl_sort['x'])/70)]
#     best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
#     best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
#     ri = 1
#     for p in range(2):
#         for nu, u in enumerate(us):
#
#             plt.subplot(6, (len(us) + 5) / 6, nu + 1)
#             # plt.errorbar(range(nt_used), qaz_data[ri, u, p], yerr=Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
#             plt.plot(qaz_data[ri, u, p])
#             plt.plot(qaz_model[ri, u, p])
#             plt.plot(best_fit[ri, u, p])
#             plt.plot(best_fit_no_additive[ri, u, p])
#             if pre_calibrate:
#                 plt.plot(qaz_add[ri, u, p])
#             if fit_for_additive:
#                 plt.plot(autocorr_vis_normalized[p] * sol2additive(w_solution)[p, u, ri])
#             plt.plot(best_fit[ri, u, p] - qaz_data[ri, u, p])
#             plt.plot(Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
#             data_range = np.max(np.abs(qaz_data[ri, u, p]))
#             plt.ylim([-1.05*data_range, 1.05*data_range])
#             plt.title("%.1f,%.1f,%.1e"%(used_common_ubls[u, 0], used_common_ubls[u, 1], la.norm(best_fit[ri, u, p] - qaz_data[ri, u, p])))
#         plt.show()

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

def plot_IQU(solution, title, col, shape=(2,3), coord='C', min=0, max=4, log=True, resize=True):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    I = sol2map(solution, resize=resize)
    plotcoordtmp = coord
    if log:
        hpv.mollview(np.log10(I), min=min, max=max, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))
    else:
        hpv.mollview(I, min=min, max=max, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))

    if col == shape[0] * shape[1]:
        plt.show()

for coord in ['C', 'CG']:
    plot_IQU(w_GSM, 'wienered GSM', 1, coord=coord)
    plot_IQU(w_sim_sol, 'wienered simulated solution', 2, coord=coord)
    # plot_IQU(w_solution, 'wienered solution', 3, coord=coord)
    plot_IQU(fake_solution, 'GSM gridded', 4, coord=coord)
    plot_IQU(w_sim_sol - w_GSM + fake_solution, 'combined sim solution', 5, coord=coord)
    # plot_IQU(w_solution - w_GSM + fake_solution, 'combined solution', 6, coord=coord)
    plt.show()
################
###look for best rcond
# ################
# AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
# for i, p in enumerate(np.arange(start_try, start_try + 7)):
#     rcond = 10**p
#     maxAtNiA = np.max(AtNiA)
#     AtNiA.shape = (len(AtNiA) ** 2)
#     # AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
#     # AtNiAi_path = datadir + tag + AtNiAi_filename
#     AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
#     AtNiA.shape = (Ashape1, Ashape1)
#     AtNiAi = sv.InverseCholeskyMatrix(AtNiA).astype(precision)
#     # del(AtNiA)
#     # AtNiAi.tofile(AtNiAi_path, overwrite=True)
#     print "%f minutes used" % (float(time.time() - timer) / 60.)
#     print "regularization stength", (maxAtNiA * rcond)**-.5 / np.median(sizes), "median GSM ranges between", np.median(equatorial_GSM_standard)
#
#     #####apply wiener filter##############
#     print "Applying Regularized AtNiAi...",
#     sys.stdout.flush()
#     # w_solution = AtNiAi.dotv(AtNi_data)
#     w_GSM = AtNiAi.dotv(AtNi_clean_sim_data)
#     w_sim_sol = AtNiAi.dotv(AtNi_sim_data)
#     print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
#     sys.stdout.flush()
#
#     AtNiA.shape = (len(AtNiA) ** 2)
#     # AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
#     # AtNiAi_path = datadir + tag + AtNiAi_filename
#     AtNiA[::Ashape1 + 1] -= maxAtNiA * rcond
#     AtNiA.shape = (Ashape1, Ashape1)
#     plot_IQU(w_GSM, 'wienered GSM', i+1, shape=(2, 7), coord='cg')
#     plot_IQU(w_sim_sol, 'wienered simulated solution', i+8, shape=(2, 7), coord='cg')
################
###full inverse
################
rcond = {'mwa': 10**-8, 'miteor': 1e-7}[INSTRUMENT]
AtNiAi_filename = 'AtNiASifi' + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
AtNiAi_path = datadir + tag + AtNiAi_filename
AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
maxAtNiA = np.max(AtNiA)
AtNiA.shape = (len(AtNiA) ** 2)

AtNiA[::Ashape1 + 1] += maxAtNiA * rcond
AtNiA.shape = (Ashape1, Ashape1)
AtNiAi = la.inv(AtNiA)
print  '###########check they are 1#################'
print AtNiAi[0].dot(AtNiA[:, 0]), AtNiAi[-1].dot(AtNiA[:, -1])
print  '###################################'
del(AtNiA)
AtNiAi.tofile(AtNiAi_path)
print "%f minutes used" % (float(time.time() - timer) / 60.)
print "regularization stength", (maxAtNiA * rcond)**-.5 / np.median(sizes), "median GSM ranges between", np.median(equatorial_GSM_standard)

print "baseline length in wavelengths between:", np.min(la.norm(used_common_ubls,axis=-1)) / (300. / freq), np.max(la.norm(used_common_ubls,axis=-1)) / (300. / freq)

#####apply wiener filter##############
print "Applying Regularized AtNiAi...",
sys.stdout.flush()
# w_solution = AtNiAi.dotv(AtNi_data)
w_GSM = AtNiAi.dot(AtNi_clean_sim_data)
w_sim_sol = AtNiAi.dot(AtNi_sim_data)
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()


AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
psf = np.einsum('ij,jk->ik', AtNiAi, AtNiA)
print '############check first one is 0###########'
print la.norm(psf.dot(fake_solution) - w_GSM), la.norm(psf.dot(fake_solution) - w_sim_sol)
print '#######################################'
def fwhm2(psf, verbose=False):
    spreaded = np.abs(psf) / np.max(np.abs(psf))
    fwhm_mask = spreaded >= .5
    return (np.sum(fwhm_mask) * 4 * PI / hpf.nside2npix(nside_start) / PI)**.5
resolution = np.array([fwhm2(pf) for pf in psf.transpose()])

rescale = np.sum(psf, axis=-1)
noise = np.sum(psf * np.transpose(AtNiAi), axis=-1)**.5 / np.abs(rescale)
result = w_sim_sol / rescale
import matplotlib
matplotlib.rcParams.update({'font.size':16})
plot_IQU(fake_solution, 'GSM (Log(K))', 1, shape=(4, 1), coord='cg')
plot_IQU(result, 'Simulated Result (Log(K))', 2, shape=(4, 1), coord='cg')
plot_IQU(noise, 'Uncertainty (K)', 3, shape=(4, 1), coord='cg')
# plot_IQU(noise, 'Uncertainty (Log(K))', 3, shape=(4, 1), coord='cg')
plot_IQU(resolution * 180/PI, 'Resolution', 4, shape=(4, 1), coord='cg', log=False, resize=False, min=2, max=5)


matplotlib.rcParams.update({'font.size':22})
plot_IQU(result, 'Simulated Dirty Map (Log(K))', 1, shape=(1, 1), coord='cg', min=2)
plot_IQU(w_GSM / rescale, 'Regulated GSM', 1, shape=(1, 1), coord='cg', min=2)
plot_IQU(noise, 'Uncertainty (Log(K))', 1, shape=(1, 1), coord='cg', min=1, max=3)
plot_IQU(np.abs(w_GSM - w_sim_sol) / np.sum(psf * np.transpose(AtNiAi), axis=-1)**.5, 'chi', 1, shape=(1, 1), coord='cg', resize=False, min=0, max=2, log=False)
plot_IQU(resolution * 180/PI, 'Resolution (degree)', 1, shape=(1, 1), coord='cg', log=False, resize=False, min=2, max=5)



###CLEAN
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
plot_IQU(cleaned_result2, 'Simulated CLEANed Map (Log(K))', 2, shape=(2, 1), coord='cg')
plot_IQU(cleaned_result2, 'Simulated CLEANed Map (Log(K))', 1, shape=(1, 1), coord='CG')