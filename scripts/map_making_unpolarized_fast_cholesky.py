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
tag = sys.argv[2]
AtNiA_only = False
if len(sys.argv) > 3 and sys.argv[3][:5] == 'atnia':
    AtNiA_only = True
    pixel_scheme_number = int(sys.argv[3][5:])

plotcoord = 'CG'
baseline_safety_factor = 10.#max_ubl = 1.4*lambda/baseline_safety_factor
crosstalk_type = 'autocorr'
pixel_directory = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

plot_pixelization = True and not AtNiA_only
plot_projection = True and not AtNiA_only
plot_data_error = True and not AtNiA_only

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
    S_type = 'dyS_lowadduniform_lowI'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data

    seek_optimal_threshs = False and not AtNiA_only
    dynamic_precision = .2#.1#ratio of dynamic pixelization error vs data std, in units of data, so not power
    thresh = 2#.2#2.#.03125#
    valid_pix_thresh = 1.e-4
    nside_start = 32
    nside_standard = 256

    pre_calibrate = True
    pre_ampcal = ('qC' in tag)
    pre_phscal = True
    pre_addcal = True
    fit_for_additive = False
    nside_beamweight = 16

    lat_degree = 45.2977
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    # tag = "q3AL_5_abscal"  #"q0AL_13_abscal"  #"q1AL_10_abscal"'q3_abscalibrated'#"q4AL_3_abscal"# L stands for lenient in flagging
    if 'qC' in tag:
        datatag = '_2016_01_20_avg'#'_seccasa.rad'#
        vartag = '_2016_01_20_avgx100'#''#
    else:
        datatag = '_2016_01_20_avg2_unpollock'#'_2016_01_20_avg_unpollock'#'_seccasa.rad'#
        vartag = '_2016_01_20_avg2_unpollock'#'_2016_01_20_avg_unpollockx100'#''#
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
    antpairs = None
    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    freqs = range(110, 200, 10)
    local_beam_unpol = si.interp1d(freqs, np.array([la.norm(np.fromfile(
        '/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
        (len(freqs), 12 * bnside ** 2, 2)), axis=-1)**2 for p in ['x', 'y']]).transpose(1, 0, 2), axis=0)

elif INSTRUMENT == 'mwa':
    S_type = 'dyS_lowadduniform_lowI'  #'none'#'dyS_min2adduniform_Iuniform'  # dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
    seek_optimal_threshs = True
    dynamic_precision = .2
    pre_calibrate = True
    pre_ampcal = False
    pre_phscal = True
    pre_addcal = False
    fit_for_additive = False
    nside_beamweight = 64
    nside_start = 64
    nside_standard = 128
    lat_degree = -26.703319
    lst_offset = 5.#tlist will be wrapped around [lst_offset, 24+lst_offset]
    # tag = "mwa_aug23_eor0" #
    datatag = '.datt4'#
    vartag = ''#''#
    datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/'
    antpairs = np.array(zip(np.fromfile(datadir + 'ant1_int16.dat', dtype='int16'), np.fromfile(datadir + 'ant2_int16.dat', dtype='int16')))
    # deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
    bnside = 256
    freqs = range(110, 200, 10)
    local_beam_unpol = si.interp1d(freqs, np.array([[np.fromfile(
        '/home/omniscope/data/GSM_data/absolute_calibrated_data/mwa_aug23_eor0_forjeff/mwa_curtin_beam_%s_nside%i_freq167.275_zenith_float32.dat'%(P, bnside), dtype='float32') for P in ['XX', 'YY']] for i in range(len(freqs))]), axis=0)

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


if len(glob.glob(datadir + tag + '_xx_*_*' + datatag)) > 1:
    raise ValueError("Found more than one files for %s."%(datadir + tag + '_xx_*_*' + datatag))
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
jansky2kelvin = 1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * PI / (12 * nside_standard ** 2))
###UBL####
ubls = {}
for p in ['x', 'y']:
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
    ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
#manually filter UBLs
used_common_ubls = common_ubls[la.norm(common_ubls, axis=-1) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor]#[np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
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
        Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nUBL, nt))[:, tmask][abs(ubl_index[p]) - 1].flatten() * jansky2kelvin ** 2)
        data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nUBL, nt))[:, tmask][abs(ubl_index[p]) - 1]
        data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
        data[pol] = (data[pol].flatten() * jansky2kelvin).conjugate()  # there's a conjugate convention difference
        data_shape[pol] = (nUBL_used, nt_used)
    elif INSTRUMENT == 'paper':
        Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[
                            abs(ubl_index[p]) - 1].flatten() * jansky2kelvin ** 2)
        data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose()[
            abs(ubl_index[p]) - 1]
        data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
        data[pol] = (data[pol].flatten() * jansky2kelvin).conjugate()  # there's a conjugate convention difference
        data_shape[pol] = (nUBL_used, nt_used)
    else:
        Ni[pol] = 1. / (np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[
                            abs(ubl_index[p]) - 1].flatten() * jansky2kelvin ** 2)
        data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose()[
            abs(ubl_index[p]) - 1]
        data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
        data[pol] = (data[pol].flatten() * jansky2kelvin).conjugate()  # there's a conjugate convention difference
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

################################
################################
####pre_calibrate################
################################
################################
#####1. antenna based calibration#######
if antpairs is not None:
    used_antpairs = antpairs[abs(ubl_index['x'])-1]
    n_usedants = np.unique(used_antpairs)
#####2. re-phasing and crosstalk#######
additive_A = np.zeros((nUBL_used, 2, nt_used, 1 + 4 * nUBL_used), dtype='complex128')

#put in autocorr regardless of whats saved on disk
for p in range(2):
    additive_A[:, p, :, 0] = fullsim_vis[:, p]
    for i in range(nUBL_used):
        additive_A[i, p, :, 1 + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
        additive_A[i, p, :, 1 + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]
additive_A.shape = (nUBL_used * 2 * nt_used, 1 + 4 * nUBL_used)

if pre_calibrate:
    import omnical.calibration_omni as omni
    raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    raw_Ni = np.copy(Ni)

    real_additive_A = np.concatenate((np.real(additive_A), np.imag(additive_A)), axis=0)
    if pre_ampcal:#if pre_ampcal, allow xx and yy to fit amp seperately
        n_prefit_amp = 2
        real_additive_A.shape = (2 * nUBL_used, 2, nt_used, 1 + 4 * nUBL_used)
        real_additive_A_expand = np.zeros((2 * nUBL_used, 2, nt_used, n_prefit_amp + 4 * nUBL_used), dtype='float64')
        for i in range(n_prefit_amp):
            real_additive_A_expand[:, i, :, i] = real_additive_A[:, i, :, 0]
        real_additive_A_expand[..., n_prefit_amp:] = real_additive_A[..., 1:]
        real_additive_A = real_additive_A_expand
        real_additive_A.shape = (2 * nUBL_used * 2 * nt_used, n_prefit_amp + 4 * nUBL_used)
    else:
        n_prefit_amp = 1

    additive_AtNiA = np.empty((n_prefit_amp + 4 * nUBL_used, n_prefit_amp + 4 * nUBL_used), dtype='float64')
    if pre_addcal:
        ATNIA(real_additive_A, Ni, additive_AtNiA)
        additive_AtNiAi = sla.inv(additive_AtNiA)
    else:
        real_additive_A[..., n_prefit_amp:] = 0.
        ATNIA(real_additive_A, Ni, additive_AtNiA)
        additive_AtNiAi = sla.pinv(additive_AtNiA)

    niter = 0
    rephases = np.zeros((2,2))
    additive_term = np.zeros_like(data)
    additive_term_incr = np.zeros_like(data)
    while (niter == 0 or la.norm(rephases) > .001 or la.norm(additive_term_incr) / la.norm(data) > .001) and niter < 50:
        niter += 1

        if pre_phscal:
            cdata = get_complex_data(data)
            for p, pol in enumerate(['xx', 'yy']):
                rephase = omni.solve_phase_degen_fast(cdata[:, p].transpose(), cdata[:, p].transpose(), fullsim_vis[:, p].transpose(), fullsim_vis[:, p].transpose(), used_common_ubls)
                rephases[p] = rephase
                if p == 0:
                    print 'pre process rephase', pol, rephase,
                else:
                    print pol, rephase
                cdata[:, p] *= np.exp(1.j * used_common_ubls[:, :2].dot(rephase))[:, None]
            data = stitch_complex_data(cdata)

        additive_sol = additive_AtNiAi.dot(np.transpose(real_additive_A).dot(data * Ni))
        print '>>>>>>>>>>>>>additive fitting amp', additive_sol[:n_prefit_amp],
        additive_term_incr = real_additive_A[:, n_prefit_amp:].dot(additive_sol[n_prefit_amp:])
        data -= additive_term_incr
        additive_term += additive_term_incr
        print "additive fraction", la.norm(additive_term_incr) / la.norm(data),



    if pre_ampcal:
        data = stitch_complex_data(get_complex_data(data) / additive_sol[:n_prefit_amp, None])
        Ni = stitch_complex_data(get_complex_data(Ni) * additive_sol[:n_prefit_amp, None]**2)
        additive_term = stitch_complex_data(get_complex_data(additive_term) / additive_sol[:n_prefit_amp, None])

    print 'saving data to', os.path.dirname(data_filename) + '/' + tag + datatag + vartag + '_gsmcal_n%i_bn%i.npz'%(nside_standard, bnside)
    np.savez(os.path.dirname(data_filename) + '/' + tag + datatag + vartag + '_gsmcal_n%i_bn%i.npz'%(nside_standard, bnside),
             data=data,
             simdata=stitch_complex_data(fullsim_vis),
             psdata=[stitch_complex_data(vis) for vis in pt_vis],
             pt_sources=pt_sources,
             ubls=used_common_ubls,
             tlist=tlist,
             Ni=Ni,
             freq=freq)


    if plot_data_error:
        cdata = get_complex_data(data)
        crdata = get_complex_data(raw_data) / additive_sol[0]
        cNi = get_complex_data(Ni)
        cadd = get_complex_data(additive_term)

        fun = np.abs
        srt = sorted((tlist - lst_offset)%24.+lst_offset)
        asrt = np.argsort((tlist - lst_offset)%24.+lst_offset)
        pncol = min(int(60. / (srt[-1] - srt[0])), 12)
        us = ubl_sort['x']
        for p in range(2):
            for nu, u in enumerate(us):

                plt.subplot(5, (len(us) + 4) / 5, nu + 1)
                plt.plot(srt, fun(cdata[u, p][asrt]))
                plt.plot(srt, fun(fullsim_vis[u, p][asrt]))
                plt.plot(srt, fun(crdata[u, p][asrt]))
                plt.plot(srt, fun(cNi[u, p][asrt])**-.5)
                if pre_calibrate:
                    plt.plot(srt, fun(cadd[u, p][asrt]))
                data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), 5 * np.max(np.abs(fun(cNi[u, p])))])
                plt.title("%.1f,%.1f"%(used_common_ubls[u, 0], used_common_ubls[u, 1]))
                plt.ylim([-1.05*data_range, 1.05*data_range])

            plt.show()

################
####Use N and the par file generated by pixel_parameter_search to determine dynamic pixel parameters
################
if seek_optimal_threshs:
    par_result_filename = full_sim_filename.replace('.simvis', '_par_search.npz')
    par_file = np.load(par_result_filename)
    qualified_par_mask = (par_file['err_norm'] / np.sum(1./Ni)**.5) < dynamic_precision
    index_min_pix_in_mask = np.argmin(par_file['n_pix'][qualified_par_mask])
    thresh, valid_pix_thresh = par_file['parameters'][qualified_par_mask][index_min_pix_in_mask]
print "<<<<<<<<<<<<picked std thresh %.3f, pix thresh %.1e"%(thresh, valid_pix_thresh)

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
clean_sim_data = A.dot(fake_solution.astype(A.dtype))

if plot_data_error:
    cdata = get_complex_data(data)
    cdynamicmodel = get_complex_data(clean_sim_data)
    cNi = get_complex_data(Ni)
    if pre_calibrate:
        cadd = get_complex_data(additive_term)

    fun = np.imag
    srt = sorted((tlist - lst_offset)%24.+lst_offset)
    asrt = np.argsort((tlist - lst_offset)%24.+lst_offset)
    pncol = min(int(60. / (srt[-1] - srt[0])), 12)
    us = ubl_sort['x'][::len(ubl_sort['x'])/pncol]
    for p in range(2):
        for nu, u in enumerate(us):

            plt.subplot(2, len(us), len(us) * p + nu + 1)
            plt.plot(srt, fun(cdata[u, p][asrt]))
            plt.plot(srt, fun(fullsim_vis[u, p][asrt]))
            plt.plot(srt, fun(cdynamicmodel[u, p][asrt]))
            plt.plot(srt, fun(cNi[u, p][asrt])**-.5)
            if pre_calibrate:
                plt.plot(srt, fun(cadd[u, p][asrt]))
            data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), 5 * np.max(np.abs(fun(cNi[u, p])))])
            plt.title("%.1f,%.1f"%(used_common_ubls[u, 0], used_common_ubls[u, 1]))
            plt.ylim([-1.05*data_range, 1.05*data_range])

    print "total deviation between dynamic and full sim compared to sim", la.norm(fullsim_vis - cdynamicmodel) / la.norm(fullsim_vis)
    print "total deviation between dynamic and full sim compared to data noise", la.norm(fullsim_vis - cdynamicmodel) / np.sum(Ni**-1)**.5
    plt.show()

    try:
        fullsim_vis2 = 4 * np.fromfile(datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis'%(nUBL_used+1, nt_used, nside_standard / 2, bnside), dtype='complex64').reshape((2, nUBL_used+1, nt_used))[:, :-1].transpose((1, 0 ,2))
        plt.plot(la.norm(used_common_ubls, axis=-1)*freq/C, la.norm(fullsim_vis - fullsim_vis2, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'g+', label='nside error')
    except:
        try:
            fullsim_vis2 = .25 * np.fromfile(datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis'%(nUBL_used+1, nt_used, nside_standard * 2, bnside), dtype='complex64').reshape((2, nUBL_used+1, nt_used))[:, :-1].transpose((1, 0 ,2))
            plt.plot(la.norm(used_common_ubls, axis=-1)*freq/C, la.norm(fullsim_vis - fullsim_vis2, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'g+', label='nside error')
        except:
            pass
    plt.plot(la.norm(used_common_ubls, axis=-1)*freq/C, np.sum(2. / np.real(get_complex_data(Ni)), axis=-1)[:, 0]**.5 / la.norm(fullsim_vis, axis=-1)[:, 0], 'b+', label='noise error')
    plt.plot(la.norm(used_common_ubls, axis=-1)*freq/C, la.norm(fullsim_vis - cdynamicmodel, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'r+', label='dynamic pixel error')
    plt.legend()
    plt.xlabel('Baseline Length (wavelength)')
    plt.ylabel('Relative RMS Error')
    plt.show()



vis_normalization = get_vis_normalization(data, stitch_complex_data(fullsim_vis))
print "Normalization from visibilities", vis_normalization


##renormalize the model
fake_solution *= vis_normalization
clean_sim_data *= vis_normalization
fullsim_vis *= vis_normalization
sim_data = stitch_complex_data(fullsim_vis) + np.random.randn(len(data)) / Ni ** .5
#add additive term
if fit_for_additive:
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

    data_max = np.transpose(np.percentile(np.abs(data.reshape((2, nUBL_used, 2, nt_used))), 95, axis=-1), (1, 2, 0)).flatten()
    if 'min2add' in S_type:
        add_supress = 1000000.
    elif 'minadd' in S_type:
        add_supress = 10000.
    elif 'lowadd' in S_type:
        add_supress = 100.
    else:
        add_supress = 1

    if 'adduniform' in S_type:
        S_diag_add = np.ones(nUBL_used * 4) * np.median(data_max)**2 / add_supress
    else:
        S_diag_add = data_max**2 / add_supress

    if not fit_for_additive:
        S_diag = S_diag_I.astype('float64')
    else:
        S_diag = np.concatenate((S_diag_I, S_diag_add)).astype('float64')
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
rcond_list = 10.**np.arange(-12., -2., 1.)

AtNiAi_candidate_files = glob.glob(datadir + tag + AtNiAi_tag + '_S%s_RE*_N%s_v%.1f'%(S_type, vartag, AtNiAi_version) + A_filename)
if len(AtNiAi_candidate_files) > 0 and not force_recompute_AtNiAi and not force_recompute and not force_recompute_S and not AtNiA_only:
    rcond = 10**min([float(fn.split('_RE')[1].split('_N')[0]) for fn in AtNiAi_candidate_files])

    AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
    AtNiAi_path = datadir + tag + AtNiAi_filename

    print "Reading Regularized AtNiAi...",
    sys.stdout.flush()
    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision)
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
    print "Computing Regularized AtNiAi, %s, expected time %.1f min"%(datetime.datetime.now(), 88. * (len(S_diag) / 4.6e4)**3.),
    sys.stdout.flush()
    timer = time.time()
    # if la.norm(S) != la.norm(np.diagonal(S)):
    #     raise Exception("Non-diagonal S not supported yet")

    for rcond in rcond_list:
        #add Si on top og AtNiA without renaming AtNiA to save memory
        maxAtNiA = np.max(AtNiA)
        AtNiA.shape = (len(AtNiA) ** 2)
        AtNiA[::len(S_diag) + 1] += 1./S_diag

        print 'trying', rcond,
        sys.stdout.flush()
        try:
            AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f'%(S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
            AtNiAi_path = datadir + tag + AtNiAi_filename
            AtNiA[::len(S_diag) + 1] += maxAtNiA * rcond
            AtNiA.shape = (Ashape1, Ashape1)
            AtNiAi = sv.InverseCholeskyMatrix(AtNiA).astype(precision)
            del(AtNiA)
            AtNiAi.tofile(AtNiAi_path, overwrite=True)
            print "%f minutes used" % (float(time.time() - timer) / 60.)
            print "regularization stength", (maxAtNiA * rcond)**-.5, "median GSM ranges between", np.median(equatorial_GSM_standard) * min(sizes), np.median(equatorial_GSM_standard) * max(sizes)
            break
        except:
            AtNiA[::len(S_diag) + 1] -= maxAtNiA * rcond
            continue


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
best_fit_no_additive = A[..., :valid_npix].dot(w_solution[:valid_npix].astype(A.dtype))

sim_best_fit = A.dot(w_sim_sol.astype(A.dtype))
sim_best_fit_no_additive = A[..., :valid_npix].dot(w_sim_sol[:valid_npix].astype(A.dtype))

if plot_data_error:
    qaz_model = (clean_sim_data * vis_normalization).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    qaz_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    if pre_calibrate:
        qaz_add = np.copy(additive_term).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
    us = ubl_sort['x'][::max(1, len(ubl_sort['x'])/70)]
    best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
    ri = 1
    for p in range(2):
        for nu, u in enumerate(us):

            plt.subplot(6, (len(us) + 5) / 6, nu + 1)
            # plt.errorbar(range(nt_used), qaz_data[ri, u, p], yerr=Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
            plt.plot(qaz_data[ri, u, p])
            plt.plot(qaz_model[ri, u, p])
            plt.plot(best_fit[ri, u, p])
            plt.plot(best_fit_no_additive[ri, u, p])
            if pre_calibrate:
                plt.plot(qaz_add[ri, u, p])
            if fit_for_additive:
                plt.plot(autocorr_vis_normalized[p] * sol2additive(w_solution)[p, u, ri])
            plt.plot(best_fit[ri, u, p] - qaz_data[ri, u, p])
            plt.plot(Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
            data_range = np.max(np.abs(qaz_data[ri, u, p]))
            plt.ylim([-1.05*data_range, 1.05*data_range])
            plt.title("%.1f,%.1f,%.1e"%(used_common_ubls[u, 0], used_common_ubls[u, 1], la.norm(best_fit[ri, u, p] - qaz_data[ri, u, p])))
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

def plot_IQU(solution, title, col, shape=(2,3), coord='C'):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    I = sol2map(solution)
    plotcoordtmp = coord
    hpv.mollview(np.log10(I), min=0, max=4, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))
    if col == shape[0] * shape[1]:
        plt.show()

for coord in ['C', 'CG']:
    plot_IQU(w_GSM, 'wienered GSM', 1, coord=coord)
    plot_IQU(w_sim_sol, 'wienered simulated solution', 2, coord=coord)
    plot_IQU(w_solution, 'wienered solution', 3, coord=coord)
    plot_IQU(fake_solution, 'GSM gridded', 4, coord=coord)
    plot_IQU(w_sim_sol - w_GSM + fake_solution, 'combined sim solution', 5, coord=coord)
    plot_IQU(w_solution - w_GSM + fake_solution, 'combined solution', 6, coord=coord)
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

#point spread function:
if True and S_type == 'none':
    print "Reading Regularized AtNiAi...",
    sys.stdout.flush()
    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision)

    AtNiA_tag = 'AtNiA_N%s'%vartag
    if not fit_for_additive:
        AtNiA_tag += "_noadd"
    elif crosstalk_type == 'autocorr':
        AtNiA_tag += "_autocorr"
    AtNiA_filename = AtNiA_tag + A_filename
    AtNiA_path = datadir + tag + AtNiA_filename
    print "Reading AtNiA...",
    sys.stdout.flush()
    AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))

    iplot = 0
    valid_thetas_phis = np.array(zip(thetas, phis))
    full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
    for theta in np.arange(0, PI*.9, PI/6.):
        for phi in np.arange(0, TPI, PI/3.):
            iplot += 1
            choose_plots = [1, 12, 24]
            if iplot in choose_plots:
                np.argmin(la.norm(valid_thetas_phis - [theta, phi], axis=-1))
                point_vec = np.zeros_like(fake_solution)

                point_vec[np.argmin(la.norm(valid_thetas_phis - [theta, phi], axis=-1))] = 1
                spreaded = sol2map(AtNiAi.dotv(AtNiA.dot(point_vec)))
                spreaded /= np.max(spreaded)
                fwhm_mask = np.abs(spreaded) >= .5
                masked_max_ind = np.argmax(spreaded[fwhm_mask])
                fwhm_thetas = full_thetas[fwhm_mask]
                fwhm_phis = full_phis[fwhm_mask]
                #rotate angles to center around PI/2 0
                fwhm_thetas, fwhm_phis = hpr.Rotator(rot=[fwhm_phis[masked_max_ind], PI/2-fwhm_thetas[masked_max_ind], 0], deg=False)(fwhm_thetas, fwhm_phis)
                print fwhm_thetas[masked_max_ind], fwhm_phis[masked_max_ind]#should print 1.57079632679 0.0 if rotation is working correctly


                fwhm_theta = max(fwhm_thetas) - min(fwhm_thetas)
                phi_offset = fwhm_phis[masked_max_ind] - PI
                fwhm_phis = (fwhm_phis - phi_offset)%TPI + phi_offset
                fwhm_phi = max(fwhm_phis) - min(fwhm_phis)
                hpv.mollview(np.log10(np.abs(spreaded)), min=-3, max=0, nest=True, coord='CG', title='FWHM = %.1f\xf8'%((fwhm_theta*fwhm_phi)**.5*180./PI), sub=(len(choose_plots), 1, choose_plots.index(iplot)+1))
    plt.show()

cheat_cal = False
if cheat_cal:
    cdata = get_complex_data(data)
    ccheat_data = np.zeros_like(cdata)
    ccheat_oldfit = np.zeros_like(cdata)
    ccheat_sol = np.zeros((nUBL, 2, 2), dtype='complex128')
    cNitmp = get_complex_data(Ni)
    cNi = (np.real(cNitmp)**-1 + np.imag(cNitmp)**-1)**-1
    cclean_sim_data = get_complex_data(clean_sim_data)
    cA = np.empty((nt_used, 2), dtype='complex128')
    for p in range(2):
        cA[:, 1] = autocorr_vis_normalized[p]
        for u in range(nUBL_used):
            cA[:, 0] = cclean_sim_data[u, p]
            ni = cNi[u, p]
            cb = sla.inv((sv.tc(cA) * ni).dot(cA)).dot(sv.tc(cA).dot(ni * cdata[u, p]))
            ccheat_data[u, p] = (cdata[u, p] - cb[1] * autocorr_vis_normalized[p]) / cb[0]
            ccheat_oldfit[u, p] = cA.dot(cb)
            ccheat_sol[u, p] = cb

    cheat_error = stitch_complex_data(ccheat_oldfit - cdata).reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1]))
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

    cheat_data = stitch_complex_data(ccheat_data)
    cheat_w_solution = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision).dotv(np.transpose(A).dot((cheat_data * Ni).astype(A.dtype)))

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

    AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision)
    new_data = np.copy(data)
    new_best_fit = np.copy(best_fit)
    new_best_fit_no_additive = np.copy(best_fit_no_additive)

    for i in range(5):
        data_noadditive = np.copy(new_data).reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) - (new_best_fit - new_best_fit_no_additive)
        complex_data = np.sum(data_noadditive.transpose((2, 3, 1, 0)) * [1, 1.j], axis=-1)
        complex_best_fit = np.sum(new_best_fit_no_additive.transpose((2, 3, 1, 0)) * [1, 1.j], axis=-1)
        for p in range(2):
            amp = 1#data_noadditive.flatten().dot(new_best_fit_no_additive.flatten()) / new_best_fit_no_additive.flatten().dot(new_best_fit_no_additive.flatten())
            print amp
            degen = solve_phase_degen(complex_data[p], complex_data[p], complex_best_fit[p], complex_best_fit[p], used_common_ubls)
            print degen,
            complex_data[p] *= np.exp(1.j * used_common_ubls[:, :2].dot(degen)) / amp
            print solve_phase_degen(complex_data[p], complex_data[p], complex_best_fit[p], complex_best_fit[p], used_common_ubls)

        new_data = stitch_complex_data(complex_data.transpose((2, 0, 1)))
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