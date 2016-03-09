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
import fitsio as fit

PI = np.pi
TPI = PI * 2

nside_standard = 256
standard_freq = 150.

total_valid_npix = 15801
pixel_dir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
Qs = []
A_fns = []
AtNiA_fns = []
data_fns = []

INSTRUMENTS = ['paper']#, 'miteor_compact']#'paper']#, 'miteor']
valid_npixs = {'paper': 14896, 'miteor': 12997, 'miteor_compact': 12997}
datatags = {'paper': '_lstbineven_avg4', 'miteor': '_2016_01_20_avg_unpollock', 'miteor_compact': '_2016_01_20_avg'}
vartags = {'paper': '_lstbineven_avg4', 'miteor': '_2016_01_20_avg_unpollockx100', 'miteor_compact': '_2016_01_20_avgx100'}
datadirs = {'paper': '/home/omniscope/data/PAPER/lstbin_fg/even/', 'miteor': '/home/omniscope/data/GSM_data/absolute_calibrated_data/', 'miteor_compact': '/home/omniscope/data/GSM_data/absolute_calibrated_data/'}
bnsides = {'paper': 64, 'miteor': 256, 'miteor_compact': 256}
noise_scales = {'paper': 10., 'miteor': 1., 'miteor_compact': 1.}

for instrument in INSTRUMENTS:
    datatag = datatags[instrument]
    vartag = vartags[instrument]
    datadir = datadirs[instrument]
    valid_npix = valid_npixs[instrument]
    instru_data_fns = glob.glob(datadir + '*' + datatag + vartag + '_gsmcal_n%i_bn%i.npz'%(nside_standard, bnsides[instrument]))
    data_fns += instru_data_fns
    for data_fn in instru_data_fns:
        Q = os.path.basename(data_fn).split(datatag)[0]
        Qs.append(Q)

        A_candidates = glob.glob(datadir + Q + 'A_dI*p%i*'%valid_npix)
        if len(A_candidates) != 1:
            raise IOError("Not unique files for %s: %s. Searched %s."%(Q, A_candidates, datadir + Q + 'A_dI*p%i*'%valid_npix))
        else:
            A_fn = A_candidates[0]

        AtNiA_candidates = glob.glob(datadir + Q + 'AtNiA_N%s_noadd*%s'%(vartag, A_fn.split(Q)[1]))
        if len(AtNiA_candidates) != 1:
            raise IOError("Not unique files for %s: %s. Searched %s."%(Q, AtNiA_candidates, datadir + Q + 'AtNiA_N%s_noadd%s'%(vartag, A_fn.split(Q)[1])))
        else:
            AtNiA_fn = AtNiA_candidates[0]
        A_fns.append(A_fn)
        AtNiA_fns.append(AtNiA_fn)

###pixel scheme
pixel_scheme_file = np.load(pixel_dir + 'pixel_scheme_%i.npz'%total_valid_npix)
fake_solution_map = pixel_scheme_file['gsm']
thetas = pixel_scheme_file['thetas']
phis= pixel_scheme_file['phis']
sizes= pixel_scheme_file['sizes']
nside_distribution= pixel_scheme_file['nside_distribution']
final_index= pixel_scheme_file['final_index']
npix = pixel_scheme_file['n_fullsky_pix']
valid_pix_mask= pixel_scheme_file['valid_pix_mask']
thresh= pixel_scheme_file['thresh']

sub_pixel_files = {}
child_masks = {}
for instrument in INSTRUMENTS:
    sub_pixel_files[instrument] = np.load(pixel_dir + 'pixel_scheme_%i.npz'%valid_npixs[instrument])
    child_masks[instrument] = sub_pixel_files[instrument]['child_mask']


def sol2map(sol, rescale_order=-1):
    solx = sol[:total_valid_npix]
    full_sol = np.zeros(npix)
    full_sol[valid_pix_mask] = solx * sizes**rescale_order
    return full_sol[final_index]

def plot_IQU(solution, title, col, shape=(1, 1), coord='C', rescale_order=-1, min=0, max=4, nside_out=None):
    # Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    # I = Es[0] + Es[3]
    # Q = Es[0] - Es[3]
    # U = Es[1] + Es[2]
    I = sol2map(solution, rescale_order=rescale_order)
    if nside_out != None:
        I = hpf.ud_grade(I, nside_out=nside_out, order_in='NESTED', order_out='NESTED')
    plotcoordtmp = coord
    hpv.mollview(np.log10(I), min=min, max=max, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))
    if col == shape[0] * shape[1]:
        plt.show()

###start calculations
n_iter = 0

while n_iter < 3:

    AtNidata_sum = np.zeros(total_valid_npix, dtype='float64')
    AtNiA_sum = np.zeros((total_valid_npix, total_valid_npix), dtype='float64')
    weights = np.zeros(len(data_fns))#synchrotron scaling, [divide data by weight and mult Ni and AtNiA by weight**2], or [multiply A by weight and AtNiA by weight**2]
    freqs = np.zeros(len(data_fns))

    for i, (Q, data_fn, A_fn, AtNiA_fn) in enumerate(zip(Qs, data_fns, A_fns, AtNiA_fns)):
        child_mask = None
        for instrument in INSTRUMENTS:
            if datatags[instrument] in data_fn:
                child_mask = child_masks[instrument]
                valid_npix = valid_npixs[instrument]
                break
        if child_mask is None:
            raise Exception('Logic error instrument not found.')
        print Q, np.sum(child_mask), len(child_mask)
        sys.stdout.flush()

        data_file = np.load(data_fn)
        freq = data_file['freq']
        freqs[i] = freq
        data = data_file['data']
        Ni = data_file['Ni'] / noise_scales[instrument]**2


        nUBL = int(A_fn.split(Q + 'A_dI_u')[1].split('_')[0])
        At = np.zeros((total_valid_npix, len(data)), dtype='float32')
        At[child_mask] = (np.fromfile(A_fn, dtype='float32').reshape((len(data)/2, valid_npix + 4*nUBL, 2))[:, :valid_npix]).transpose((1, 2, 0)).reshape((valid_npix, len(data)))
        if n_iter == 0:
            weights[i] = (freq / standard_freq)**-2.7
        else:
            Ax = At.transpose().dot(result)
            weights[i] = np.sum(data * Ni * Ax) / np.sum(Ax * Ni * Ax)
        AtNidata_sum += At.dot(data * Ni) * weights[i]
        AtNiA_sum[np.ix_(child_mask, child_mask)] += np.fromfile(AtNiA_fn, dtype='float64').reshape((valid_npix, valid_npix)) * weights[i]**2 / noise_scales[instrument]**2
    plt.plot(sorted(freqs), weights[np.argsort(freqs)])
    for reg in 10.**np.arange(-7, -6, .5):
        print "trying", reg
        sys.stdout.flush()
        try:
            AtNiAi = sv.InverseCholeskyMatrix(AtNiA_sum + np.eye(total_valid_npix) * reg)
            break
        except TypeError:
            continue
    result = AtNiAi.dotv(AtNidata_sum)
    n_iter += 1
plt.show()


plot_IQU(result, '+'.join(INSTRUMENTS), 1, shape=(2, 2), coord='CG')

#####GSM reg version
I_supress = 25.
S_diag = fake_solution_map ** 2 / I_supress
AtNiASi = np.copy(AtNiA_sum)
AtNiASi.shape = (len(AtNiASi) ** 2)
AtNiASi[::len(S_diag) + 1] += 1./S_diag
AtNiASi.shape = (len(S_diag), len(S_diag))
AtNiASii = sv.InverseCholeskyMatrix(AtNiASi)
AtNiSidata = AtNidata_sum + fake_solution_map / S_diag
combined_result = AtNiASii.dotv(AtNiSidata)

plot_IQU(combined_result, '+'.join(INSTRUMENTS) + '+GSM', 2, shape=(2, 2), coord='CG')



###parkes
parkes_150 = fit.read("/home/omniscope/data/polarized foregrounds/parkes_150mhz.bin")[0]
parkes_150[:, :-1] = np.roll(parkes_150[:, :-1], 180, axis=1)[:, ::-1]
parkes_150[:, -1] = parkes_150[:, 0]
parkes_150[parkes_150 > 7.e3] = -1e-9
parkes_150[parkes_150 <= 0] = -1e-9
parkes_150 = sv.equirectangular2heapix(parkes_150, nside_standard)
parkes_150[parkes_150 <= 0] = np.nan
hpv.mollview(np.log10(parkes_150), nest=True, min=0, max=4, sub=(2, 2, 3), title='parkes150MHz')

####GSM####
plot_IQU(fake_solution_map, 'GSM', 4, shape=(2, 2), coord='CG')






clean = False
if clean:#take abt 10 min, not quite working. ringing seems not caused by wiener filter??

    bright_pt_mask = np.abs(result/sizes) > np.percentile(np.abs(result/sizes), 99)
    bright_pt_indices = np.arange(valid_npix)[bright_pt_mask]
    psf = AtNiAi.dotM(AtNiA_sum.dot(np.eye(valid_npix, dtype='float64')[:, bright_pt_mask]))

    clean_stop = np.percentile(np.abs(result/sizes), 95)
    niter = 0
    clean_residue = np.copy(result)
    clean_accumulate = np.zeros_like(clean_residue)
    step_size = .2
    while np.max(np.abs((clean_residue / sizes)[bright_pt_mask])) > clean_stop and niter < 10000:
        niter += 1
        ipix_bright = np.argmax(np.abs((clean_residue / sizes)[bright_pt_mask]))
        ipix = bright_pt_indices[ipix_bright]
        origin_ipix_bright = np.argmax(np.abs(psf[ipix]))
        origin_ipix = bright_pt_indices[origin_ipix_bright]
        print niter, ipix_bright, (clean_residue[ipix] / sizes[ipix])

        delta = clean_residue[ipix] * step_size
        clean_residue -= delta / psf[ipix, origin_ipix_bright] * psf[:, ipix_bright]
        clean_accumulate[origin_ipix] += delta / psf[ipix, origin_ipix_bright]

    plot_IQU(result, 'MITEoR', 1, shape=(2,1), coord='CG')
    plot_IQU(clean_residue + clean_accumulate, 'MITEoR', 2, shape=(2,1), coord='CG')