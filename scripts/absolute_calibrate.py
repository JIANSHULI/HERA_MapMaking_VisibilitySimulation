import numpy as np, omnical.calibration_omni as omni, matplotlib.pyplot as plt, scipy.interpolate as si
import glob, sys, ephem, warnings
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
PI = np.pi
TPI = 2*np.pi


Q = 'q3A'#'q2C'#
delay_compression = 15
freqs_dic = {'q3A': np.arange(156., 168.5, 50./1024/delay_compression*256),'q2C':np.arange(145., 157.5, 50./1024/delay_compression*256),}
freqs = freqs_dic[Q]
lambdas = 299.792 / freqs
pick_f = 4
print freqs[pick_f]


correction_mat = np.array([[0.999973, 0.010705, -0.00263492], [-0.0105871, 0.999799, 0.0167222], [0.00142779, -0.00475158, -0.0178542]])#ubl.dot(correction_mat) gives better ubl for omniscope. this is directly from "antlocvecX5 // Transpose" in mathematica


omnifits = {}
for pol in ['xx', 'xy','yx', 'yy']:
    omnifits[pol] = [omni.load_omnifit(fname, info=fname.replace("%s.omnifit"%pol, "xx.binfo")) for fname in sorted(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*%s.omnifit"%(Q,pol)))]
nTimes = [fit.shape[0] for fit in omnifits['xx']]
nFrequencies = [int(fit[0, 0, 5]) for fit in omnifits['xx']]
nfreq = nFrequencies[0]
print nTimes, nFrequencies

jds = np.concatenate([omni.get_omnitime(fit) for fit in omnifits['xx']], axis=0)

flags = {}
for pol in ['xx', 'yy']:
    flags[pol] = np.concatenate([np.fromfile(fname, dtype = 'bool').reshape((nTimes[i], nFrequencies[i])) for i, fname in enumerate(sorted(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*%s.omniflag"%(Q, pol))))], axis=0)
flag = flags['xx']|flags['yy']

for pol in ['xx', 'xy','yx', 'yy']:
    omnifits[pol] = np.concatenate([fit[..., 6::2] + 1.j*fit[..., 7::2] for fit in omnifits[pol]], axis=0).transpose((0,2,1))
    omnifits[pol][flag] = np.nan

info = omni.read_redundantinfo(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*xx.binfo"%Q)[0])
redundancy_sort = np.argsort(info['ublcount'])

raw_vars = {}
for pol in ['xx', 'xy','yx', 'yy']:
    omnichisq = np.concatenate([omni.load_omnichisq(fname)[:,3:] for fname in sorted(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*%s.omnichisq"%(Q,pol)))], axis=0)
    if pol[0] == pol[1]:
        raw_vars[pol] = omnichisq / (info['nBaseline'] - info['nAntenna'] - info['nUBL'] + 2)
    else:
        raw_vars[pol] = omnichisq / info['nBaseline']
    raw_vars[pol] = np.outer(raw_vars[pol], 1./info['ublcount']).reshape(list(raw_vars[pol].shape) + [info['nUBL']])

print "First Round Compressing: ",#just to flag those with large compression errors (3 sigma)
compressed_data = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='complex64')
compr_error = np.zeros((4, len(omnifits['xx']), nfreq, info['nUBL']), dtype='complex64')
compr_flag = np.zeros((len(omnifits['xx']), delay_compression), dtype='bool')
compr_var = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='float32')
for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
    print pol,
    sys.stdout.flush()
    for t in range(len(omnifits['xx'])):
        d = np.copy(omnifits[pol][t])
        d[flag[t]] = 0
        compr_result = omni.deconvolve_spectra2(d, ~flag[t], (delay_compression+1)/2, var = raw_vars[pol][t,:,0], correction_weight=1e-6)
        compressed_data[p, t] = float(delay_compression)/nfreq * compr_result[0]
        compr_error[p,t] = compr_result[2]

        compr_error_bar = float(delay_compression)/nfreq * np.array([np.abs(compr_result[3][i,i]) for i in range(delay_compression)])**.5
        compr_var[p, t] = np.outer(compr_error_bar**2 * info['ublcount'][0], 1./info['ublcount'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        flag = flag | (np.linalg.norm(compr_error[0]*(nfreq/(nfreq-delay_compression))**.5, axis=-1)/np.linalg.norm(raw_vars['xx']**.5,axis=-1) > 3)
print ""

print "Second Round Compressing: ",
compressed_data = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='complex64')
compr_error = np.zeros((4, len(omnifits['xx']), nfreq, info['nUBL']), dtype='complex64')
compr_flag = np.zeros((len(omnifits['xx']), delay_compression), dtype='bool')
compr_var = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='float32')
for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
    print pol,
    sys.stdout.flush()
    for t in range(len(omnifits['xx'])):
        d = np.copy(omnifits[pol][t])
        d[flag[t]] = 0
        compr_result = omni.deconvolve_spectra2(d, ~flag[t], (delay_compression+1)/2, var = raw_vars[pol][t,:,0], correction_weight=1e-6)
        compressed_data[p, t] = float(delay_compression)/nfreq * compr_result[0]
        compr_error[p,t] = compr_result[2]

        compr_error_bar = float(delay_compression)/nfreq * np.array([np.abs(compr_result[3][i,i]) for i in range(delay_compression)])**.5
        compr_var[p, t] = np.outer(compr_error_bar**2 * info['ublcount'][0], 1./info['ublcount'])
    compr_flag = compr_flag | (compr_var[p,...,0]>1.1*np.nanmedian(compr_var[p,np.sum(~flag,axis=-1)!=0,...,0]))
print ""
def plot_u(ru, mode='phs'):
    plt.subplot('251')
    plt.imshow(flag, aspect = 1/np.e/2, interpolation='none')
    for i, pol in enumerate(['xx', 'xy','yx', 'yy']):
        plt.subplot(2,5,i+2)
        if mode == 'phs':
            plt.imshow(np.angle(omnifits[pol][..., redundancy_sort[ru]]), aspect = 1/np.e/2, interpolation='none')#, vmin = 0, vmax=.01)
        else:
            plt.imshow(np.abs(omnifits[pol][..., redundancy_sort[ru]]), aspect = 1/np.e/2, interpolation='none', vmin = 0, vmax=.01)
    plt.subplot('256')
    plt.imshow(compr_flag, aspect = 1/np.e/2/16, interpolation='none')
    for i, pol in enumerate(['xx', 'xy','yx', 'yy']):
        plt.subplot(2,5,i+7)
        plotd = np.copy(compressed_data[i, ..., redundancy_sort[ru]])
        plotd[compr_flag] = np.nan
        if mode == 'phs':
            plt.imshow(np.angle(plotd), aspect = 1/np.e/2/16, interpolation='none')#, vmin = 0, vmax=.01)
        else:
            plt.imshow(np.abs(plotd), aspect = 1/np.e/2/16, interpolation='none', vmin = 0, vmax=.01)

    plt.show()

plot_u(-1)

#############################
###get PS model
##############################
sa = ephem.Observer()
sa.pressure = 0
sa.lat = 45.297728 / 180 * PI
sa.lon = -69.987182 / 180 * PI

valid_jds = jds[~compr_flag[:, pick_f]]
lsts = np.empty_like(valid_jds)

###ps####
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
    southern_points[source]['pos'] = np.zeros((len(valid_jds), 2))

flux_func = {}
flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])
###lst###
for nt,jd in enumerate(valid_jds):
    sa.date = jd - omni.julDelta
    lsts[nt] = float(sa.sidereal_time())#on 2pi
    for source in southern_points.keys():
        southern_points[source]['body'].compute(sa)
        southern_points[source]['pos'][nt] = southern_points[source]['body'].alt, southern_points[source]['body'].az

##beam###
bnside = 16
local_beam = si.interp1d(range(110,200,10), np.concatenate([np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin'%(bnside,p), dtype='complex64').reshape((9,12*bnside**2,2)) for p in ['x', 'y']], axis=-1).transpose(0,2,1), axis=0)

beam_healpix={}
beam_healpix['x'] = abs(local_beam(freqs[pick_f])[0])**2 + abs(local_beam(freqs[pick_f])[1])**2
beam_healpix['y'] = abs(local_beam(freqs[pick_f])[2])**2 + abs(local_beam(freqs[pick_f])[3])**2

###################################
###########simulate##
cal_sources = ['cyg', 'cas']
ps_visibilities = {}
for source in cal_sources:
    phis = PI - southern_points[source]['pos'][:, 1]
    thetas = PI/2 - southern_points[source]['pos'][:, 0]
    ks = np.array([np.cos(phis)*np.sin(thetas), np.sin(phis)*np.sin(thetas), np.cos(thetas)])
    fringes = np.exp(1.j * TPI/lambdas[pick_f] * (info['ubl'].dot(correction_mat)).dot(ks)).transpose()
    ps_visibilities[source] = {}
    for pol in ['xx', 'yy']:
        b_values = hpf.get_interp_val(beam_healpix[pol[0]], thetas, phis)
        ps_visibilities[source][pol] = b_values[:, None]*fringes*flux_func[source](freqs[pick_f])/2

###################################
##########calibrate
#################################


##This part decides which lst_range and min ubl length to use
rawdata = {}
simdata = {}
rawdata['xx'] = np.copy(compressed_data[0,~compr_flag[:, pick_f],pick_f])
rawdata['yy'] = np.copy(compressed_data[3,~compr_flag[:, pick_f],pick_f])
for pol in ['xx', 'yy']:
    simdata[pol] = np.sum([ps_visibilities[source][pol] for source in cal_sources], axis = 0)
for i in range(info['nUBL']):
    pol='xx';
    plt.subplot(10,info['nUBL']/10+1,i+1)
    u=np.argsort(info['ublcount'])[i]
    plt.scatter((lsts-3)%TPI+3,np.abs(simdata[pol][:,u]), color='g')
    plt.scatter((lsts-3)%TPI+3,.45e6*np.abs(rawdata[pol][:,u]))
    plt.scatter((lsts-3)%TPI+3,np.abs(ps_visibilities['cyg'][pol][:,u]),color='c')
    plt.scatter((lsts-3)%TPI+3,np.abs(ps_visibilities['cas'][pol][:,u]), color='r')
    plt.title(info['ubl'][u]);plt.xlim(4.5,6.5)
plt.show()


cal_lst_range = [5, 6]
calibrate_ubl_length = 18.
cal_time_mask = (lsts>cal_lst_range[0]) & (lsts<cal_lst_range[1])#a True/False mask on all good data to get good data in cal time range
cal_ubl_mask = np.linalg.norm(info['ubl'], axis=1) >= calibrate_ubl_length


##calibrate cas flux and over all calibration amp

#first try fitting xx and yy seperately as a jackknife
error = {'xx':1.e9, 'yy':1.e9}
tmp_result = -1
for pol in ['xx','yy']:
    b = np.abs(rawdata[pol][cal_time_mask][:, cal_ubl_mask].flatten())
    fracs = {}
    fracs['cyg'] = 1.
    for fracs['cas'] in np.arange(.8,1.2,.001):
        A = np.abs(np.sum([fracs[source]*ps_visibilities[source][pol][cal_time_mask][:, cal_ubl_mask].flatten() for source in cal_sources], axis = 0))
        if np.linalg.norm(A * A.dot(b)/A.dot(A) - b) < error[pol]:
            tmp_result = fracs['cas']
            error[pol] = np.linalg.norm(A * A.dot(b)/A.dot(A) - b)
    print pol+' CasA fit:', tmp_result

###fit jointly to get both cas fraction and the amplitude calibrations for xx and yy
error = 1.e9
tmp_result = -1
b = {}
A = {}
fracs = {}
ampcals = {} #caldata= data * ampcal
fracs['cyg'] = 1.
for fracs['cas'] in np.arange(.8,1.2,.001):
    for pol in ['xx','yy']:
        b[pol] = np.abs(rawdata[pol][cal_time_mask][:, cal_ubl_mask].flatten())

        A[pol] = np.abs(np.sum([fracs[source]*ps_visibilities[source][pol][cal_time_mask][:, cal_ubl_mask].flatten() for source in cal_sources], axis = 0))
    if np.sum([np.linalg.norm(A[pol] * A[pol].dot(b[pol])/A[pol].dot(A[pol]) - b[pol])**2 for pol in ['xx', 'yy']]) < error:
        error = np.sum([np.linalg.norm(A[pol] * A[pol].dot(b[pol])/A[pol].dot(A[pol]) - b[pol])**2 for pol in ['xx', 'yy']])
        tmp_result = fracs['cas']
        for pol in ['xx', 'yy']:
            ampcals[pol] = 1./(A[pol].dot(b[pol])/A[pol].dot(A[pol]))
fracs['cas'] = tmp_result
print 'Joint CasA fit:', fracs['cas'], "amp calibrations:", ampcals['xx'], ampcals['yy']

###apply cas fraction and amp cals and plot again
ampdata = {}
simdata = {}
ampdata['xx'] = ampcals['xx'] * np.copy(compressed_data[0,~compr_flag[:, pick_f],pick_f])
ampdata['yy'] = ampcals['yy'] * np.copy(compressed_data[3,~compr_flag[:, pick_f],pick_f])
for pol in ['xx', 'yy']:
    simdata[pol] = np.sum([fracs[source] * ps_visibilities[source][pol] for source in cal_sources], axis = 0)
for i in range(info['nUBL']):
    pol='xx';
    plt.subplot(10,info['nUBL']/10+1,i+1)
    u=np.argsort(info['ublcount'])[i]
    plt.scatter((lsts-3)%TPI+3,np.abs(simdata[pol][:,u]), color='g')
    plt.scatter((lsts-3)%TPI+3,np.abs(ampdata[pol][:,u]))
    plt.title(info['ubl'][u]);plt.xlim(4.5,6.5)
plt.show()

###start solving for phase
#set up phase data, A is ubl vector, b is phase difference sim/data
A = []
b = []
for pol in ['xx', 'yy']:
    for u, ubl in enumerate(info['ubl']):
        if np.linalg.norm(ubl) >= calibrate_ubl_length:
            Aelement = ubl[:2]
            belement = omni.medianAngle(np.angle(simdata[pol][cal_time_mask, u] / ampdata[pol][cal_time_mask, u])[np.abs(ampdata[pol][cal_time_mask, u])>np.median(np.abs(ampdata[pol][cal_time_mask, u]))/2.])
            b = b + [belement]
            A = A + [Aelement]
#sooolve
phase_cal = omni.solve_slope(np.array(A), np.array(b), 1)
plt.hist((np.array(A).dot(phase_cal)-b + PI)%TPI-PI)
plt.title('phase fitting error');plt.show()

###############################################
##############compress in time dimension
############################################
tcompress_fac = 10
tcompress = len(jds) / tcompress_fac / 2 * 2 + 1
compressed2_data = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'complex64')
compressed2_var = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'float32')
compressed2_flag = np.zeros(tcompress, dtype='bool')
for p in range(4):
    compressed2_result = omni.deconvolve_spectra2(compressed_data[p,:,pick_f], ~compr_flag[:, pick_f], (tcompress + 1)/2, correction_weight = 1e-6, var = compr_var[p, :, pick_f, 0])
    compressed2_data[p] = compressed2_result[0] * float(tcompress) / len(jds)
    compressed2_var[p] = (float(tcompress) / len(jds)) ** 2 * np.outer(np.abs([compressed2_result[3][i, i] for i in range(tcompress)]) * info['ublcount'][0], 1./info['ublcount'])
    compressed2_flag = compressed2_flag | (compressed2_var[p,...,0] > 1.1 * np.nanmedian(compressed2_var[p,...,0]))
compressed2_jds = np.arange(jds[0], jds[-1], np.mean(jds[1:]-jds[:-1]) * len(jds) / tcompress)
compressed2_lsts = []
for jd in compressed2_jds[~compressed2_flag]:
    sa.date = jd - omni.julDelta
    compressed2_lsts = compressed2_lsts + [float(sa.sidereal_time())]
compressed2_lsts = np.array(compressed2_lsts)
#######################################
########apply amp and phase to all 4 pols
################################
calibrated_data = np.empty_like(compressed2_data[:,~compressed2_flag])
calibrated_var = np.empty_like(compressed2_var[:,~compressed2_flag])
for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
    #original_data = np.copy(compressed_data[p,~compr_flag[:, pick_f],pick_f])
    calibrated_data[p] = compressed2_data[p,~compressed2_flag] * (ampcals[pol[0]*2] * ampcals[pol[1]*2])**.5 * np.exp(1.j * info['ubl'][:,:2].dot(phase_cal))
    calibrated_var[p] = compressed2_var[p,~compressed2_flag] * (ampcals[pol[0]*2] * ampcals[pol[1]*2])

##############################
###plot calibrated data
##############################
for i in range(info['nUBL']):
    pol='xx';
    p = 0
    plt.subplot(10,info['nUBL']/10+1,i+1)
    u=np.argsort(info['ublcount'])[i]
    plt.scatter((lsts-3)%TPI+3,np.imag(simdata[pol][:,u]), color='g')
    plt.scatter((compressed2_lsts-3)%TPI+3,np.imag(calibrated_data[p,:,u]))
    plt.title(info['ubl'][u]);plt.xlim(4.5,6.5)
plt.show()

#################################
#######output data
###################################
tag = "%s_abscal"%Q
datatag = '_2015_05_09'
vartag = '_2015_05_09'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = calibrated_data.shape[1]
nf = 1
nUBL = calibrated_data.shape[2]
for p, pol in enumerate(['xx','xy','yx','yy']):

    (compressed2_lsts/TPI*24 + 1.j * freqs[pick_f]).astype('complex64').tofile(datadir + tag + '_%s_%i_%i.tf'%(pol, nt, nf))
    calibrated_data[p].astype('complex64').tofile(datadir + tag + '_%s_%i_%i'%(pol, nt, nUBL) + datatag)
    calibrated_var[p].astype('float32').tofile(datadir + tag + '_%s_%i_%i'%(pol, nt, nUBL) + vartag + '.var')
    info['ubl'].dot(correction_mat).astype('float32').tofile(datadir + tag + '_%s_%i_%i.ubl'%(pol, nUBL, 3))






