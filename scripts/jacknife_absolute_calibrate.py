import numpy as np
import numpy.linalg as la
import omnical.calibration_omni as omni
import matplotlib.pyplot as plt
import scipy.interpolate as si
import glob, sys, ephem, warnings
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import matplotlib.pyplot as plt
import simulate_visibilities.simulate_visibilities as sv
import time
PI = np.pi
TPI = 2*np.pi

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

    if plot:
        plt.hist((np.array(A).dot(phase_cal)-b + PI)%TPI-PI)
        plt.title('phase fitting error')
        plt.show()

    #sooolve
    return omni.solve_slope(np.array(A), np.array(b), 1)

Q = 'q3AL'#'q0C'#'q3A'#'q2C'#

fit_cas = True
frac_cas = .9
delay_compression = 15
ntjack = 4
compress_method = 'average'
pick_fs = range(delay_compression)

freqs_dic = {
    'q0C': np.arange(136., 123.5, -50./1024/delay_compression*256)[::-1],
    'q0CL': np.arange(136., 123.5, -50./1024/delay_compression*256)[::-1],
    'q0AL': np.arange(136., 123.5, -50./1024/delay_compression*256)[::-1],
    'q1A': np.arange(146., 133.5, -50./1024/delay_compression*256)[::-1],
    'q1AL': np.arange(146., 133.5, -50./1024/delay_compression*256)[::-1],
    'q3A': np.arange(156., 168.5, 50./1024/delay_compression*256),
    'q3AL': np.arange(156., 168.5, 50./1024/delay_compression*256),
    'q4AL': np.arange(167., 179.5, 50./1024/delay_compression*256),
    'q4AL': np.arange(167., 179.5, 50./1024/delay_compression*256),
    'qC3A': np.arange(156., 168.5, 50./1024/delay_compression*256),
    'qC3B': np.arange(156., 168.5, 50./1024/delay_compression*256),
    'q2C': np.arange(145., 157.5, 50./1024/delay_compression*256),
    'q2CL': np.arange(145., 157.5, 50./1024/delay_compression*256),
    'q2AL': np.arange(145., 157.5, 50./1024/delay_compression*256),
}
freqs = freqs_dic[Q]
lambdas = 299.792 / freqs
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


for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
    omnifits[pol] = np.concatenate([fit[..., 6::2] + 1.j*fit[..., 7::2] for fit in omnifits[pol]], axis=0).transpose((0,2,1))
    if pol == 'yy':
        plt.subplot(1,2,1)
        plt.imshow(np.angle(omnifits[pol][...,0]), aspect=.1, interpolation='none');plt.colorbar();plt.title('Before flagging')
    omnifits[pol][flag] = np.nan
plt.subplot(1,2,2)
plt.imshow(np.angle(omnifits['yy'][...,0]), aspect=.1, interpolation='none');plt.colorbar();plt.title('Flagged')
plt.show()
info = omni.read_redundantinfo(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*xx.binfo"%Q)[0])
redundancy_sort = np.argsort(info['ublcount'])

raw_vars = {}
for pol in ['xx', 'xy', 'yx', 'yy']:
    omnichisq = np.concatenate([omni.load_omnichisq(fname)[:,3:] for fname in sorted(glob.glob("/home/omniscope/data/X5/2015calibration/*_%s_*%s.omnichisq"%(Q,pol)))], axis=0)
    if pol[0] == pol[1]:
        raw_vars[pol] = omnichisq / (info['nBaseline'] - info['nAntenna'] - info['nUBL'] + 2)
    else:
        raw_vars[pol] = omnichisq / info['nBaseline']
    raw_vars[pol] = np.outer(raw_vars[pol], 1./info['ublcount']).reshape(list(raw_vars[pol].shape) + [info['nUBL']])

#plot b4 pi jump fixes
plt.subplot(2,2,1)
plt.imshow(np.angle(omnifits['xy'][..., redundancy_sort[-1]]), interpolation='none', aspect=1/np.e/3)
plt.subplot(2,2,2)
plt.imshow(np.angle(omnifits['xy'][..., redundancy_sort[-10]]), interpolation='none', aspect=1/np.e/3)

if 'q3' not in Q:
    ####process the pi jump in xy and yx
    per_file_avg = {}
    pi_flip = {}
    flip_use_ubls = redundancy_sort[-10:]
    for pol in ['xy', 'yx']:
        tmp_t = 0
        per_file_avg[pol] = np.zeros((len(nTimes), omnifits[pol].shape[1], len(flip_use_ubls)), dtype='complex64')
        for nfile in range(len(nTimes)):
            per_file_avg[pol][nfile] = np.nanmean(omnifits[pol][tmp_t:tmp_t+nTimes[nfile]], axis=0)[..., flip_use_ubls]
            tmp_t += nTimes[nfile]

        norms = np.linalg.norm(per_file_avg[pol][:-1], axis=-1)
        pi_flip[pol] = np.sum(per_file_avg[pol][1:] * np.conjugate(per_file_avg[pol][:-1]), axis=-1) / norms**2

        #redo the ones that have previous time as nan
        for nfile in range(1, pi_flip[pol].shape[0]):
            for tmp_f in range(pi_flip[pol].shape[1]):
                if np.isnan(norms[nfile, tmp_f]) and not np.isnan(norms[:nfile, tmp_f]).all():
                    prev_nfile = np.arange(nfile)[~np.isnan(norms[:nfile, tmp_f])][-1]
                    pi_flip[pol][nfile, tmp_f] = np.sum(per_file_avg[pol][1 + nfile, tmp_f] * np.conjugate(per_file_avg[pol][prev_nfile, tmp_f])) / norms[prev_nfile, tmp_f]**2


    pi_flip_both_pol = (np.abs(np.angle(pi_flip[pol])%(2*np.pi) - np.pi) < np.pi/2) & (np.abs(np.angle(pi_flip[pol])%(2*np.pi) - np.pi) < np.pi/2)
    final_pi_flip = np.zeros((len(nTimes), omnifits[pol].shape[1]), dtype=bool)
    for nfile in range(1, len(nTimes)):
        final_pi_flip[nfile] = final_pi_flip[nfile - 1] ^ pi_flip_both_pol[nfile - 1]


    for pol in ['xy', 'yx']:
        tmp_t = 0
        for nfile in range(0, len(nTimes)):
            omnifits[pol][tmp_t:tmp_t+nTimes[nfile]] *= ((final_pi_flip[nfile] - .5) * -2)[None, :, None]
            tmp_t += nTimes[nfile]


    #######deal with pi flip over frequency

    shortest_ubl_xy_avg = np.angle(np.nanmean(omnifits['xy'][:, :, redundancy_sort[-1]], axis=0))
    shortest_ubl_yx_avg = np.angle(np.nanmean(omnifits['yx'][:, :, redundancy_sort[-1]], axis=0))
    xy_flip = ((shortest_ubl_xy_avg % np.pi) - shortest_ubl_xy_avg) > .1
    yx_flip = ((shortest_ubl_yx_avg % np.pi) - shortest_ubl_yx_avg) > .1
    sign_list = -((xy_flip & xy_flip) * 2 - 1)
    omnifits['xy'] *= sign_list[None, :, None]
    omnifits['yx'] *= sign_list[None, :, None]

#plot after pi jump fixes
plt.subplot(2,2,3)
plt.imshow(np.angle(omnifits['xy'][..., redundancy_sort[-1]]), interpolation='none', aspect=1/np.e/3)
plt.subplot(2,2,4)
plt.imshow(np.angle(omnifits['xy'][..., redundancy_sort[-10]]), interpolation='none', aspect=1/np.e/3)
plt.show()

####start compression
compressed_data = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='complex64')
compr_flag = np.zeros((len(omnifits['xx']), delay_compression), dtype='bool')
compr_var = np.zeros((4, len(omnifits['xx']), delay_compression, info['nUBL']), dtype='float32')

if compress_method == 'deconvolve':
    print "First Round Compressing: ",#just to flag those with large compression errors (3 sigma)
    compr_error = np.zeros((4, len(omnifits['xx']), nfreq, info['nUBL']), dtype='complex64')

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
else:
    fcompr_fac = omnifits['xx'].shape[1] / delay_compression
    pre_fcompr_data = np.zeros((4, omnifits['xx'].shape[0], delay_compression, fcompr_fac, info['nUBL']), dtype='complex64')
    pre_fcompr_var = np.zeros((4, omnifits['xx'].shape[0], delay_compression, fcompr_fac, info['nUBL']), dtype='float32')
    pre_fcompr_flag = flag[:, :delay_compression * fcompr_fac].reshape((omnifits['xx'].shape[0], delay_compression, fcompr_fac))
    for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
        pre_fcompr_data[p] = omnifits[pol][:, :delay_compression * fcompr_fac].reshape((omnifits[pol].shape[0], delay_compression, fcompr_fac, info['nUBL']))
        pre_fcompr_data[p][pre_fcompr_flag] = 0
        pre_fcompr_var[p] = raw_vars[pol][:, :delay_compression * fcompr_fac].reshape((omnifits[pol].shape[0], delay_compression, fcompr_fac, info['nUBL']))
        pre_fcompr_var[p][pre_fcompr_flag] = 0

    compr_count = np.sum(~pre_fcompr_flag, axis=2)
    compr_flag = (compr_count == 0)
    compressed_data = np.sum(pre_fcompr_data, axis=3) / compr_count[None, :, :, None]
    compr_var = np.sum(pre_fcompr_var, axis=3) / compr_count[None, :, :, None]**2





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
    for i, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
        plt.subplot(2,5,i+7)
        plotd = np.copy(compressed_data[i, ..., redundancy_sort[ru]])
        plotd[compr_flag] = np.nan
        if mode == 'phs':
            plt.imshow(np.angle(plotd), aspect = 1/np.e/2/16, interpolation='none')#, vmin = 0, vmax=.01)
        else:
            plt.imshow(np.abs(plotd), aspect = 1/np.e/2/16, interpolation='none', vmin = 0, vmax=.01)

    plt.show()

plot_u(-1)

cyg_cas_iquv = np.zeros((len(pick_fs), 2, 4))
cyg_cas_iquv_std = np.zeros((len(pick_fs), 2, 4))
psols = [{} for i in range(ntjack)]
flipsols = {}
realb_fits = {}
flipb_fits = {}
perrors = {}
fliperrors = {}
phase_degens = {}
flipphase_degens = {}
psolstds = {}
calibrated_results = {}
for pick_f_i, pick_f in enumerate(pick_fs):
    print "########################################"
    print pick_f, freqs[pick_f]
    try:
        #############################
        ###get PS model
        ##############################
        sa = ephem.Observer()
        sa.pressure = 0
        sa.lat = 45.297728 / 180 * PI
        sa.lon = -69.987182 / 180 * PI

        valid_jds = jds[~compr_flag[:, pick_f]]
        if len(valid_jds) == 0:
            continue
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
        bnside = 256
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
        if len(pick_fs) == 1:
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


        cal_lst_range = [5., 6.]
        if Q[:2] == 'qC':
            calibrate_ubl_length = 10.
        else:
            calibrate_ubl_length = 2600 / freqs[pick_f] #18.
        cal_time_mask = (lsts>cal_lst_range[0]) & (lsts<cal_lst_range[1])#a True/False mask on all good data to get good data in cal time range
        cal_ubl_mask = np.linalg.norm(info['ubl'], axis=1) >= calibrate_ubl_length

        fracs = {}
        fracs['cyg'] = 1.
        ##calibrate cas flux and over all calibration amp
        if fit_cas:
            #first try fitting xx and yy seperately as a jackknife
            cas_frac_options = np.arange(.1, 3, .01)
            error = {'xx':1.e9, 'yy':1.e9}
            tmp_result = -1
            for pol in ['xx','yy']:
                b = np.abs(rawdata[pol][cal_time_mask][:, cal_ubl_mask].flatten())
                fracs = {}
                fracs['cyg'] = 1.
                for fracs['cas'] in cas_frac_options:
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

            ampcals = {} #caldata= data * ampcal

            for fracs['cas'] in cas_frac_options:
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
        else:
            fracs['cas'] = frac_cas
            for pol in ['xx','yy']:
                b[pol] = np.abs(rawdata[pol][cal_time_mask][:, cal_ubl_mask].flatten())

                A[pol] = np.abs(np.sum([fracs[source]*ps_visibilities[source][pol][cal_time_mask][:, cal_ubl_mask].flatten() for source in cal_sources], axis = 0))
            error = np.sum([np.linalg.norm(A[pol] * A[pol].dot(b[pol])/A[pol].dot(A[pol]) - b[pol])**2 for pol in ['xx', 'yy']])
            for pol in ['xx', 'yy']:
                ampcals[pol] = 1./(A[pol].dot(b[pol])/A[pol].dot(A[pol]))

        ###apply cas fraction and amp cals and plot again
        ampdata = {}
        simdata = {}
        ampdata['xx'] = ampcals['xx'] * np.copy(compressed_data[0,~compr_flag[:, pick_f],pick_f])
        ampdata['yy'] = ampcals['yy'] * np.copy(compressed_data[3,~compr_flag[:, pick_f],pick_f])
        for pol in ['xx', 'yy']:
            simdata[pol] = np.sum([fracs[source] * ps_visibilities[source][pol] for source in cal_sources], axis = 0)

        if len(pick_fs) == 1:
            for i in range(info['nUBL']):
                pol = 'xx'
                plt.subplot(10,info['nUBL']/10+1,i+1)
                u=np.argsort(info['ublcount'])[i]
                plt.scatter((lsts-3)%TPI+3,np.abs(simdata[pol][:,u]), color='g')
                plt.scatter((lsts-3)%TPI+3,np.abs(ampdata[pol][:,u]))
                plt.title(info['ubl'][u]);plt.xlim(4.5,6.5)

            plt.show()

        ###start solving for phase

        #sooolve
        phase_cal = solve_phase_degen(ampdata['xx'][np.ix_(cal_time_mask, cal_ubl_mask)], ampdata['yy'][np.ix_(cal_time_mask, cal_ubl_mask)], simdata['xx'][np.ix_(cal_time_mask, cal_ubl_mask)], simdata['yy'][np.ix_(cal_time_mask, cal_ubl_mask)], info['ubl'][cal_ubl_mask], plot=(len(pick_fs) == 1))


        ###############################################
        ##############compress in time dimension
        ############################################
        tcompress_fac = int(np.round(6.22e-5 * 25 / np.mean(jds[1:]-jds[:-1])))#25

        if compress_method == 'deconvolve':
            tcompress = len(jds) / tcompress_fac / 2 * 2 + 1
            compressed2_data = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'complex64')
            compressed2_var = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'float32')
            compressed2_flag = np.zeros(tcompress, dtype='bool')
            for p in range(4):
                compressed2_result = omni.deconvolve_spectra2(compressed_data[p,:,pick_f], ~compr_flag[:, pick_f], (tcompress + 1)/2, correction_weight = 1e-6, var = compr_var[p, :, pick_f, 0])
                compressed2_data[p] = compressed2_result[0] * float(tcompress) / len(jds)
                compressed2_var[p] = (float(tcompress) / len(jds)) ** 2 * np.outer(np.abs([compressed2_result[3][i, i] for i in range(tcompress)]) * info['ublcount'][0], 1./info['ublcount'])
                if 'L' in Q:
                    compressed2_flag = compressed2_flag | (compressed2_var[p,...,0] > 4 * np.nanmin(compressed2_var[p,...,0]))
                else:
                    compressed2_flag = compressed2_flag | (compressed2_var[p,...,0] > 1.1 * np.nanmedian(compressed2_var[p,...,0]))
        else:#plain average.
            tcompress = len(jds) / tcompress_fac
            # compressed2_data = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'complex64')
            # compressed2_var = np.empty((4, tcompress, compressed_data.shape[-1]), dtype = 'float32')
            # compressed2_flag = np.zeros(tcompress, dtype='bool')

            pre_average_data = np.copy(compressed_data[:, :tcompress * tcompress_fac, pick_f])
            pre_average_var = np.copy(compr_var[:, :tcompress * tcompress_fac, pick_f])

            pre_average_data[:, compr_flag[:tcompress * tcompress_fac, pick_f]] = 0
            pre_average_var[:, compr_flag[:tcompress * tcompress_fac, pick_f]] = 0

            pre_average_count = np.sum((~compr_flag[:tcompress * tcompress_fac, pick_f]).reshape((tcompress, tcompress_fac)), axis=1)

            compressed2_data = np.sum(pre_average_data.reshape((4, tcompress, tcompress_fac, pre_average_data.shape[-1])), axis=2) / pre_average_count[None, :, None]
            compressed2_var = np.sum(pre_average_var.reshape((4, tcompress, tcompress_fac, pre_average_data.shape[-1])), axis=2) / pre_average_count[None, :, None]**2
            compressed2_flag = (pre_average_count == 0)

        # compressed2_jds = np.arange(jds[0], jds[-1], np.min(jds[1:]-jds[:-1]) * len(jds) / tcompress)
        compressed2_jds = jds[tcompress_fac/2::tcompress_fac][:tcompress]
        compressed2_lsts = []
        for jd in compressed2_jds[~compressed2_flag]:
            sa.date = jd - omni.julDelta
            compressed2_lsts = compressed2_lsts + [float(sa.sidereal_time())]
        compressed2_lsts = np.array(compressed2_lsts)

        #######################################
        ########apply amp and phase to all 4 pols
        ################################
        calibrated_data = np.empty_like(compressed2_data[:, ~compressed2_flag])
        calibrated_var = np.empty_like(compressed2_var[:, ~compressed2_flag])
        for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
            #original_data = np.copy(compressed_data[p,~compr_flag[:, pick_f],pick_f])
            calibrated_data[p] = compressed2_data[p,~compressed2_flag] * (ampcals[pol[0]*2] * ampcals[pol[1]*2])**.5 * np.exp(1.j * info['ubl'][:, :2].dot(phase_cal))
            calibrated_var[p] = compressed2_var[p,~compressed2_flag] * (ampcals[pol[0]*2] * ampcals[pol[1]*2])

        if len(pick_fs) == 1:
        ##############################
        ###plot calibrated data
        ##############################
            for i in range(info['nUBL']):
                pol = 'xx'
                p = 0
                plt.subplot(10,info['nUBL']/10+1,i+1)
                u=np.argsort(info['ublcount'])[i]
                plt.scatter((lsts-3)%TPI+3,np.imag(simdata[pol][:,u]), color='g')
                plt.scatter((compressed2_lsts-3)%TPI+3,np.imag(calibrated_data[p,:,u]))
                plt.title(info['ubl'][u])
                plt.xlim(4.5,6.5)

            plt.show()

        ###simulate polarized##############
        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, sa.lat])  # self.zenithequ
        beam_heal_equ = np.array(
            [sv.rotate_healpixmap(beam_healpixi, 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for
             beam_healpixi in local_beam(freqs[pick_f])])


        time_points = [cal_lst_range[0]]
        for ijack in range(1, ntjack + 1):
            time_points += [cal_lst_range[0] + (cal_lst_range[1] - cal_lst_range[0]) * ijack / ntjack]

        for jack in range(ntjack):
            pcal_time_mask = (compressed2_lsts >= time_points[jack]) & (compressed2_lsts < time_points[jack + 1])
            print "Computing polarized point sources matrix..."
            sys.stdout.flush()
            Apol = np.empty((np.sum(cal_ubl_mask), 4 * np.sum(pcal_time_mask), 4, len(cal_sources)), dtype='complex64')
            timer = time.time()
            for n, source in enumerate(['cyg', 'cas']):
                ra = southern_points[source]['body']._ra
                dec = southern_points[source]['body']._dec

                Apol[..., n] = vs.calculate_pol_pointsource_visibility(ra, dec, info['ubl'][cal_ubl_mask].dot(correction_mat), freqs[pick_f], beam_heal_equ=beam_heal_equ,
                                                                    tlist=compressed2_lsts[pcal_time_mask] / TPI * 24.).dot(
                    [[.5, .5, 0, 0], [0, 0, .5, .5j], [0, 0, .5, -.5j], [.5, -.5, 0, 0]])

            pcal_pol_mask = np.array([True, True, True, True])#for debugging when only want to allow certain pols
            Apol = np.conjugate(Apol).reshape((np.sum(cal_ubl_mask), 4, np.sum(pcal_time_mask), 4, len(cal_sources)))[:, pcal_pol_mask].reshape((np.sum(cal_ubl_mask), np.sum(pcal_pol_mask) * np.sum(pcal_time_mask), 4, len(cal_sources)))
            #TODO this neeeds fix otherwhere!

            Ni = 1 / np.transpose(calibrated_var[np.ix_(pcal_pol_mask, pcal_time_mask)], (2,0,1))[cal_ubl_mask]

            realA = np.zeros((2 * Apol.shape[0] * Apol.shape[1], Apol.shape[2] * Apol.shape[3] + 2 * np.sum(cal_ubl_mask) * np.sum(pcal_pol_mask)), dtype='float32')
            for coli, ncol in enumerate(range(Apol.shape[2] * Apol.shape[3], realA.shape[1])):
                realA[coli * np.sum(pcal_time_mask): (coli + 1) * np.sum(pcal_time_mask), ncol] = 1
            realA[:, :Apol.shape[2] * Apol.shape[3]] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2] * Apol.shape[3]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2] * Apol.shape[3])))), axis=0)#consider only include non-V? doesnt seem to change answer much

            realNi = np.concatenate((Ni.flatten() * 2, Ni.flatten() * 2))
            realAtNiAinv = np.linalg.pinv(np.einsum('ji,j,jk->ik', realA, realNi, realA))

            b = np.transpose(calibrated_data[np.ix_(pcal_pol_mask, pcal_time_mask)], (2, 0, 1))[cal_ubl_mask]
            phase_degen_niter = 0
            phase_degen2 = np.zeros(2)
            phase_degen_iterative = np.zeros(2)
            while (phase_degen_niter < 50 and np.linalg.norm(phase_degen_iterative) > 1e-5) or phase_degen_niter == 0:
                phase_degen_niter += 1
                b = b * np.exp(1.j * info['ubl'][cal_ubl_mask][:, :2].dot(phase_degen_iterative))[:, None, None]
                realb = np.concatenate((np.real(b.flatten()), np.imag(b.flatten())))

                psol = realAtNiAinv.dot(np.transpose(realA).dot(realNi * realb))
                realb_fit = realA.dot(psol)
                perror = ((realb_fit - realb) * (realNi**.5)).reshape((2, np.sum(cal_ubl_mask), np.sum(pcal_pol_mask), np.sum(pcal_time_mask)))

                bfit = realb_fit.reshape((2, np.sum(cal_ubl_mask), np.sum(pcal_pol_mask), np.sum(pcal_time_mask)))
                bfit = bfit[0] + 1.j * bfit[1]
                phase_degen_iterative = solve_phase_degen(np.transpose(b[:, 0]), np.transpose(b[:, -1]), np.transpose(bfit[:, 0]), np.transpose(bfit[:, -1]), info['ubl'][cal_ubl_mask])
                phase_degen2 += phase_degen_iterative
                # print phase_degen_niter, phase_degen2, np.linalg.norm(perror)

            psols[jack][pick_f] = psol

    except Exception:
        print "&&&ERROR&&&&&&"

iquvs = [np.zeros((len(freqs), 2, 4)) for i in range(len(psols))]

for tjack in range(len(psols)):
    for pick_f in psols[tjack].keys():
        iquvs[tjack][pick_f] = np.transpose(psols[tjack][pick_f][:8].reshape((4, 2)))

plt.subplot(2,1,1)
for s in [0, 1]:
    for tjack in range(len(psols)):
        plt.plot(freqs, la.norm(iquvs[tjack][:, s, 1:3], axis=-1) / iquvs[tjack][:, s, 0], ['b', 'g'][s] + ['^', 'o', 's', 'v'][tjack])
plt.ylim([0, .5])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Fraction")
plt.subplot(2,1,2)
for s in [0, 1]:
    for tjack in range(len(psols)):
        plt.plot(freqs, np.angle(iquvs[tjack][:, s, 1] + 1.j * iquvs[tjack][:, s, 2])%TPI / 2. * (180./PI), ['b', 'g'][s] + ['^', 'o', 's', 'v'][tjack])
plt.ylim([0, 180])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Angle (degree)")
plt.show()