import numpy as np
import numpy.linalg as la
import omnical.calibration_omni as omni
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.fftpack as sfft
import glob, sys, ephem, warnings
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import matplotlib.pyplot as plt
import simulate_visibilities.simulate_visibilities as sv
import time
PI = np.pi
TPI = 2*np.pi

def solve_phase_degen_old(data_xx, data_yy, model_xx, model_yy, ubls, plot=False):#data should be time by ubl at single freq. data * phasegensolution = model
    if data_xx.shape != data_yy.shape or data_xx.shape != model_xx.shape or data_xx.shape != model_yy.shape or data_xx.shape[1] != ubls.shape[0]:
        raise ValueError("Shapes mismatch: %s %s %s %s, ubl shape %s"%(data_xx.shape, data_yy.shape, model_xx.shape, model_yy.shape, ubls.shape))
    A = np.zeros((len(ubls) * 2, 2))
    b = np.zeros(len(ubls) * 2)

    nrow = 0
    for p, (data, model) in enumerate(zip([data_xx, data_yy], [model_xx, model_yy])):
        for u, ubl in enumerate(ubls):
            amp_mask = (np.abs(data[:, u]) > (np.nanmedian(np.abs(data[:, u])) / 2.))
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



sa = ephem.Observer()
sa.pressure = 0
sa.lat = 45.297728 / 180 * PI
sa.lon = -69.987182 / 180 * PI

omnifitss = {}
lstss = {}
infos = {}
omnivarss = {}
for q, Q in enumerate(['q2C', 'q2AL']):

    fit_cas = True
    frac_cas = .9
    delay_compression = 15
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
        omnifits[pol][flag] = np.nan
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


    lsts = np.empty_like(jds)
    for nt,jd in enumerate(jds):
        sa.date = jd - omni.julDelta
        lsts[nt] = float(sa.sidereal_time())#on 2pi
    lsts = (lsts - 2)%(TPI) + 2

    omnifitss[Q] = omnifits
    omnivarss[Q] = raw_vars
    lstss[Q] = lsts
    infos[Q] = info

###merge ubls
common_ubl_indices = {}
common_ubl_conjugates = {}
for key in infos.keys():
    common_ubl_indices[key] = []
    common_ubl_conjugates[key] = []
for u, ubl in enumerate(infos[infos.keys()[0]]['ubl']):
    diffnorm = la.norm(infos[infos.keys()[1]]['ubl'] - ubl, axis=-1)
    if np.min(diffnorm) < .1:
        common_ubl_indices[infos.keys()[0]] += [u]
        common_ubl_indices[infos.keys()[1]] += [u]
        common_ubl_conjugates[infos.keys()[0]] += [False]
        common_ubl_conjugates[infos.keys()[1]] += [False]
    else:
        diffnorm = la.norm(-infos[infos.keys()[1]]['ubl'] - ubl, axis=-1)
        if np.min(diffnorm) < .1:
            common_ubl_indices[infos.keys()[0]] += [u]
            common_ubl_indices[infos.keys()[1]] += [u]
            common_ubl_conjugates[infos.keys()[0]] += [False]
            common_ubl_conjugates[infos.keys()[1]] += [True]

for key in infos.keys():
    for pol in ['xx', 'xy', 'yx', 'yy']:
        omnifitss[key][pol] = omnifitss[key][pol][..., common_ubl_indices[key]] * np.exp(1.j * np.pi * np.array(common_ubl_conjugates[key]))
        omnivarss[key][pol] = omnivarss[key][pol][..., common_ubl_indices[key]]
ubls = infos[infos.keys()[0]]['ubl'][common_ubl_indices[infos.keys()[0]]]

###get same time steps
for i in range(2):
    if np.round(np.mean(lstss[omnifitss.keys()[1-i]][1:] - lstss[omnifitss.keys()[1-i]][:-1]) / np.mean(lstss[omnifitss.keys()[i]][1:] - lstss[omnifitss.keys()[i]][:-1])) == 2.:
        lstss[omnifitss.keys()[i]] = lstss[omnifitss.keys()[i]][:-1:2]
        half_len_t = len(lstss[omnifitss.keys()[i]])
        dshape = omnifitss[omnifitss.keys()[i]]['xx'].shape
        for pol in ['xx', 'xy', 'yx', 'yy']:
            omnifitss[omnifitss.keys()[i]][pol] = np.nanmean(omnifitss[omnifitss.keys()[i]][pol].reshape((half_len_t, 2, dshape[1], dshape[2])), axis=1)
            omnivarss[omnifitss.keys()[i]][pol] = np.nanmean(omnivarss[omnifitss.keys()[i]][pol].reshape((half_len_t, 2, dshape[1], dshape[2])), axis=1) / 2

###start matching two chunks
for i in range(2):
    if (lstss[omnifitss.keys()[1-i]][[0, -1]] < lstss[omnifitss.keys()[i]][[0, -1]]).all():#pick the later chunk of the 2, and match it to the previous chunk
        overlap_start_index = np.argmin(np.abs(lstss[omnifitss.keys()[1-i]] - lstss[omnifitss.keys()[i]][0]))
        overlap_len = len(lstss[omnifitss.keys()[1-i]]) - overlap_start_index


        #fit for phase first, all freq at once
        phase_cal = omni.solve_phase_degen(omnifitss[omnifitss.keys()[i]]['xx'][:overlap_len].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[i]]['yy'][:overlap_len].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[1-i]]['xx'][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[1-i]]['yy'][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2])), ubls, [3, 3, 1e3], plot=False)
        print phase_cal,
        if not np.isnan(phase_cal).any():
            for pol in ['xx', 'xy', 'yx', 'yy']:
                omnifitss[omnifitss.keys()[i]][pol] *= np.exp(1.j * ubls[:, :2].dot(phase_cal))
        print omni.solve_phase_degen(omnifitss[omnifitss.keys()[i]]['xx'][:overlap_len].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[i]]['yy'][:overlap_len].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[1-i]]['xx'][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2])), omnifitss[omnifitss.keys()[1-i]]['yy'][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2])), ubls, [3, 3, 1e3], plot=False)


        #fit for amp
        amps = {}
        for pol in ['xx', 'yy']:
            a = np.concatenate((np.real(omnifitss[omnifitss.keys()[i]][pol][:overlap_len].reshape((overlap_len * dshape[1], dshape[2]))), np.imag(omnifitss[omnifitss.keys()[i]][pol][:overlap_len].reshape((overlap_len * dshape[1], dshape[2]))))).flatten()
            b = np.concatenate((np.real(omnifitss[omnifitss.keys()[1-i]][pol][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2]))), np.imag(omnifitss[omnifitss.keys()[1-i]][pol][overlap_start_index:].reshape((overlap_len * dshape[1], dshape[2]))))).flatten()
            nan_mask = np.isnan(a+b)
            a = a[~nan_mask]
            b = b[~nan_mask]
            amps[pol[0]] = (a.dot(b) / a.dot(a))**.5

        print amps
        if not np.isnan([amps['x'], amps['y']]).any():
            for pol in ['xx', 'xy', 'yx', 'yy']:
                omnifitss[omnifitss.keys()[i]][pol] *= (amps[pol[0]] * amps[pol[1]])
                omnivarss[omnifitss.keys()[i]][pol] *= (amps[pol[0]] * amps[pol[1]])**2
        break

for q, Q in enumerate(['q2C', 'q2AL']):

    for p, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
        plt.subplot(2,4,q*4 + p+1)
        plt.imshow(np.angle(omnifitss[Q][pol][...,0]), interpolation='none', extent=[freqs[0], freqs[-1], lstss[Q][0], lstss[Q][-1]], origin='lower', aspect='auto');plt.colorbar();plt.title(pol)

plt.show()



f = 255
fun = np.real
for u, ubl in enumerate(ubls):
    plt.subplot(8, 10, u+1)
    plt.plot(lstss[omnifitss.keys()[0]], fun(omnifitss[omnifitss.keys()[0]]['xx'][:, f, u]))
    plt.plot(lstss[omnifitss.keys()[1]], fun(omnifitss[omnifitss.keys()[1]]['xx'][:, f, u]))
    plt.title(ubl)
plt.show()