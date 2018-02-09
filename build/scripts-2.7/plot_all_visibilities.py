import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, glob
import aipy as ap
import matplotlib.pyplot as plt
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import omnical.calibration_omni as omni


vis_Qs = ["q0AL_*_abscal", "q0C_*_abscal", "q1AL_*_abscal", "q2AL_*_abscal", "q2C_*_abscal", "q3AL_*_abscal", "q4AL_*_abscal"]  # L stands for lenient in flagging
datatag = '_2016_01_20_avg2_unpollock'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
vis_tags = []
for vis_Q in vis_Qs:
    filenames = glob.glob(datadir + vis_Q + '_xx*' + datatag)
    vis_tags = vis_tags + [os.path.basename(fn).split('_xx')[0] for fn in filenames]


for p in ['x']:#, 'y']:
    vis_data = {}
    ubls = {}
    freqs = {}
    ts = {}
    for tag in vis_tags:
        print tag,
        nf = 1
        pol = p+p
        data_filename = glob.glob(datadir + tag + '_%s%s_*_*'%(p, p) + datatag)[0]
        nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('%s%s_'%(p, p))[-1]
        nt = int(nt_nUBL.split('_')[0])
        nUBL = int(nt_nUBL.split('_')[1])

        #tf file
        tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
        tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt, nf))
        ts[tag] = np.real(tflist[:, 0])
        flist = np.imag(tflist[0, :])
        freqs[tag] = flist[0]
        print freqs[tag]

        #ubl file
        ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
        ubls[tag] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))

        vis_data[tag] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))


    t_step = .05
    t_start = 16.5
    nt = 170
    f_step = 1.
    f_start = 126.
    nf = 50
    print ubls['q0C_10_abscal'][np.argsort(la.norm(ubls['q0C_10_abscal'], axis=-1))]
    for u, big_ubl in enumerate(np.array([[0,0,0], [0., 3., 0.], [9., 9., 0.], [0., 21., 0.]])):
        big_data = np.zeros((nt, nf), dtype='complex64') + np.nan
        for tag in sorted(vis_tags):
            t_is = np.round(((ts[tag]-1)%24+1 - t_start) / t_step).astype(int)
            f_i = np.round((freqs[tag] - f_start) / f_step).astype(int)
            if u == 0:
                Q = tag.split('_')[0]
                big_data[t_is, f_i] = {'q0AL':30, 'q0C':31, 'q1AL':29, 'q2AL':26, 'q2C':30, 'q3AL':27, 'q4AL':28, }[Q]
            else:
                if la.norm(ubls[tag][np.argmin(la.norm(ubls[tag] - big_ubl, axis=-1))] - big_ubl) < 1.:
                    big_data[t_is, f_i] = vis_data[tag][:, np.argmin(la.norm(ubls[tag] - big_ubl, axis=-1))]
        plt.subplot(1, 4, u + 1)
        if u==0:
            plt.imshow(np.real(big_data[::-1]), interpolation='none', extent=[f_start, f_start + nf*f_step, t_start, t_start + nt*t_step], aspect='auto')
            plt.ylabel('Local Sidereal Time (Hour)')
            plt.title("Observing Schedule")
        else:
            plt.imshow(np.real(big_data[::-1]), vmin=-10000, vmax=10000, interpolation='none', extent=[f_start, f_start + nf*f_step, t_start, t_start + nt*t_step], aspect='auto')
            plt.title("(%im S, %im E)"%(big_ubl[0], big_ubl[1]))
        plt.xlabel('Frequency (MHz)')





    plt.show()