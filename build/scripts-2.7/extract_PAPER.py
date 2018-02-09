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
import os
import aipy as ap
PI = np.pi
TPI = 2*np.pi

t_avg = 4
dataoptag = '_lstbineven_avg%i'%t_avg
varoptag = '_lstbineven_avg%i'%t_avg
datadir = '/home/omniscope/data/PAPER/lstbin_fg/even/'
calfile = 'psa6622_v002'
overwrite = True


beam_freqs = np.arange(120., 185., 5.)

print "Reading calfile %s..."%calfile,
sys.stdout.flush()

######cal file loaded
aa = ap.cal.get_aa(calfile, beam_freqs/1000.)
print "Done. Antenna layout:"
print aa.ant_layout
sys.stdout.flush()

###redundant claibrator
rc = omni.RedundantCalibrator_PAPER(aa)
badUBLpair = []
for r in range(aa.ant_layout.shape[0]):
    for c in range(2, aa.ant_layout.shape[1]):
        if r != 0:
            badUBLpair.append([aa.ant_layout[0, 0], aa.ant_layout[r, c]])
        badUBLpair.append([aa.ant_layout[-1, 0], aa.ant_layout[r, c]])

badAntenna_x = [8, 14, 15, 16, 26, 27, 28, 34, 37, 38, 42, 46, 50, 53, 72, 74, 82, 84, 85, 110]
badAntenna_y = [3, 7, 15, 16, 23, 26, 27, 34, 38, 46, 50, 56, 57, 72, 100]
badAntenna = sorted(np.unique(badAntenna_x + badAntenna_y).tolist())
rc.compute_redundantinfo(badAntenna=badAntenna, badUBLpair=badUBLpair, antennaLocationTolerance=.3)

##start reading data
rawdata = {}
rawflag = {}

for p, pol in enumerate(['xx', 'yy']):
    data_list = glob.glob(datadir+'*.%s.uv'%pol)
    print len(data_list)
    rawdata[pol], t, timing, raw_lst, rawflag[pol] = omni.importuvs(data_list, {pol: ap.miriad.str2pol[pol]}, totalVisibilityId=rc.Info.totalVisibilityId[rc.Info.subsetbl[rc.Info.crossindex]], timingTolerance=0.001)
    #reorder and sort lst
    rawdata[pol] = rawdata[pol][:, np.argsort(raw_lst)]
    rawflag[pol] = rawflag[pol][:, np.argsort(raw_lst)]
    raw_lst = sorted(raw_lst)
    plt.plot(raw_lst)
plt.show()


nf = rawdata['xx'].shape[2]
nbl = rawdata['xx'].shape[3]
uv = ap.miriad.UV(data_list[0])
freqs = (uv['freq'] + np.arange(nf) * uv['sdf']) * 1.e3

t_chunks = len(raw_lst) / t_avg
lst = np.mean(np.array(raw_lst)[:t_chunks * t_avg].reshape((t_chunks, t_avg)), axis=-1)

ubl_avg_flag = rawflag['xx'][0].any(axis=-1) | rawflag['yy'][0].any(axis=-1)#t by f
ubl_avg_flag = ubl_avg_flag[:t_avg * t_chunks].reshape((t_chunks, t_avg, nf)).any(axis=1)

ubl_avg_data = np.zeros((2, t_chunks, nf, rc.Info.nUBL), dtype='complex64')#p by t by f by ubl
ubl_avg_var = np.zeros((2, t_chunks, nf, rc.Info.nUBL), dtype='float32')#p by t by f by ubl
for p, pol in enumerate(['xx', 'yy']):
    tavg_data = np.mean(rawdata[pol][0, :t_chunks * t_avg].reshape((t_chunks, t_avg, nf, nbl)), axis=1)

    for u in range(rc.Info.nUBL):
        ubl_subindices = rc.Info.ublindex[u][:, 2].astype(int)
        real_part = np.real(tavg_data[..., ubl_subindices])
        imag_part = np.imag(tavg_data[..., ubl_subindices]) * rc.Info.reversed[ubl_subindices]
        ubl_avg_data[p, ..., u] = np.mean(real_part, axis=-1) + 1.j * np.mean(imag_part, axis=-1)
        ubl_avg_var[p, ..., u] = (np.var(real_part, axis=-1) + np.var(imag_part, axis=-1)) / (rc.Info.ublcount[u] - 1)


    for pick_f in range(nf):
        tag = 'psa128_epoch2_%i'%pick_f
        tflag = ubl_avg_flag[:, pick_f]
        nt = len(tflag) - np.sum(tflag)
        if nt > 0:
            (np.array(lst)[~tflag] + 1.j * freqs[pick_f]).astype('complex64').tofile(datadir + tag + '_%s_%i_%i.tf'%(pol, nt, 1))
            ubl_avg_data[p, ~tflag, pick_f].astype('complex64').tofile(datadir + tag + '_%s_%i_%i'%(pol, nt, rc.Info.nUBL) + dataoptag)
            ubl_avg_var[p, ~tflag, pick_f].astype('float32').tofile(datadir + tag + '_%s_%i_%i'%(pol, nt, rc.Info.nUBL) + varoptag + '.var')
            (rc.Info.ubl * [-1, 1, 1]).astype('float32').tofile(datadir + tag + '_%s_%i_%i.ubl'%(pol, rc.Info.nUBL, 3))#there's a N/S sign flip between RedundantCalibrator_PAPER and our convention

for p, pol in enumerate(['xx', 'yy']):
    f = 70
    p = 0
    fun = np.abs
    for u in range(rc.Info.nUBL):
        plt.subplot(4, 5, u + 1)
        plt.plot(lst[~ubl_avg_flag[:, f]], fun(ubl_avg_data[p, :, f, u])[~ubl_avg_flag[:, f]], 'bo')
        plt.plot(lst[~ubl_avg_flag[:, f]], ubl_avg_var[p, :, f, u][~ubl_avg_flag[:, f]]**.5)
    plt.show()

sys.exit(0)

#####get ephem observer
sa = ephem.Observer()
sa.lon = aa.lon
sa.lat = aa.lat
sa.pressure = 0
# sa.lat = -30.72153 / 180 * PI
# sa.lon = 21.42831 / 180 * PI

######load beam
bnside = 64
beam_freqs = np.arange(110., 195., 5.)
print "Reading calfile %s..."%calfile,
sys.stdout.flush()

######cal file loaded
aa = ap.cal.get_aa(calfile, beam_freqs/1000.)
print "Done. Antenna layout:"
print aa.ant_layout
sys.stdout.flush()


beam_healpix = np.zeros((len(beam_freqs), 2, 12*bnside**2), dtype='float32')

healpixvecs = np.array(hpf.pix2vec(bnside, range(12*bnside**2)))
paper_healpixvecs = (healpixvecs[:, healpixvecs[2]>=0]).transpose().dot(sv.rotatez_matrix(-np.pi/2).transpose())#in paper's bm_response convention, (x,y) = (0,1) points north.
for p, pol in enumerate(['x', 'y']):
    for i, paper_angle in enumerate(paper_healpixvecs):
        beam_healpix[:, p, i] = (aa[0].bm_response(paper_angle, pol)**2.).flatten()

local_beam_unpol = si.interp1d(beam_freqs, beam_healpix, axis=0)

freq = 110.
for p in range(2):
    plt.subplot(2, 1, p+1)
    plt.plot(hpf.get_interp_val(local_beam_unpol(freq)[p], np.arange(0, PI/2, .01), 0))
    plt.plot(hpf.get_interp_val(local_beam_unpol(freq)[p], np.arange(0, PI/2, .01), PI/2))
plt.show()

