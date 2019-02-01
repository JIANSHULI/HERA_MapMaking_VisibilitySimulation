%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import uvtools, pyuvdata, glob, aipy, hera_qm
from scipy.interpolate import RectBivariateSpline
from HERA_MapMaking_VisibilitySimulation import DATA_PATH
from HERA_MapMaking_VisibilitySimulation import UVData as UVData_HR


files = glob.glob(DATA_PATH + '/ObservingSession1232039492/2458504/*.HH.uvh5')
data, times = {}, {}
print ('Reading', files[0], 'to', files[-1])

for f in files:
    uvf = pyuvdata.UVData()
    uvf.read_uvh5(f)
    keys = uvf.get_antpairpols()
    for k in keys:
        data[k] = data.get(k, []) + [uvf.get_data(k)]
        lsts = uvf.lst_array[np.append(*uvf._key2inds(k)[:2])]
        times[k] = times.get(k, []) + [lsts]
    data = {k: np.concatenate(dk, axis=0) for k,dk in data.items()}
    times = {k: np.concatenate(tk, axis=0) for k, tk in times.items()}
    freqs = uvf.freq_array
    
for k, d in data.items():
    if not np.all(d == 0) and k[-1] == 'xx' and k[0] != k[1]:
        print (k, d.shape)
        

POL = 'xx'
ks = [(0, 1, POL), (0, 12, POL), (1, 12, POL), (13, 0, POL), (13, 1, POL), (13, 12, POL), (13, 25, POL), (13,26,POL), (25, 0, POL), (25, 1, POL), (25, 12, POL), (25, 26, POL), (26, 0, POL), (26, 1, POL), (26, 12, POL)]

plt.figure(figsize=(21,28))
for cnt,k in enumerate(ks):
    plt.subplot(15, 1, cnt + 1)
    plt.title(str(k) + ' uncalibrated')
    uvtools.plot.waterfall(data[k], mode='phs',
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9, times[k][-1], times[k][0]))
    plt.ylabel('Local Sideral Time [radians]')
     
plt.subplots_adjust(top=0.95, bottom=0.07, hspace=.3)
plt.xlabel('Frequency [GHz]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-{0}'.format((POL)), bbox_inches='tight')
plt.show()


files = sorted(glob.glob(DATA_PATH + '/IDR2_1/LSTBIN/one_group/grp1/zen.*.*.{0}.LST.*.*.uvOCRSL'.format(POL)))[:-1]
ks_h1c = [(0, 1, POL), (0, 12, POL), (1, 12, POL), (0, 13, POL), (1, 13, POL), (13, 12, POL), (13, 25, POL), (13,26,POL), (0, 25, POL), (1, 25, POL), (12, 25, POL), (25, 26, POL), (0, 26, POL), (1, 26, POL), (12, 26, POL)] #  [(0, 1, POL), (0, 12, POL), (1, 12, POL), (13, 0, POL), (13, 1, POL), (13, 12, POL), (13, 25, POL), (13,26,POL), (25, 0, POL), (25, 1, POL), (25, 12, POL), (25, 26, POL), (26, 0, POL), (26, 1, POL), (26, 12, POL)] # [(0, 1, POL), (0, 12, POL), (1, 12, POL), (0, 13, POL), (1, 13, POL), (13, 12, POL), (13, 25, POL), (13,26,POL), (0, 25, POL), (1, 25, POL), (12, 25, POL), (25, 26, POL), (0, 26, POL), (1, 26, POL), (12, 26, POL)]

data_h1c = {}
times_h1c = {}
for f in files:
    print ('Reading', f)
    # uvf = pyuvdata.UVData()
    try:
        uvf = UVData_HR()
        uvf.read_miriad(f, run_check=False)
        for k in ks_h1c:
            if k[-1] != POL: continue
            if k[0] < k[1]:
                data_h1c[k] = data_h1c.get(k, []) + [uvf.get_data(k)]
            else:
                data_h1c[k] = data_h1c.get((k[1], k[0], k[2]), []) + [uvf.get_data((k[1], k[0], k[2]))]
            lsts = uvf.lst_array[np.append(*uvf._key2inds(k)[:2])]
            times_h1c[k] = times_h1c.get(k, []) + [lsts]
    
        data_h1c = {k: np.concatenate(dk, axis=0) for k, dk in data_h1c.items()}
        times_h1c = {k: np.concatenate(tk, axis=0) for k, tk in times_h1c.items()}
        freqs_h1c = uvf.freq_array
    except:
        print('file: {0} not loaded'.format(f))
        pass

data_h1c_interp = {}
for k,kh1c in zip(ks,ks_h1c):
    t, f, d = times_h1c[kh1c], freqs_h1c[0], data_h1c[kh1c]
    intr = RectBivariateSpline(t, f, d.real)
    inti = RectBivariateSpline(t, f, d.imag)
    data_h1c_interp[k] = intr(times[k], freqs[0]) + 1j * inti(times[k], freqs[0])

plt.figure(figsize=(21, 28))
for cnt,(k,kh1c) in enumerate(zip(ks, ks_h1c)):
    plt.subplot(15, 2, 2 * cnt + 1)
    uvtools.plot.waterfall(data_h1c[kh1c], mode='phs',
                           extent=(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9,
                                   times_h1c[kh1c][-1], times_h1c[kh1c][0]))
    plt.subplot(15, 2, 2 * cnt + 2)
    uvtools.plot.waterfall(data_h1c_interp[k], mode='phs',
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9,
                                   times[k][-1], times[k][0]))

plt.subplot(15, 2, 1); plt.title('H1C data')
plt.subplot(15, 2, 2); plt.title('resampled H1C data')
plt.subplot(15, 2, 15); plt.ylabel('Local Sideral Time [radians]')
plt.subplot(15, 2, 29); plt.xlabel('Frequency [GHz]')
plt.subplot(15, 2, 30); plt.xlabel('Frequency [GHz]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-IDR21-{0}'.format((POL)), bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 100))
for cnt, k in enumerate(ks):
    ratio = data[k] * data_h1c_interp[k].conj() / np.abs(data_h1c_interp[k]) ** 2
    plt.subplot(2, 15, cnt + 1)
    uvtools.plot.waterfall(ratio, mode='phs',
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9,
                                   times[k][-1], times[k][0]))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.title(str(k) + ' H2C/H1C')
    plt.subplot(2, 15, cnt + 16)
    plt.plot(freqs[0] / 1e9, np.angle(np.median(ratio, axis=0)))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.xlabel('Frequency [GHz]')
    plt.grid()

plt.subplots_adjust(top=0.95, bottom=0.1, hspace=.2, wspace=.3)
plt.subplot(2, 15, 1); plt.ylabel('Local Sidereal Time [radians]')
plt.subplot(2, 15, 16); plt.ylabel('Phase [radians]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-IDR21-ratio-phase-{0}'.format((POL)), bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 100))
for cnt, k in enumerate(ks):
    ratio = data[k] * data_h1c_interp[k].conj() / np.abs(data_h1c_interp[k]) ** 2
    plt.subplot(2, 15, cnt + 1)
    uvtools.plot.waterfall(ratio, mode='lin', mx=2, drng=2,
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9,
                                   times[k][-1], times[k][0]))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.title(str(k) + ' H2C/H1C')
    plt.subplot(2, 15, cnt + 16)
    plt.plot(freqs[0] / 1e9, np.abs(np.median(ratio, axis=0)))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.xlabel('Frequency [GHz]')
    plt.grid()

plt.subplots_adjust(top=0.95, bottom=0.1, hspace=.2, wspace=.3)
plt.subplot(2, 15, 1); plt.ylabel('Gain [ratio]')
plt.subplot(2, 15, 16); plt.ylabel('Gain [ratio]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-IDR21-ratio-abs-{0}'.format((POL)), bbox_inches='tight')
plt.show()


data_cal = {}
data_cal[0,1,POL] = data[0,1,,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[0,12,POL] = data[0,12,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[1,12,POL] = data[1,12,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[13,0,POL] = data[13,0,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[13,1,POL] = data[13,1,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[13,12,POL] = data[13,12,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[13,25,POL] = data[13,25,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[13,26,POL] = data[13,26,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[25,0,POL] = data[25,0,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[25,1,POL] = data[25,1,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[25,12,POL] = data[25,12,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[25,26,POL] = data[25,26,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)
data_cal[26,0,POL] = data[26,0,POL] * 0.6e-3 * np.exp(2j * np.pi* 6.59e-9 * freqs)
data_cal[26,1,POL] = data[26,1,POL] * 0.6e-3 * np.exp(2j * np.pi* -5.15e-9 * freqs)
data_cal[26,12,POL] = data[26,12,POL] * 0.7e-3 * np.exp(2j * np.pi* 11.75e-9 * freqs)



plt.figure(figsize=(12, 20))
for cnt, k in enumerate(ks):
    ratio = data_cal[k] * data_h1c_interp[k].conj() / np.abs(data_h1c_interp[k]) ** 2
    plt.subplot(2, 15, cnt + 1)
    uvtools.plot.waterfall(ratio, mode='phs',
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9,
                                   times[k][-1], times[k][0]))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.title(str(k) + ' H2C/H1C')
    plt.subplot(2, 15, cnt + 16)
    plt.plot(freqs[0] / 1e9, np.angle(np.median(ratio, axis=0)))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.xlabel('Frequency [GHz]')
    plt.grid()

plt.subplots_adjust(top=0.95, bottom=0.1, hspace=.2, wspace=.3)
plt.subplot(2, 15, 1); plt.ylabel('Local Sidereal Time [radians]')
plt.subplot(2, 15, 16); plt.ylabel('Phase [radians]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-IDR21-caled-ratio-phase-{0}'.format((POL)), bbox_inches='tight')
plt.show()



plt.figure(figsize=(12, 20))
for cnt, k in enumerate(ks):
    ratio = data_cal[k] * data_h1c_interp[k].conj() / np.abs(data_h1c_interp[k]) ** 2
    plt.subplot(2, 15, cnt + 1)
    uvtools.plot.waterfall(ratio, mode='lin', mx=2, drng=2,
                           extent=(freqs[0, 0] / 1e9, freqs[0, -1] / 1e9,
                                   times[k][-1], times[k][0]))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.title(str(k) + ' H2C/H1C')
    plt.subplot(2, 15, cnt + 16)
    plt.plot(freqs[0] / 1e9, np.abs(np.median(ratio, axis=0)))
    plt.xlim(freqs_h1c[0, 0] / 1e9, freqs_h1c[0, -1] / 1e9)
    plt.xlabel('Frequency [GHz]')
    plt.grid()

plt.subplots_adjust(top=0.95, bottom=0.1, hspace=.2, wspace=.3)
plt.subplot(2, 15, 1); plt.ylabel('Gain [ratio]')
plt.subplot(2, 15, 16); plt.ylabel('Gain [ratio]')
plt.savefig(DATA_PATH + '/../Output/vivaldi-6-IDR21-caled-ratio-abs-{0}'.format((POL)), bbox_inches='tight')
plt.show()






