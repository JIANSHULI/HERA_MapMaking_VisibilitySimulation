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
PI = np.pi
TPI = 2*np.pi

##############This script takes the IQUV calibrated results from absolute_calibrate.py, which are very likely to be systematic -dominated due to strange properties as seen in plot_cygcas.py
######################we re-calibrated the xx and yy using unpolarized model

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

datatag = '_2016_01_20_avg'
dataoptag = '_2016_01_20_avg_unpol'
vartag = '_2016_01_20_avg'
varoptag = '_2016_01_20_avg_unpol'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

overwrite = True

for Q in ['q0AL', 'q0C', 'q1AL', 'q2AL', 'q2C', 'q3AL', 'q4AL']:
    vis_Q = Q + '_*_abscal'
    filenames = glob.glob(datadir + vis_Q + '_xx*' + datatag)
    if len(filenames) > 16:
        raise IOError('Too many files detected')
    vis_tags = [os.path.basename(fn).split('_xx')[0] for fn in filenames]

    #############################
    ###get PS model
    ##############################
    sa = ephem.Observer()
    sa.pressure = 0
    sa.lat = 45.297728 / 180 * PI
    sa.lon = -69.987182 / 180 * PI

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

    #beam
    bnside = 16
    local_beam = si.interp1d(range(110,200,10), np.concatenate([np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin'%(bnside,p), dtype='complex64').reshape((9,12*bnside**2,2)) for p in ['x', 'y']], axis=-1).transpose(0,2,1), axis=0)


    flux_func = {}
    flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
    flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])


    cyg_cas_iquv = np.zeros((np.max([int(tag.split('_')[1]) for tag in vis_tags]) + 1, 2, 4))
    cyg_cas_iquv_std = np.zeros((np.max([int(tag.split('_')[1]) for tag in vis_tags]) + 1, 2, 4))
    freqs = np.zeros(np.max([int(tag.split('_')[1]) for tag in vis_tags]) + 1)
    for tag_i, tag in enumerate(vis_tags):
        print tag
        fi = int(tag.split('_')[1])
        sys.stdout.flush()
        nf = 1



        C = .299792458
        kB = 1.3806488 * 1.e-23



        vis_data = {}
        Ni = {}
        ubls = {}
        ubl_sort = {}
        #read in data and var
        for p in ['x', 'y']:
            pol = p+p
            data_filename = glob.glob(datadir + tag + '_%s%s_*_*'%(p, p) + datatag)[0]
            print "*************************"
            print data_filename
            print "*************************"
            nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('%s%s_'%(p, p))[-1]
            nt = int(nt_nUBL.split('_')[0])
            nUBL = int(nt_nUBL.split('_')[1])

            #tf file
            tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
            tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
            tlist = np.real(tflist[:, 0])
            flist = np.imag(tflist[0, :])    #will be later assuming flist only has 1 element
            vis_freq = flist[0]
            print vis_freq


            #ubl file
            ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
            ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))


            cal_lst_range = np.array([5, 6]) / TPI * 24.
            calibrate_ubl_length = 2600 / vis_freq #18.
            cal_time_mask = (tlist>cal_lst_range[0]) & (tlist<cal_lst_range[1])#a True/False mask on all good data to get good data in cal time range
            cal_ubl_mask = np.linalg.norm(ubls[p], axis=1) >= calibrate_ubl_length


            #get Ni (1/variance) and data
            var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, vartag)
            Ni[p] = 1./np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[np.ix_(cal_time_mask, cal_ubl_mask)].transpose()

            vis_data[p] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
            ubls[p] = ubls[p][cal_ubl_mask]
            ubl_sort[p] = np.argsort(la.norm(ubls[p], axis=1))

            print "%i UBLs to include"%len(ubls[p])

        #beam
        beam_healpix={}
        beam_healpix['x'] = abs(local_beam(vis_freq)[0])**2 + abs(local_beam(vis_freq)[1])**2
        beam_healpix['y'] = abs(local_beam(vis_freq)[2])**2 + abs(local_beam(vis_freq)[3])**2

        ###simulate UNpolarized##############
        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, sa.lat])  # self.zenithequ
        beam_heal_equ_x = sv.rotate_healpixmap(beam_healpix['x'], 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])
        beam_heal_equ_y = sv.rotate_healpixmap(beam_healpix['y'], 0, np.pi / 2 - vs.initial_zenith[1], vs.initial_zenith[0])


        print "Computing UNpolarized point sources matrix..."
        sys.stdout.flush()
        cal_sources = ['cyg', 'cas']
        Apol = np.empty((np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask), len(cal_sources)), dtype='complex64')
        timer = time.time()
        for n, source in enumerate(['cyg', 'cas']):
            ra = southern_points[source]['body']._ra
            dec = southern_points[source]['body']._dec

            Apol[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq, beam_heal_equ=beam_heal_equ_x, tlist=tlist[cal_time_mask])
            Apol[:, 1, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq, beam_heal_equ=beam_heal_equ_y, tlist=tlist[cal_time_mask])

        Apol = np.conjugate(Apol).reshape((np.sum(cal_ubl_mask), 2 * np.sum(cal_time_mask), len(cal_sources)))
        Ni = np.transpose([Ni['x'], Ni['y']], (1, 0, 2))

        realA = np.zeros((2 * Apol.shape[0] * Apol.shape[1], Apol.shape[2] + 2 * np.sum(cal_ubl_mask) * 2), dtype='float32')
        for coli, ncol in enumerate(range(Apol.shape[2], realA.shape[1])):
            realA[coli * np.sum(cal_time_mask): (coli + 1) * np.sum(cal_time_mask), ncol] = 1
        realA[:, :Apol.shape[2]] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2])))), axis=0)#consider only include non-V? doesnt seem to change answer much

        realNi = np.concatenate((Ni.flatten() * 2, Ni.flatten() * 2))
        realAtNiAinv = np.linalg.pinv(np.einsum('ji,j,jk->ik', realA, realNi, realA))


        b = np.transpose([vis_data['x'], vis_data['y']], (1, 0, 2))
        phase_degen_niter = 0
        phase_degen2 = {'x': np.zeros(2), 'y': np.zeros(2)}
        phase_degen_iterative_x = np.zeros(2)
        phase_degen_iterative_y = np.zeros(2)
        def tocomplex(realdata):
            reshapedata = realdata.reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
            return reshapedata[0] + reshapedata[1] * 1.j


        # while (phase_degen_niter < 50 and np.linalg.norm(phase_degen_iterative) > 1e-5) or phase_degen_niter == 0:
        #     phase_degen_niter += 1
        #     b = b * np.exp(1.j * ubls['x'][:, :2].dot(phase_degen_iterative))[:, None, None]
        #     realb = np.concatenate((np.real(b.flatten()), np.imag(b.flatten())))
        #
        #     psol = realAtNiAinv.dot(np.transpose(realA).dot(realNi * realb))
        #     realb_fit = realA.dot(psol)
        #     perror = ((realb_fit - realb) * (realNi**.5)).reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
        #
        #     bfit = tocomplex(realb_fit)
        #     phase_degen_iterative = solve_phase_degen(np.transpose(b[:, 0]), np.transpose(b[:, -1]), np.transpose(bfit[:, 0]), np.transpose(bfit[:, -1]), ubls['x'])
        #     phase_degen2 += phase_degen_iterative
        #     print phase_degen_niter, phase_degen2, np.linalg.norm(perror)
        phase_degen_niter_max = 20
        while (phase_degen_niter < phase_degen_niter_max and max(np.linalg.norm(phase_degen_iterative_x), np.linalg.norm(phase_degen_iterative_y)) > 1e-5) or phase_degen_niter == 0:
            phase_degen_niter += 1
            b[:, 0] = b[:, 0] * np.exp(1.j * ubls['x'][:, :2].dot(phase_degen_iterative_x))[:, None]
            b[:, -1] = b[:, -1] * np.exp(1.j * ubls['y'][:, :2].dot(phase_degen_iterative_y))[:, None]
            realb = np.concatenate((np.real(b.flatten()), np.imag(b.flatten())))

            psol = realAtNiAinv.dot(np.transpose(realA).dot(realNi * realb))
            realb_fit = realA.dot(psol)
            perror = ((realb_fit - realb) * (realNi**.5)).reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))

            realbfit_noadditive = np.sum(realA[:, :Apol.shape[2]] * psol[:Apol.shape[2]], axis=-1)
            realbfit_additive = realb_fit - realbfit_noadditive
            realb_noadditive = realb - realbfit_additive
            bfit_noadditive = tocomplex(realbfit_noadditive)
            b_noadditive = tocomplex(realb_noadditive)
            if phase_degen_niter == phase_degen_niter_max:
                phase_degen_iterative_x = omni.solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'], [3, 3, 1e3])
                phase_degen_iterative_y = omni.solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'], [3, 3, 1e3])

            else:
                phase_degen_iterative_x = solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'])
                phase_degen_iterative_y = solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'])
            phase_degen2['x'] += phase_degen_iterative_x
            phase_degen2['y'] += phase_degen_iterative_y
            print phase_degen_niter, phase_degen2['x'], phase_degen2['y'], np.linalg.norm(perror)

        renorm = flux_func['cyg'](vis_freq) / (2 * psol[0])
        cyg, cas = 2 * psol[:2] * renorm

        print vis_freq, renorm, phase_degen2['x'], phase_degen2['y'], flux_func['cas'](vis_freq) / cas

        cyg_cas_iquv[fi, :, 0] = [cyg, cas]
        cyg_cas_iquv_std[fi, :, 0] = np.diagonal(realAtNiAinv[:2, :2])**.5 * renorm
        freqs[fi] = vis_freq

        #re-read in data and var and output unpolarized version
        for p in ['x', 'y']:
            pol = p+p
            data_filename = glob.glob(datadir + tag + '_%s%s_*_*'%(p, p) + datatag)[0]
            nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('%s%s_'%(p, p))[-1]
            nt = int(nt_nUBL.split('_')[0])
            nUBL = int(nt_nUBL.split('_')[1])
            var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, vartag)

            op_data_filename = datadir + tag + '_%s%s_%i_%i%s'%(p, p, nt, nUBL, dataoptag)
            op_var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, varoptag)

            #ubl file
            ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
            ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))

            #get data and var and apply change
            new_data = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL)) * np.exp(1.j * ubls[:, :2].dot(phase_degen2[p])) * renorm
            new_var = np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL)) * renorm**2

            #save
            if os.path.isfile(op_data_filename) and not overwrite:
                raise IOError(op_data_filename + ' exists.')
            else:
                new_data.astype('complex64').tofile(op_data_filename)
            if os.path.isfile(op_var_filename) and not overwrite:
                raise IOError(op_var_filename + ' exists.')
            else:
                new_var.astype('float32').tofile(op_var_filename)

    np.savez(datadir + 'cygcas_' + Q + dataoptag + varoptag, cyg_cas_iquv=cyg_cas_iquv, cyg_cas_iquv_std=cyg_cas_iquv_std, freqs=freqs)
