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

def pinv_sym(M, rcond = 1.e-15, verbose = True):
    eigvl,eigvc = la.eigh(M)
    max_eigv = max(eigvl)
    if verbose and min(eigvl) < 0 and np.abs(min(eigvl)) > max_eigv * rcond:
        print "!WARNING!: negative eigenvalue %.2e is smaller than the added identity %.2e! min rcond %.2e needed."%(min(eigvl), max_eigv * rcond, np.abs(min(eigvl))/max_eigv)
    eigvli = 1 / (max_eigv * rcond + eigvl)
    #for i in range(len(eigvli)):
        #if eigvl[i] < max_eigv * rcond:
            #eigvli[i] = 0
        #else:
            #eigvli[i] = 1/eigvl[i]
    return (eigvc*eigvli).dot(eigvc.transpose())

# tag = "q3_abscalibrated"
# datatag = '_seccasa_polcor.rad'
# vartag = '_seccasa_polcor'
# datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
# nt = 440
# nf = 1
# nUBL = 75

vis_Qs = ["q0AL_*_abscal", "q0C_*_abscal", "q1AL_*_abscal", "q2AL_*_abscal", "q2C_*_abscal", "q3AL_*_abscal", "q4AL_*_abscal"]  # L stands for lenient in flagging
datatag = '_2016_01_20_avg_unpol'
vartag = '_2016_01_20_avg_unpol'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
vis_tags = []
for vis_Q in vis_Qs:
    filenames = glob.glob(datadir + vis_Q + '_xx*' + datatag)
    vis_tags = vis_tags + [os.path.basename(fn).split('_xx')[0] for fn in filenames]


for tag in vis_tags:
# tag = "q2C_abscal"  # L stands for lenient in flagging
    print tag
    # nt = {"q3A_abscal": 253, "q3AL_abscal": 368}[tag]
    nf = 1


    nside = 64
    bnside = 16
    lat_degree = 45.2977

    force_recompute = False

    C = 299.792458
    kB = 1.3806488* 1.e-23

    #deal with beam: create a dictionary for 'x' and 'y' each with a callable function of the form y(freq) in MHz
    # local_beam = {}
    # for p in ['x', 'y']:
    #     freqs = range(150,170,10)
    #     beam_array = np.zeros((len(freqs), 12*bnside**2))
    #     for f in range(len(freqs)):
    #         beam_array[f] = np.fromfile('/home/omniscope/simulate_visibilities/data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s%s.bin'%(bnside, freqs[f], p, p), dtype='float32')
    #     local_beam[p] = si.interp1d(freqs, beam_array, axis=0)

    local_beam = {}
    freqs = range(110, 200, 10)
    for p in ['x', 'y']:
        local_beam[p] = si.interp1d(freqs, np.linalg.norm(np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin'%(bnside, p), dtype='complex64').reshape((len(freqs), 12 * bnside ** 2, 2)), axis=-1)**2, axis=0)

    A = {}
    for p in ['x', 'y']:
        pol = p+p
        data_filename = glob.glob(datadir + tag + '_%s%s_*_*'%(p, p) + datatag)[0]
        nt_nUBL = os.path.basename(data_filename).split(datatag)[0].split('%s%s_'%(p, p))[-1]
        nt = int(nt_nUBL.split('_')[0])
        nUBL = int(nt_nUBL.split('_')[1])

        #tf file
        tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
        tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt, nf))
        tlist = np.real(tflist[:, 0])
        flist = np.imag(tflist[0, :])
        freq = flist[0]
        print freq

        #ubl file
        ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
        ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
        print "%i UBLs to include"%len(ubls)


        #compute A matrix
        A_filename = datadir + tag + '_%s%s_%i_%i.Agsm'%(p, p, len(tlist)*len(ubls), 12*nside**2)

        if os.path.isfile(A_filename) and not force_recompute:
            print A_filename + " already exists."

        else:
            #beam
            beam_healpix = local_beam[p](freq)
            #hpv.mollview(beam_healpix, title='beam %s'%p)
            #plt.show()

            vs = sv.Visibility_Simulator()
            vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
            beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))
            print "Computing A matrix for %s pol..."%p
            sys.stdout.flush()
            timer = time.time()
            A[p] = np.empty((len(tlist)*len(ubls), 12*nside**2), dtype='complex64')
            rotator = hpr.Rotator(coord=['G', 'C'])
            for i in range(12*nside**2):
                dec, ra = rotator(hpf.pix2ang(nside, i, nest=True))#gives theta phi
                dec = np.pi/2 - dec
                print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(12.*nside**2), (12.*nside**2-i)/(i+1)*(float(time.time()-timer)/60.)),
                sys.stdout.flush()

                A[p][:, i] = vs.calculate_pointsource_visibility(ra, dec, ubls, freq, beam_heal_equ = beam_heal_equ, tlist = tlist).flatten()

            print "%f minutes used"%(float(time.time()-timer)/60.)
            sys.stdout.flush()
            A[p].tofile(A_filename)
