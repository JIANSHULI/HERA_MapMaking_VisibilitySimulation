import glob, sys
import numpy as np
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as interp
import simulate_visibilities.simulate_visibilities as sv
import time
import matplotlib.pyplot as plt
nside_standard = 512



#out put Ephi, Ealpha in healpix ring convention

heal_thetas, heal_phis = hpf.pix2ang(nside_standard, range(12*nside_standard**2))

pos_alpha_mask = heal_thetas < np.pi/2

unique_heal_thetas = np.unique(heal_thetas[pos_alpha_mask])
#The 1st column is the azimuthal angle "phi", the second column is the elevation angle "alpha" (theta=pi/2-alpha, if you think of theta as the angle from the z-axis), and the only other relevant column is the 7th one - that's the linear beam strength (the 8th column is a 10log10 of the 7th, but it's not nearly as intuitive in representation).
#Then the 3-4th columns are the real and imaginary components of the electric field in the phi direction "E_phi" and the 5-6th columns are the real and imaginary components of the electric field in the alpha direction "E_alpha"

# print np.sum(pos_alpha_mask), len(pos_alpha_mask)
# plt.plot(pos_alpha_mask); plt.show()

##original script
for pol in ['x', 'y']:
    result = np.zeros((9, 12*nside_standard**2, 2), dtype='complex64')
    d = np.concatenate([np.loadtxt(f) for f in sorted(glob.glob('/home/omniscope/data/mwa_beam/mwa_*-*%s*'%pol))])
    d.shape = (9, 180, 45, 8)
    d = np.transpose(d, (1,2,0,3))
    phi = d[:,0,0,0]/180.*np.pi
    alpha = d[0,:,0,1]/180.*np.pi

    ##dbg
    # f = 5
    # col = 2
    # inter_f = interp.interp2d(alpha, phi, d[..., f, col], fill_value=0)
    # dbg_result = np.zeros_like(result)
    # for heal_theta in unique_heal_thetas:
    #     theta_mask = heal_thetas == heal_theta
    #
    #     qaz_phis = (heal_phis[theta_mask] + np.pi/2)%(np.pi*2)
    #     qaz = np.zeros_like(heal_phis[theta_mask])
    #     qaz[np.argsort(qaz_phis)] = inter_f(np.pi / 2 - heal_theta, np.sort(qaz_phis)).flatten()
    #     # print (heal_phis[theta_mask] + np.pi/2)%(np.pi*2), qaz
    #     # for healpix_n in np.arange(12*nside_standard**2)[theta_mask]:
    #     #     print healpix_n, (heal_phis[healpix_n] + np.pi/2)%(np.pi*2), inter_f(np.pi / 2 - heal_thetas[healpix_n], (heal_phis[healpix_n] + np.pi/2)%(np.pi*2))[0]
    #
    #     dbg_result[0, theta_mask, col/2-1] = dbg_result[0, theta_mask, col/2-1] + qaz * np.exp(1.j*(col%2)*np.pi/2)
    #
    #
    # # for healpix_n in np.arange(12*nside_standard**2)[pos_alpha_mask]:
    # #     dbg_result[1, healpix_n, col/2-1] = dbg_result[1, healpix_n, col/2-1] + inter_f(np.pi / 2 - heal_thetas[healpix_n], (heal_phis[healpix_n] + np.pi/2)%(np.pi*2))[0] * np.exp(1.j*(col%2)*np.pi/2)
    #
    # # plt.plot(dbg_result[0,:,col/2-1])
    # # plt.plot(dbg_result[1,:,col/2-1])
    # # plt.show()
    #
    # sys.exit(0)
    ##dbg ends



    for f in range(9):
        for col in [2,3,4,5]:
            inter_f = interp.interp2d(alpha, phi, d[..., f, col], fill_value=0)
            #    loop over heal_thetas
            for heal_theta in unique_heal_thetas:
                theta_mask = heal_thetas == heal_theta

                #doing some complicated juggling bc interp function automatically sort the list input and output according to that implicitly re-arranged inuput list
                qaz_phis = (heal_phis[theta_mask] + np.pi/2) % (np.pi*2)
                qaz = np.zeros_like(heal_phis[theta_mask])
                qaz[np.argsort(qaz_phis)] = inter_f(np.pi / 2 - heal_theta, np.sort(qaz_phis)).flatten()
                result[f, theta_mask, col/2-1] = result[f, theta_mask, col/2-1] + qaz * np.exp(1.j*(col%2)*np.pi/2)


            # for healpix_n in np.arange(12*nside_standard**2)[pos_alpha_mask]:
            #     result[f, healpix_n, col/2-1] = result[f, healpix_n, col/2-1] + inter_f(np.pi / 2 - heal_thetas[healpix_n], (heal_phis[healpix_n] + np.pi/2)%(np.pi*2))[0] * np.exp(1.j*(col%2)*np.pi/2)
            print '.',
            sys.stdout.flush()
    result.tofile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin'%(nside_standard,pol))
    print pol, "done"
    sys.stdout.flush()

#start testing
bnside = nside_standard#bnside = 32
freqs = range(110, 200, 10)

raw_beam_data = np.concatenate([np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin' % (bnside, p), dtype='complex64').reshape(
    (len(freqs), 12 * bnside ** 2, 2)) for p in ['x', 'y']], axis=-1).transpose(0, 2, 1) #freq by 4 by pix
sv.plot_jones(raw_beam_data[5])


vstest = sv.Visibility_Simulator()
vstest.initial_zenith = np.array([0, 0])
tm = time.time()
beam_heal_equ = np.array(
            [sv.rotate_healpixmap(beam_healpixi, 0, np.pi / 2 - vstest.initial_zenith[1], vstest.initial_zenith[0]) for
             beam_healpixi in raw_beam_data[5]])
print (time.time()-tm) / 60.
sv.plot_jones(beam_heal_equ)
