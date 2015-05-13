import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, datetime, warnings
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
PI = np.pi
TPI = np.pi * 2

def pixelize(sky, nside_distribution, nside_standard, nside_start, thresh, final_index, thetas, phis, sizes):
    #thetas = []
    #phis = []
    for inest in range(12*nside_start**2):
        pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis, sizes)
        #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
        #thetas += newt.tolist()
        #phis += newp.tolist()
    #return np.array(thetas), np.array(phis)

def pixelize_helper(sky, nside_distribution, nside_standard, nside, inest, thresh, final_index, thetas, phis, sizes):
    #print "visiting ", nside, inest
    starti, endi = inest*nside_standard**2/nside**2, (inest+1)*nside_standard**2/nside**2
    ##local mean###if nside == nside_standard or np.std(sky[starti:endi])/np.mean(sky[starti:endi]) < thresh:
    if nside == nside_standard or np.std(sky[starti:endi]) < thresh:
        nside_distribution[starti:endi] = nside
        final_index[starti:endi] = len(thetas)#range(len(thetas), len(thetas) + endi -starti)
        #return hp.pix2ang(nside, [inest], nest=True)
        newt, newp = hp.pix2ang(nside, [inest], nest=True)
        thetas += newt.tolist()
        phis += newp.tolist()
        sizes += (np.ones_like(newt) * nside_standard**2 / nside**2).tolist()
        #sizes += (np.ones_like(newt) / nside**2).tolist()

    else:
        #thetas = []
        #phis = []
        for jnest in range(inest * 4, (inest + 1) * 4):
            pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh, final_index, thetas, phis, sizes)
            #newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh)
            #thetas += newt.tolist()
            #phis += newp.tolist()
        #return np.array(thetas), np.array(phis)

nside_start = 16
nside_beamweight = 16
nside_standard = 128
bnside = 16
plotcoord = 'C'
thresh = 0.3
#S_scale = 2
#S_thresh = 1000#Kelvin
#S_type = 'gsm%irm%i'%(S_scale,S_thresh)
S_type = 'dySP' #dynamic S polarized [[.25,0,0,.25], [0,p,0,0], [0,0,p,0], [.25,0,0,.25]]
remove_additive = True


lat_degree = 45.2977
C = 299.792458
kB = 1.3806488* 1.e-23
script_dir = os.path.dirname(os.path.realpath(__file__))


plot_pixelization = True
plot_projection = True
plot_data_error = True

force_recompute = False
force_recompute_AtNiAi_eig = False
force_recompute_AtNiAi = False
force_recompute_S = True
force_recompute_SEi = False

####################################################
################data file and load beam##############
####################################################
tag = "q3A_abscal"
datatag = '_2015_05_09'
vartag = '_2015_05_09'
datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
nt = 253
nf = 1
nUBL = 78

#deal with beam: create a callable function of the form y(freq) in MHz and returns npix by 4
freqs = range(110,200,10)
local_beam = si.interp1d(freqs, np.concatenate([np.fromfile('/home/omniscope/data/mwa_beam/healpix_%i_%s.bin'%(bnside,p), dtype='complex64').reshape((len(freqs),12*bnside**2,2)) for p in ['x', 'y']], axis=-1).transpose(0,2,1), axis=0)


A = {}

for p in ['x', 'y']:
    pol = p+p

    #tf file, t in lst hours
    tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p, nt, nf)
    tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
    tlist = np.real(tflist[:, 0])
    flist = np.imag(tflist[0, :])
    freq = flist[0]

    #tf mask file, 0 means flagged bad data
    try:
        tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
        tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
        tmask = np.array(tfmlist[:,0].astype('bool'))
        #print tmask
    except:
        print "No mask file found"
        tmask = np.ones_like(tlist).astype(bool)
    #print freq, tlist

    #ubl file
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
    print "%i UBLs to include, longest baseline is %i wavelengths"%(len(ubls), np.max(np.linalg.norm(ubls, axis = 1)) / (C/freq))

    A_filename = datadir + tag + '_%s%s_%i_%i.A'%(p, p, len(tlist)*len(ubls), 12*nside_beamweight**2)

    if os.path.isfile(A_filename) and not force_recompute:
        print "Reading A matrix from %s"%A_filename
        sys.stdout.flush()
        A[p] = np.fromfile(A_filename, dtype='complex64').reshape((len(ubls), len(tlist), 12*nside_beamweight**2))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), 12*nside_beamweight**2))
    else:
        #beam
        if p == 'x':
            beam_healpix = abs(local_beam(freq)[0])**2 + abs(local_beam(freq)[1])**2
        elif p == 'y':
            beam_healpix = abs(local_beam(freq)[2])**2 + abs(local_beam(freq)[3])**2
        #hpv.mollview(beam_healpix, title='beam %s'%p)
        #plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
        beam_heal_equ = np.array(sv.rotate_healpixmap(beam_healpix, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]))
        print "Computing A matrix for %s pol..."%p
        sys.stdout.flush()
        timer = time.time()
        A[p] = np.empty((len(tlist)*len(ubls), 12*nside_beamweight**2), dtype='complex64')
        for i in range(12*nside_beamweight**2):
            dec, ra = hpf.pix2ang(nside_beamweight, i)#gives theta phi
            dec = np.pi/2 - dec
            print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(12.*nside_beamweight**2), (12.*nside_beamweight**2-i)/(i+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[p][:, i] = np.array([vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_heal_equ = beam_heal_equ, tlist = tlist) for d in ubls]).flatten()

        print "%f minutes used"%(float(time.time()-timer)/60.)
        sys.stdout.flush()
        A[p].tofile(A_filename)
        A[p] = A[p].reshape((len(ubls), len(tlist), 12*nside_beamweight**2))[:,tmask].reshape((len(ubls)*len(tlist[tmask]), 12*nside_beamweight**2))

####################################################
###beam weights using an equal pixel A matrix######
#################################################
print "Computing beam weight...",
sys.stdout.flush()
beam_weight = ((la.norm(A['x'], axis = 0)**2 + la.norm(A['y'], axis = 0)**2)**.5)[hpf.nest2ring(nside_beamweight, range(12*nside_beamweight**2))]
beam_weight = beam_weight/np.mean(beam_weight)
beam_weight = np.array([beam_weight for i in range(nside_standard**2/nside_beamweight**2)]).transpose().flatten()
print "done."
sys.stdout.flush()

################################################
#####################GSM###########################
#############################################
pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
components = np.loadtxt(script_dir + '/../data/components.dat')
scale_loglog = si.interp1d(np.log(components[:,0]), np.log(components[:,1]))
w1 = si.interp1d(components[:,0], components[:,2])
w2 = si.interp1d(components[:,0], components[:,3])
w3 = si.interp1d(components[:,0], components[:,4])
gsm_standard = np.exp(scale_loglog(np.log(freq))) * (w1(freq)*pca1 + w2(freq)*pca2 + w3(freq)*pca3)

#rotate sky map and converts to nest
equatorial_GSM_standard = np.zeros(12*nside_standard**2,'float')
print "Rotating GSM_standard and converts to nest...",
sys.stdout.flush()
equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000,stdtime=2013.58))
ang0, ang1 =hp.rotator.rotateDirection(equ2013_to_gal_matrix, hpf.pix2ang(nside_standard, range(12*nside_standard**2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
print "done."
sys.stdout.flush()



########################################################################
########################processing dynamic pixelization######################
########################################################################

nside_distribution = np.zeros(12*nside_standard**2)
final_index = np.zeros(12*nside_standard**2)
thetas, phis, sizes = [], [], []
abs_thresh = np.mean(equatorial_GSM_standard * beam_weight) * thresh
pixelize(equatorial_GSM_standard * beam_weight, nside_distribution, nside_standard, nside_start, abs_thresh, final_index, thetas, phis, sizes)
npix = len(thetas)
valid_pix_mask = hpf.get_interp_val(beam_weight, thetas, phis, nest=True) > 1e-3*max(beam_weight)
valid_npix = np.sum(valid_pix_mask)
fake_solution = (hpf.get_interp_val(equatorial_GSM_standard, thetas, phis, nest=True)*sizes)[valid_pix_mask]
fake_solution = np.concatenate((fake_solution, np.zeros_like(fake_solution), np.zeros_like(fake_solution), np.zeros_like(fake_solution)))
sizes = np.concatenate((np.array(sizes)[valid_pix_mask], np.array(sizes)[valid_pix_mask], np.array(sizes)[valid_pix_mask], np.array(sizes)[valid_pix_mask]))
def sol2map(solx):

    final_index4 = np.concatenate((final_index, final_index+npix, final_index+npix*2, final_index+npix*3)).astype(int)
    full_sol = np.zeros(4*npix)
    full_sol[np.concatenate((valid_pix_mask,valid_pix_mask,valid_pix_mask,valid_pix_mask))] = solx/sizes
    return full_sol[final_index4]

#final_index_filename = datadir + tag + '_%i.dyind%i_%.3f'%(nside_standard, npix, thresh)
#final_index.astype('float32').tofile(final_index_filename)
#sizes_filename = final_index_filename.replace('dyind', "dysiz")
#np.array(sizes).astype('float32').tofile(sizes_filename)
if plot_pixelization:
    ##################################################################
    ####################################sanity check########################
    ###############################################################
    #npix = 0
    #for i in nside_distribution:
        #npix += i**2/nside_standard**2
    #print npix, len(thetas)

    stds = np.std((equatorial_GSM_standard*beam_weight).reshape(12*nside_standard**2/4,4), axis = 1)

    ##################################################################
    ####################################plotting########################
    ###############################################################
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=RuntimeWarning)
        hpv.mollview(beam_weight, min=0,max=4, coord=plotcoord, title='beam', nest=True)
        hpv.mollview(np.log10(equatorial_GSM_standard), min=0,max=4, coord=plotcoord, title='GSM', nest=True)
        hpv.mollview(np.log10(sol2map(fake_solution)[:len(equatorial_GSM_standard)]), min=0,max=4, coord=plotcoord, title='GSM gridded', nest=True)
        hpv.mollview(np.log10(stds/abs_thresh), min=np.log10(thresh)-3, max = 3, coord=plotcoord, title='std', nest=True)
        hpv.mollview(np.log2(nside_distribution), min=np.log2(nside_start),max=np.log2(nside_standard), coord=plotcoord, title='count %i %.3f'%(len(thetas), float(len(thetas))/(12*nside_standard**2)), nest=True)
    plt.show()


##################################################################
####################compute dynamic A matrix########################
###############################################################
ubls = {}
for p in ['x', 'y']:
    ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p, nUBL, 3)
    ubls[p] = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
ubl_index = {}#stored index in each pol's ubl for the common ubls
for p in ['x', 'y']:
    ubl_index[p] = np.zeros(len(common_ubls), dtype='int')
    for i, u in enumerate(common_ubls):
        if u in ubls[p]:
            ubl_index[p][i] = np.argmin(la.norm(ubls[p] - u, axis=-1)) + 1
        elif -u in ubls[p]:
            ubl_index[p][i] = - np.argmin(la.norm(ubls[p] + u, axis=-1)) - 1
        else:
            raise Exception('Logical Error')
#vs = sv.Visibility_Simulator()
#vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
#beam_heal_equ = np.array([sv.rotate_healpixmap(beam_healpixi, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]) for beam_healpixi in local_beam(freq)])
#print vs.calculate_pol_pointsource_visibility(0, .5, ubls[0], freq, beam_heal_equ = beam_heal_equ, tlist = tlist).shape
#sys.exit(0)

A_filename = datadir + tag + '_%i_%i.AdpcIQUV%i_%.3f'%(len(tlist)*len(common_ubls), valid_npix, nside_standard, thresh)
def get_A():
    if os.path.isfile(A_filename) and not force_recompute:
        print "Reading A matrix from %s"%A_filename
        sys.stdout.flush()
        A = np.fromfile(A_filename, dtype='complex64').reshape((len(common_ubls), 4, len(tlist), 4, valid_npix))
    else:
        #beam
        beam_healpix = local_beam(freq)
        #hpv.mollview(beam_healpix, title='beam %s'%p)
        #plt.show()

        vs = sv.Visibility_Simulator()
        vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
        beam_heal_equ = np.array([sv.rotate_healpixmap(beam_healpixi, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]) for beam_healpixi in local_beam(freq)])
        print "Computing A matrix..."
        sys.stdout.flush()
        A = np.empty((len(common_ubls), 4 * len(tlist), 4, valid_npix), dtype='complex64')
        timer = time.time()
        for n, i in enumerate(np.arange(npix)[valid_pix_mask]):
            ra = phis[i]
            dec = np.pi/2 - thetas[i]
            print "\r%.1f%% completed, %f minutes left"%(100.*float(n)/(valid_npix), float(valid_npix-n)/(n+1)*(float(time.time()-timer)/60.)),
            sys.stdout.flush()

            A[..., n] = vs.calculate_pol_pointsource_visibility(ra, dec, common_ubls, freq, beam_heal_equ = beam_heal_equ, tlist = tlist).dot([[.5,.5,0,0],[0,0,.5,.5j],[0,0,.5,-.5j],[.5,-.5,0,0]])

        print "%f minutes used"%(float(time.time()-timer)/60.)
        sys.stdout.flush()
        A.tofile(A_filename)
        A.shape = (len(common_ubls), 4, len(tlist), 4, valid_npix)
    tmask = np.ones_like(tlist).astype(bool)
    for p in ['x', 'y']:
        #tf mask file, 0 means flagged bad data
        try:
            tfm_filename = datadir + tag + '_%s%s_%i_%i.tfm'%(p, p, nt, nf)
            tfmlist = np.fromfile(tfm_filename, dtype='float32').reshape((nt,nf))
            tmask = tmask&np.array(tfmlist[:,0].astype('bool'))
            #print tmask
        except:
            print "No mask file found"
        #print freq, tlist
    #Merge A
    A.shape = (len(common_ubls)*4*len(tlist[tmask]), 4*valid_npix)
    return np.concatenate((np.real(A), np.imag(A)))
A = get_A()
##Compute autocorr
#beam_healpix = local_beam(freq)
#vs = sv.Visibility_Simulator()
#vs.initial_zenith = np.array([0, lat_degree*np.pi/180])#self.zenithequ
#beam_heal_equ = np.array([sv.rotate_healpixmap(beam_healpixi, 0, np.pi/2 - vs.initial_zenith[1], vs.initial_zenith[0]) for beam_healpixi in local_beam(freq)])
#print "Computing autocorr..."
#sys.stdout.flush()
#timer = time.time()
#autocorr = np.empty((4 * len(tlist), 4, npix), dtype='complex64')
#for i in range(npix):
    #ra = phis[i]
    #dec = np.pi/2 - thetas[i]
    #print "\r%.1f%% completed, %f minutes left"%(100.*float(i)/(npix), float(npix-i)/(i+1)*(float(time.time()-timer)/60.)),
    #sys.stdout.flush()

    #autocorr[..., i] = vs.calculate_pol_pointsource_visibility(ra, dec, [[0,0,0]], freq, beam_heal_equ = beam_heal_equ, tlist = tlist)[0]

#print "%f minutes used"%(float(time.time()-timer)/60.)
#sys.stdout.flush()



data = {}
Ni = {}
data_shape = {}
ubl_sort = {}
for p in ['x', 'y']:
    for p2 in ['x', 'y']:
        pol = p+p2
        #tf file
        tf_filename = datadir + tag + '_%s%s_%i_%i.tf'%(p, p2, nt, nf)
        tflist = np.fromfile(tf_filename, dtype='complex64').reshape((nt,nf))
        tlist = np.real(tflist[:, 0])

        #ubl file
        ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl'%(p, p2, nUBL, 3)
        ubls = np.fromfile(ubl_filename, dtype='float32').reshape((nUBL, 3))
        print "%i UBLs to include, longest baseline is %i wavelengths"%(len(common_ubls), np.max(np.linalg.norm(common_ubls, axis = 1)) / (C/freq))


        #get Ni (1/variance) and data
        var_filename = datadir + tag + '_%s%s_%i_%i'%(p, p2, nt, nUBL) + vartag + '.var'
        Ni[pol] = 1./(np.fromfile(var_filename, dtype='float32').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p])-1].flatten() * (1.e-26*(C/freq)**2/2/kB/(4*np.pi/(12*nside_standard**2)))**2)
        data_filename = datadir + tag + '_%s%s_%i_%i'%(p, p2, nt, nUBL) + datatag
        data[pol] = np.fromfile(data_filename, dtype='complex64').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p])-1]
        data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
        data[pol] = (data[pol].flatten()*1.e-26*(C/freq)**2/2/kB/(4*np.pi/(12*nside_standard**2))).conjugate()#there's a conjugate convention difference
        data_shape[pol] = (len(common_ubls), np.sum(tmask))
        ubl_sort[p] = np.argsort(la.norm(common_ubls, axis = 1))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#Merge data
original_data = np.array([data['xx'],data['xy'],data['yx'],data['yy']]).reshape([4]+list(data_shape['xx'])).transpose((1,0,2))
data = np.array([data['xx'],data['xy'],data['yx'],data['yy']]).reshape([4]+list(data_shape['xx'])).transpose((1,0,2)).flatten()
data = np.concatenate((np.real(data), np.imag(data))).astype('float32')
Ni = np.concatenate((Ni['xx'],Ni['xy'],Ni['yx'],Ni['yy'])).reshape([4]+list(data_shape['xx'])).transpose((1,0,2)).flatten()
Ni = np.concatenate((Ni/2, Ni/2))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()



print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()
#simulate visibilities

#clean_sim_data = np.array([Aiter.dot(fake_solution) for Aiter in A])
clean_sim_data = A.dot(fake_solution.astype(A.dtype))


vis_normalization = np.median(np.linalg.norm(data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:,:, [0,3]],axis=0) / np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:,:, [0,3]],axis=0))
print "Normalization from visibilities", vis_normalization
diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
diff_data = diff_data[0] + 1j * diff_data[1]
diff_norm = {}
diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis = 1)
diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis = 1)

if plot_data_error:
        plt.plot(diff_norm['x'][ubl_sort['x']])
        plt.plot(diff_norm['y'][ubl_sort['y']])

#todo use autocorr rather than constant as removal term
#todo remove cross-pol terms
if remove_additive:
    niter = 0
    #additive = {'x':0, 'y':0}
    #additive_inc = {'x':0, 'y':0}
    #while niter == 0 or (abs(vis_normalization - np.median(data / clean_sim_data)) > 1e-2 and niter < 20):
        #niter += 1
        #vis_normalization = np.median(data / clean_sim_data)
        #print "Normalization from visibilities", vis_normalization
        #diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
        #diff_data = diff_data[0] + 1j * diff_data[1]
        #diff_norm = {}
        #diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis = 1)
        #diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis = 1)
        #additive_inc['x'] = np.repeat(np.mean(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis = 1, keepdims = True), data_shape['xx'][1], axis = 1)
        #additive_inc['y'] = np.repeat(np.mean(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis = 1, keepdims = True), data_shape['yy'][1], axis = 1)
        #additive['x'] = additive['x'] + additive_inc['x']
        #additive['y'] = additive['y'] + additive_inc['y']
        #data = data + np.concatenate((np.real(np.concatenate((additive_inc['x'].flatten(), np.zeros(len(data)/4,dtype='float32'), additive_inc['y'].flatten()))), np.imag(np.concatenate((additive_inc['x'].flatten(), np.zeros(len(data)/4,dtype='float32'), additive_inc['y'].flatten()))))).reshape((2, 4, data_shape['xx'][0], data_shape['xx'][1])).transpose((0, 2, 1, 3)).flatten()

    additive = 0
    while niter == 0 or (abs(vis_normalization - np.median((data / clean_sim_data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:,:,[0,3]])) > 1e-2 and niter < 20):
        niter += 1
        vis_normalization = np.median((data / clean_sim_data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:,:,[0,3]])
        print "Normalization from visibilities", vis_normalization
        diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
        diff_data = diff_data[0] + 1j * diff_data[1]
        diff_norm = {}
        diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis = 1)
        diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis = 1)

        additive_inc = np.repeat(np.mean((clean_sim_data * vis_normalization - data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1]), axis = -1, keepdims = True), data_shape['xx'][1], axis = -1)

        additive = additive + additive_inc
        data = data + additive_inc.flatten()


    if plot_data_error:
        vis_normalization = np.median((data / clean_sim_data).reshape(2, data_shape['xx'][0], 4, data_shape['xx'][1])[:,:,[0,3]])
        print "Normalization from visibilities", vis_normalization
        diff_data = (clean_sim_data * vis_normalization - data).reshape(2, len(data) / 2)
        diff_data = diff_data[0] + 1j * diff_data[1]
        diff_norm = {}
        diff_norm['x'] = la.norm(diff_data.reshape(data_shape['xx'][0], 4, data_shape['xx'][1])[:, 0], axis = 1)
        diff_norm['y'] = la.norm(diff_data.reshape(data_shape['yy'][0], 4, data_shape['yy'][1])[:, 3], axis = 1)
        plt.plot(diff_norm['x'][ubl_sort['x']])
        plt.plot(diff_norm['y'][ubl_sort['y']])
plt.show()


#vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
#print "Normalization from visibilities", vis_normalization
#diff_data = (clean_sim_data * vis_normalization - data)
#diff_norm = {}
#diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
#diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)

#if plot_data_error:
        #plt.plot(diff_norm['x'][ubl_sort['x']], label='original x error')
        #plt.plot(diff_norm['y'][ubl_sort['y']], label='original y error')

#if remove_additive:
    #niter = 0
    #additive = {'x':0, 'y':0}
    #additive_inc = {'x':0, 'y':0}
    #while niter == 0 or (abs(vis_normalization - np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))) > 1e-2 and niter < 20):
        #niter += 1
        #vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
        #print "Normalization from visibilities", vis_normalization
        #diff_data = clean_sim_data * vis_normalization - data
        #diff_norm = {}
        #diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
        #diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)
        #additive_inc['x'] = np.repeat(np.mean(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1, keepdims = True), data_shape['x'][1], axis = 1)
        #additive_inc['y'] = np.repeat(np.mean(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1, keepdims = True), data_shape['y'][1], axis = 1)
        #additive['x'] = additive['x'] + additive_inc['x']
        #additive['y'] = additive['y'] + additive_inc['y']
        #data = data + np.concatenate((additive_inc['x'].flatten(), additive_inc['y'].flatten()))

#if plot_data_error:
    #vis_normalization = np.median(np.concatenate((np.real(data) / np.real(clean_sim_data), np.imag(data) / np.imag(clean_sim_data))))
    #print "Normalization from visibilities", vis_normalization
    #diff_data = clean_sim_data * vis_normalization - data
    #diff_norm = {}
    #diff_norm['x'] = la.norm(diff_data[:data_shape['x'][0] * data_shape['x'][1]].reshape(*data_shape['x']), axis = 1)
    #diff_norm['y'] = la.norm(diff_data[data_shape['x'][0] * data_shape['x'][1]:].reshape(*data_shape['y']), axis = 1)
    #plt.plot(diff_norm['x'][ubl_sort['x']], label='new x error')
    #plt.plot(diff_norm['y'][ubl_sort['y']], label='new y error')
    #plt.legend();plt.show()

##renormalize the model
fake_solution = fake_solution * vis_normalization
clean_sim_data = clean_sim_data * vis_normalization
sim_data = clean_sim_data  + np.random.randn(len(data))/Ni**.5


#compute AtNi and AtNi.y

AtNi = A.transpose() * Ni
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

AtNi_data = AtNi.dot(data.astype(AtNi.dtype))
AtNi_data = AtNi.dot(sim_data.astype(AtNi.dtype))
#AtNi_data = np.array([AtNiiter.dot(data) for AtNiiter in AtNi])
#AtNi_sim_data = np.array([AtNiiter.dot(sim_data) for AtNiiter in AtNi])


precision = 'float64'
#compute AtNiA eigensystems
eigvl_filename = datadir + tag + '_%i%s_IQUV_%s.AtNiAel'%(valid_npix, vartag, precision)
eigvc_filename = datadir + tag + '_%i_%i%s_IQUV_%s.AtNiAev'%(valid_npix, valid_npix, vartag, precision)
if os.path.isfile(eigvl_filename) and os.path.isfile(eigvc_filename) and not force_recompute_AtNiAi_eig:
    print "Reading eigen system of AtNiA from %s and %s"%(eigvl_filename, eigvc_filename)
    sys.stdout.flush()
    del(AtNi)
    del(A)
    if precision != 'longdouble':
        eigvl = np.fromfile(eigvl_filename, dtype=precision)
        eigvc = np.fromfile(eigvc_filename, dtype=precision).reshape((4*valid_npix, 4*valid_npix))
    else:
        eigvl = np.fromfile(eigvl_filename, dtype='float64')
        eigvc = np.fromfile(eigvc_filename, dtype='float64').reshape((4*valid_npix, 4*valid_npix))
else:
    print "Computing AtNiA...", datetime.datetime.now()
    sys.stdout.flush()
    timer = time.time()
    AtNiA = AtNi.dot(A)
    print "%f minutes used"%(float(time.time()-timer)/60.)
    sys.stdout.flush()
    del(AtNi)
    del(A)
    AtNiA = AtNiA.astype(precision)
    print "Computing AtNiA eigensystem...", datetime.datetime.now()
    sys.stdout.flush()
    timer = time.time()
    eigvl, eigvc = sla.eigh(AtNiA)
    print "%f minutes used"%(float(time.time()-timer)/60.)
    sys.stdout.flush()
    del(AtNiA)

    if la.norm(eigvl) == 0:
        print "ERROR: Eigensistem calculation failed...matrix %i by %i is probably too large."%(4*valid_npix, 4*valid_npix)
    eigvl.tofile(eigvl_filename)
    eigvc.tofile(eigvc_filename)

plt.plot(eigvl)

#compute AtNiAi
precision = 'float64'
if eigvl.dtype != precision or eigvc.dtype != precision:
    print "casting eigen system from %s to %s"%(eigvl.dtype, precision)
    eigvl = eigvl.astype(precision)
    eigvc = eigvc.astype(precision)
max_eigv = np.max(eigvl)
rcondA = (1.e-12) / max_eigv
#if min(eigvl) <= 0:
    #rcondA = (1.e-12) / max_eigv#3.e-6 * 256/nside_standard
#else:
    #rcondA = 0
#print "Min eig %.2e"%min(eigvl), "Max eig %.2e"%max_eigv, "Add eig %.2e"%(max_eigv * rcondA)
#if min(eigvl) < 0 and np.abs(min(eigvl)) > max_eigv * rcondA:
    #print "!WARNING!"
    #print "!WARNING!: negative eigenvalue %.2e is smaller than the added identity %.2e! min rcond %.2e needed."%(min(eigvl), max_eigv * rcondA, np.abs(min(eigvl))/max_eigv)
    #print "!WARNING!"
eigvli = 1. / (max_eigv * rcondA + np.maximum(eigvl, 0))

AtNiAi_filename = datadir + tag + '_%i_%i%s_IQUV.AtNiAi%s%.1f'%(valid_npix, valid_npix, vartag, precision, np.log10(rcondA))
def get_AtNiAi():
    if os.path.isfile(AtNiAi_filename) and not force_recompute_AtNiAi:
        print "Reading AtNiAi matrix from %s"%AtNiAi_filename
        return np.fromfile(AtNiAi_filename, dtype=precision).reshape((4*valid_npix, 4*valid_npix))

    else:
        print "Computing AtNiAi matrix...", datetime.datetime.now(),
        sys.stdout.flush()
        timer = time.time()

        AtNiAi = (eigvc * eigvli).dot(eigvc.transpose())
        print "%f minutes used"%(float(time.time()-timer)/60.)
        del(eigvc)
        sys.stdout.flush()
        AtNiAi.tofile(AtNiAi_filename)
        return AtNiAi
AtNiAi = get_AtNiAi()
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#solve for x
x = AtNiAi.dot(AtNi_data.astype(AtNiAi.dtype))
sim_x = AtNiAi.dot(AtNi_sim_data.astype(AtNiAi.dtype))
sim_x_clean = fake_solution
del(AtNiAi)
A = get_A()
if remove_additive:
    chisq = np.sum(Ni * np.abs((A.dot(x.astype(A.dtype)) - data))**2) / float(len(data) - valid_npix - data_shape['xx'][0] - data_shape['yy'][0])
else:
    chisq = np.sum(Ni * np.abs((A.dot(x.astype(A.dtype)) - data))**2) / float(len(data) - valid_npix)
chisq_sim = np.sum(Ni * np.abs((A.dot(sim_x.astype(A.dtype)) - sim_data))**2) / float(len(sim_data) - valid_npix)

#####investigate optimal rcondA
print "lambda we use is", max_eigv * rcondA
print "best lambda inferred from data is", (len(data) - chisq*float(len(data) - valid_npix))/np.sum(x**2)
print "best lambda inferred from simulated data is", (len(data) - chisq_sim*float(len(sim_data) - valid_npix))/np.sum(sim_x**2)
del(A)
#compare measurement and simulation
if plot_projection:
    xproj = (x.dot(eigvc))[::-1]
    cleanproj = (sim_x_clean.dot(eigvc))[::-1]
    print "normalization", np.median(xproj[:200]/cleanproj[:200])
    simproj = (sim_x.dot(eigvc))[::-1]

    plt.subplot('211')
    plt.plot(xproj, 'b')
    plt.plot(cleanproj, 'g')
    plt.plot(simproj, 'r')
    plt.ylim(-3e5, 3e5)
    plt.subplot('212')
    plt.plot(xproj-cleanproj, 'b')
    plt.plot(simproj-cleanproj, 'r')
    plt.plot(eigvli[::-1]**.5, 'g')#(1/np.abs(eigvl[::-1])**.5)
    plt.ylim(-3e5, 3e5)
    plt.show()


#compute S
print "computing S...",
sys.stdout.flush()
timer = time.time()

pol_frac = .4#assuming QQ=UU=pol_frac*II
S = np.zeros((len(x), len(x)), dtype='float32')
for i in range(len(x)/4):
    S[i::len(x)/4, i::len(x)/4] = np.array([[1,0,0,0],[0,pol_frac,0,0],[0,0,pol_frac,0],[0,0,0,0]]) * sim_x_clean[i]**2#np.array([[1+pol_frac,0,0,1-pol_frac],[0,pol_frac,pol_frac,0],[0,pol_frac,pol_frac,0],[1-pol_frac,0,0,1+pol_frac]]) / 4 * (2*sim_x_clean[i])**2
#S = np.diag(sim_x_clean**2.)
print "Done."
print "%f minutes used"%(float(time.time()-timer)/60.)
sys.stdout.flush()

##generalized eigenvalue problem
#genSEv_filename = datadir + tag + '_%i_%i.genSEv_%s_%i'%(len(S), len(S), S_type, np.log10(rcondA))
#genSEvec_filename = datadir + tag + '_%i_%i.genSEvec_%s_%i'%(len(S), len(S), S_type, np.log10(rcondA))
#print "Computing generalized eigenvalue problem...",
#sys.stdout.flush()
#timer = time.time()
#genSEv, genSEvec = sla.eigh(S, b=AtNiAi)
#print "%f minutes used"%(float(time.time()-timer)/60.)
#genSEv.tofile(genSEv_filename)
#genSEvec.tofile(genSEvec_filename)

#genSEvecplot = np.zeros_like(equatorial_GSM_standard)
#for eigs in [-1,-2,1,0]:
    #genSEvecplot[pix_mask] = genSEvec[:,eigs]
    #hpv.mollview(genSEvecplot, coord=plotcoord, title=genSEv[eigs])

#plt.show()
#quit()

#####compute wiener filter##############
SEi_filename = datadir + tag + '_%i_%i%s.CSEi_IQUV_%s_%s_%.1f'%(len(S), len(S), vartag, S_type, precision, np.log10(rcondA))
if os.path.isfile(SEi_filename) and not force_recompute_SEi:
    print "Reading Wiener filter component...",
    sys.stdout.flush()
    SEi = sv.InverseCholeskyMatrix.fromfile(SEi_filename, len(S), precision)
else:
    print "Computing Wiener filter component...",
    sys.stdout.flush()
    timer = time.time()
    AtNiAi = get_AtNiAi()
    SEi = sv.InverseCholeskyMatrix(S + AtNiAi).astype(precision)
    SEi.tofile(SEi_filename, overwrite = True)
    print "%f minutes used"%(float(time.time()-timer)/60.)
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

#####apply wiener filter##############
print "Applying Wiener filter...",
sys.stdout.flush()
w_solution = S.dot(SEi.dotv(x))
w_GSM = S.dot(SEi.dotv(sim_x_clean))
w_sim_sol = S.dot(SEi.dotv(sim_x))
print "Memory usage: %.3fMB"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

def plot_IQU(solution, title, col, ncol = 6):
    #Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
    #I = Es[0] + Es[3]
    #Q = Es[0] - Es[3]
    #U = Es[1] + Es[2]
    IQUV = sol2map(solution)
    IQUV.shape = (4, IQUV.shape[0]/4)
    I = IQUV[0]
    Q = IQUV[1]
    U = IQUV[2]
    V = IQUV[3]
    pangle = 180*np.arctan2(Q,U)/2/PI
    plotcoordtmp = 'CG'
    hpv.mollview(np.log10(I), min=0, max =5, coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, col))
    #hpv.mollview(np.arcsinh(Q)/np.log(10), min=-np.arcsinh(10.**5)/np.log(10), max = np.arcsinh(10.**5)/np.log(10), coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, ncol + col))
    #hpv.mollview(np.arcsinh(U)/np.log(10), min=-np.arcsinh(10.**5)/np.log(10), max = np.arcsinh(10.**5)/np.log(10), coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, 2*ncol + col))
    hpv.mollview((Q**2+U**2)**.5/I, min = 0, max = 1, coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, ncol + col))
    hpv.mollview(pangle, min=-90, max = 90, coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, 2*ncol + col))

    hpv.mollview(np.arcsinh(V)/np.log(10), min=-np.arcsinh(10.**5)/np.log(10), max = np.arcsinh(10.**5)/np.log(10), coord=plotcoordtmp, title=title, nest=True,sub = (4, ncol, 3*ncol + col))
    if col == ncol:
        plt.show()

plot_IQU(fake_solution/sizes, 'GSM gridded', 1)
plot_IQU(x/sizes, 'raw solution, chi^2=%.2f'%chisq, 2)
plot_IQU(sim_x/sizes, 'raw simulated solution, chi^2=%.2f'%chisq_sim, 3)
plot_IQU(w_GSM/sizes, 'wienered GSM', 4)
plot_IQU(w_solution/sizes, 'wienered solution', 5)
plot_IQU(w_sim_sol/sizes, 'wienered simulated solution', 6)

#hpv.mollview(np.log10(fake_solution[np.array(final_index).tolist()]), min= 0, max =4, coord=plotcoord, title='GSM gridded', nest=True)
#hpv.mollview(np.log10((x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw solution, chi^2=%.2f'%chisq, nest=True)
#hpv.mollview(np.log10((sim_x/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='raw simulated solution, chi^2=%.2f'%chisq_sim, nest=True)
#hpv.mollview(np.log10((w_GSM/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered GSM', nest=True)
#hpv.mollview(np.log10((w_solution/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered solution', nest=True)
#hpv.mollview(np.log10((w_sim_sol/sizes)[np.array(final_index).tolist()]), min=0, max=4, coord=plotcoord, title='wienered simulated solution', nest=True)
#plt.show()
