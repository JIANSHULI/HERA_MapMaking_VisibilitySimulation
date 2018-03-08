#!/Users/JianshuLi/anaconda3/envs/Cosmology-Python27/bin/python

import simulate_visibilities.Bulm as Bulm
import simulate_visibilities.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time, ephem, sys, os, resource, datetime, warnings
import aipy as ap
import matplotlib.pyplot as plt
import healpy as hp
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob
import optparse, sys, os
from pylab import arange, show, cm

try:
	__file__
except NameError:
	script_dir = '/Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/simulate_visibilities/scripts'
	print 'Run IPython'
else:
	script_dir = os.path.dirname(os.path.realpath(__file__))
	print 'Run Python'

o = optparse.OptionParser()
o.add_option('-f', '--frequency', action='store', type='float', default=150., help='Frequency of the map in GHz, default 150 MHz.')
o.add_option('-r', '--resolution', action='store', type='float', default=0, help='Required resolution in arcminutes. The output resolution will be either 5 degrees or 0.8 degrees below 10 GHz, either 5 degrees or 0.4 degrees above 10 GHz.')
o.add_option('-u', '--unit', action='store', default='MJysr', help='Output unit, default MJysr. Other options include TCMB for CMB temperatures in Kelvin or TRJ for Rayleigh-Jeans temperatures in Kelvin.')
o.add_option('-o', '--outputpath', action='store', default=None, help='Path to store the output map, including file name.')
o.add_option('--ring', action='store_true', default=False, help='Output HEALPIX RING format. Default is NEST.')
o.add_option('--coord', action='store', default='CG', help='cood parameter for mollview plot. (C,G,E,or two combined for transformation from the first to the second.)')
o.add_option('--min', action='store', default=None, help='Minimum value for mollview plot.')
o.add_option('--max', action='store', default=None, help='Maximum value for mollview plot.')
o.add_option('--format', action='store', default='Log', help='Plot format for mollview plot.(lin or log)')
o.add_option('--cnorm', action='store', default='Lin', help='Color normalization for mollview plot.(hist,log,none=lin)')
o.add_option('--cmap', action='store', default='jet', help='Cmap option for mollview plot.(rainbow,hsv,bwr,binary,etc,none=jet)')
o.add_option('--nside', action='store', default='32', type='int', help='Nside value for map calculation.(8,16,32,64,128,256,512)')
o.add_option('--PCA', action='store', default='True', help='Whether to plot PCA components seperately or not.')

opts, args = o.parse_args(sys.argv[1:])
freq = opts.frequency #Unit: MHz
resolution = opts.resolution
unit = opts.unit
unit = 'Default'
oppath = opts.outputpath
convert_ring = opts.ring
mv_coord = opts.coord
mv_min = opts.min
mv_max = opts.max
mv_format = opts.format
mv_cnorm = opts.cnorm
mv_cmap = opts.cmap
mv_nside = opts.nside
mv_pca = opts.PCA

PI = np.pi
TPI = PI * 2
		
C = 299.792458
kB = 1.3806488 * 1.e-23
#script_dir = os.path.dirname(os.path.realpath(__file__))
#freq = 150 #Unit: MHz
plotcoord = mv_coord
nside_standard = mv_nside

################################################
#####################GSM###########################
#############################################
pca_num = 3
components = np.loadtxt(script_dir + '/../data/components.dat')
scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
pca_w = np.array([np.exp(scale_loglog(np.log(freq))) * hp.fitsfunc.read_map(script_dir + '/../data/gsm%s.fits'%pca_index + str(nside_standard)) * si.interp1d(components[:, 0], components[:, pca_index+1])(freq) for pca_index in range(1, pca_num+1)])
gsm_standard = np.sum(pca_w, axis=0)
#pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
#pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
#pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
#w1 = si.interp1d(components[:, 0], components[:, 2])
#w2 = si.interp1d(components[:, 0], components[:, 3])
#w3 = si.interp1d(components[:, 0], components[:, 4])
#pca1_w = np.exp(scale_loglog(np.log(freq))) * w1(freq) * pca1
#pca2_w = np.exp(scale_loglog(np.log(freq))) * w1(freq) * pca2
#pca3_w = np.exp(scale_loglog(np.log(freq))) * w1(freq) * pca3
#gsm_standard = pca1_w + pca2_w +pca3_w

# rotate sky map and converts to nest
equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
print "Rotating GSM_standard and converts to nest...",
sys.stdout.flush()
equ2013_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=2013.58))
ang0, ang1 = hp.rotator.rotateDirection(equ2013_to_gal_matrix,
					hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
print "done."
sys.stdout.flush()

if convert_ring:
	print 'Converting to HEALPIX RING...',
	sys.stdout.flush()
	try:
		import healpy as hp
	except:
		print "Healpy package not found. Cannot convert to HEALPIX RING format."
	equatorial_GSM_standard = hp.reorder(equatorial_GSM_standard, n2r=True)
	print 'done.'
	sys.stdout.flush()

# plot GSM map
print 'Plotting GSM'
sys.stdout.flush()
if mv_format == 'Log':
	if mv_pca:
		for pca_index in range(1, pca_num+1):
			hpv.mollview(np.log10(pca_w[pca_index-1]), min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%s PCA of %iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_index, pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
			#hpv.mollview(np.log10(globals()['pca'+ str(pca_index) +'_w']), min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%s PCA of %iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_index, pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
			plt.savefig(script_dir + '/../Output/PCA%s_GSM_Map_%.3eghz_%.2fres_%s_%s_healpy%s_%s_%s.pdf'%(pca_index, freq, resolution, mv_format, unit, ['NEST', 'RING'][int(convert_ring)], mv_cnorm, mv_cmap))
			plt.show(block=False)
			plt.close()
	hpv.mollview(np.log10(equatorial_GSM_standard), min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
	plt.savefig(script_dir + '/../Output/%iC-GSM_Map-%.3eMHz_%sNside_%s_%s_healpy%s_%s_%s.pdf'%(pca_num, freq, nside_standard, mv_format, unit, ['NEST', 'RING'][int(convert_ring)], mv_cnorm, mv_cmap))
	plt.show()
	plt.close()
elif mv_format == 'Lin':
	if mv_pca:
		for pca_index in range(1, pca_num+1):
			hpv.mollview(pca_w[pca_index-1], min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%s PCA of %iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_index, pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
			#hpv.mollview(globals()['pca'+ str(pca_index) +'_w'], min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%s PCA of %iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_index, pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
			plt.savefig(script_dir + '/../Output/PCA%s_GSM_Map_%.3eghz_%.2fres_%s_%s_healpy%s_%s_%s.pdf'%(pca_index, freq, resolution, mv_format, unit, ['NEST', 'RING'][int(convert_ring)], mv_cnorm, mv_cmap))
			plt.show(block=False)
			plt.close()
	hpv.mollview(equatorial_GSM_standard, min=mv_min, max=mv_max, coord=mv_coord, norm=mv_cnorm, cmap=getattr(cm,mv_cmap), title='%iC-GSM at %.3fMHz (%s View, Unit: %s, Nside: %s, Cmap: %s, Cnorm: %s)'%(pca_num, freq, mv_format, unit, nside_standard, mv_cmap, mv_cnorm), nest=True)
	plt.savefig(script_dir + '/../Output/%iC-GSM_Map-%.3eMHz_%sNside_%s_%s_healpy%s_%s_%s.pdf'%(pca_num, freq, nside_standard, mv_format, unit, ['NEST', 'RING'][int(convert_ring)], mv_cnorm, mv_cmap))
	plt.show()
	plt.close()
print 'Plotting Completed'
sys.stdout.flush()

#sys.stdout.flush()
#hpv.mollview(np.log10(equatorial_GSM_standard), min=0, max=4, coord=plotcoord, title='GSM at %.3fMHz (Coord: %s, Nside: %s)'%(freq, plotcoord, nside_standard), nest=True)
#plt.savefig(script_dir + '/../Output/GSM_Map-%.3fMHz-%s-%s.pdf'%(freq, nside_standard, plotcoord))
#plt.show()
#print 'Plotting Completed'
#sys.stdout.flush()