#!/usr/bin/python

import time, datetime
Timer_Start = time.time()
print('Programme Starts at: %s'%str(datetime.datetime.now()))
import ephem, sys, os, resource, warnings
# import simulate_visibilities.Bulm as Bulm
import HERA_MapMaking_VisibilitySimulation.Bulm
# import simulate_visibilities.simulate_visibilities as sv
import HERA_MapMaking_VisibilitySimulation.simulate_visibilities as sv
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import aipy as ap
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import healpy as hp
import healpy.rotator as hpr
import healpy.pixelfunc as hpf
import healpy.visufunc as hpv
import scipy.interpolate as si
import glob
import astropy
import hera_qm as hqm
# from aipy.miriad import pol2str
from astropy.io import fits
import HERA_MapMaking_VisibilitySimulation as mmvs
from pyuvdata import UVData, UVCal, UVFITS
from HERA_MapMaking_VisibilitySimulation import UVData as UVData_HR
from HERA_MapMaking_VisibilitySimulation import UVCal as UVCal_HR
from HERA_MapMaking_VisibilitySimulation import UVFITS as UVFITS_HR
import hera_cal as hc
from hera_cal.data import DATA_PATH as data_path_heracal
from HERA_MapMaking_VisibilitySimulation import DATA_PATH
from collections import OrderedDict as odict
from pyuvdata import utils as uvutils
import copy
import uvtools as uvt
import linsolve
from hera_cal.datacontainer import DataContainer
from astropy.time import Time
import omnical
import omnical.calibration_omni as omni
from memory_profiler import memory_usage as memuse
from collections import OrderedDict as odict
from collections import Counter
import pandas
import aipy.miriad as apm
import re
import copy
from hera_cal import utils, firstcal, cal_formats, redcal
from HERA_MapMaking_VisibilitySimulation.RA2LST import RA2LST
import multiprocessing
from multiprocessing import Pool
# from HERA_MapMaking_VisibilitySimulation.HERA_MapMaking_VisibilitySimulation_Functions import *

PI = np.pi
TPI = PI * 2

try:
	str2pol = {
		'I' :  1,   # Stokes Paremeters
		'Q' :  2,
		'U' :  3,
		'V' :  4,
		'rr': -1,   # Circular Polarizations
		'll': -2,
		'rl': -3,
		'lr': -4,
		'xx': -5,   # Linear Polarizations
		'yy': -6,
		'xy': -7,
		'yx': -8,
	}
	
	pol2str = {}
	for k in str2pol: pol2str[str2pol[k]] = k
except:
	from aipy.miriad import pol2str
	from aipy.miriad import str2pol

def pixelize(sky, nside_distribution, nside_standard, nside_start, thresh, final_index, thetas, phis, sizes):
	# thetas = []
	# phis = []
	for inest in range(12 * nside_start ** 2):
		pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis,
						sizes)
# newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside_start, inest, thresh, final_index, thetas, phis)
# thetas += newt.tolist()
# phis += newp.tolist()
# return np.array(thetas), np.array(phis)


def pixelize_helper(sky, nside_distribution, nside_standard, nside, inest, thresh, final_index, thetas, phis, sizes):
	# print "visiting ", nside, inest
	starti, endi = inest * nside_standard ** 2 / nside ** 2, (inest + 1) * nside_standard ** 2 / nside ** 2
	##local mean###if nside == nside_standard or np.std(sky[starti:endi])/np.mean(sky[starti:endi]) < thresh:
	if nside == nside_standard or np.std(sky[starti:endi]) < thresh:
		nside_distribution[starti:endi] = nside
		final_index[starti:endi] = len(thetas)  # range(len(thetas), len(thetas) + endi -starti)
		# return hp.pix2ang(nside, [inest], nest=True)
		newt, newp = hp.pix2ang(nside, [inest], nest=True)
		thetas += newt.tolist()
		phis += newp.tolist()
		sizes += (np.ones_like(newt) * nside_standard ** 2 / nside ** 2).tolist()
	# sizes += (np.ones_like(newt) / nside**2).tolist()
	
	else:
		# thetas = []
		# phis = []
		for jnest in range(inest * 4, (inest + 1) * 4):
			pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh, final_index, thetas,
							phis, sizes)
	# newt, newp = pixelize_helper(sky, nside_distribution, nside_standard, nside * 2, jnest, thresh)
	# thetas += newt.tolist()
	# phis += newp.tolist()
	# return np.array(thetas), np.array(phis)


def dot(A, B, C, nchunk=10):
	if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
		raise ValueError("A B C not all have 2 dims: %i %i %i" % (str(A.ndim), str(B.ndim), str(C.ndim)))
	
	chunk = len(C) / nchunk
	for i in range(nchunk):
		C[i * chunk:(i + 1) * chunk] = A[i * chunk:(i + 1) * chunk].dot(B)
	if chunk * nchunk < len(C):
		C[chunk * nchunk:] = A[chunk * nchunk:].dot(B)


def ATNIA(A, Ni, C, nchunk=20, dot=True):  # C=AtNiA
	if A.ndim != 2 or C.ndim != 2 or Ni.ndim != 1:
		raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i" % (A.ndim, C.ndim, Ni.ndim))
	
	expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1]) ** 2
	print('Process Starts at: %s' % str(datetime.datetime.now()))
	print " >>>>>>>>>>> Estimated time for A %i by %i <<<<<<<<<<<<" % (A.shape[0], A.shape[1]), expected_time, "minutes",
	sys.stdout.flush()
	
	chunk = len(C) / nchunk
	for i in range(nchunk):
		ltm = time.time()
		if dot:
			C[i * chunk:(i + 1) * chunk] = np.dot((A[:, i * chunk:(i + 1) * chunk] * Ni[:, None]).transpose(), A)
		else:
			C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
		if expected_time >= 1.:
			print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
			sys.stdout.flush()
	if chunk * nchunk < len(C):
		if dot:
			C[chunk * nchunk:] = np.dot((A[:, chunk * nchunk:] * Ni[:, None]).transpose(), A)
		else:
			C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)

def Chunk_Multiply_p(A, Ni, chunk, nchunk, expected_time, i, dot=True):
	ltm = time.time()
	print('Process Starts at: %s' % str(datetime.datetime.now()))
	if dot:
		C = np.dot((A_1 * Ni[:, None]).transpose(), A_2)
	else:
		C = np.einsum('ji,jk->ik', A_1 * Ni[:, None], A_2)
	if expected_time >= 1.:
		print "%i/%i: %.5fmins " % (i, nchunk, (time.time() - ltm) / 60.),
		sys.stdout.flush()
	return C

def Chunk_Multiply_small_p(A_1, A_2, Ni, chunk, nchunk, expected_time, i, j, dot=True):
	ltm = time.time()
	print('SubProcess Starts at: %s' % str(datetime.datetime.now()))
	try:
		from psutil import Process as process
		print('>>>>>>>> Number of Cpu_Affinity: %s. <<<<<<<<<' %str(len(process.cpu_affinity())))
	except:
		pass
	if dot:
		C = np.dot((A_1 * Ni[:, None]).transpose(), A_2)
	else:
		C = np.einsum('ji,jk->ik', A_1 * Ni[:, None], A_2)
	if expected_time >= 1.:
		print "%i-%i/%i: %.5fmins " % (i, j, nchunk, (time.time() - ltm) / 60.),
		sys.stdout.flush()
	return C

def Chunk_Multiply(i=0):
	ltm = time.time()
	print('Process Starts at: %s' % str(datetime.datetime.now()))
	C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
	if expected_time >= 1.:
		print "%i/%i: %.5fmins " % (i, nchunk**2, (time.time() - ltm) / 60.),
		sys.stdout.flush()
	return C[i * chunk:(i + 1) * chunk]

def ATNIA_parallel(A, Ni, C, nchunk=48, dot=True):  # C=AtNiA
	if A.ndim != 2 or C.ndim != 2 or Ni.ndim != 1:
		raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i" % (A.ndim, C.ndim, Ni.ndim))
	
	print('Processes Starts at: %s' % str(datetime.datetime.now()))
	expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1]) ** 2
	print " >>>>>>>>>> Estimated time for A %i by %i <<<<<<<<<<<<" % (A.shape[0], A.shape[1]), expected_time, "minutes",
	sys.stdout.flush()
	
	# nchunk = np.min((nchunk, multiprocessing.cpu_count()))
	print('Number of Multiprocesses: %s'%nchunk)
	chunk = len(C) / nchunk
	if chunk * len(A) > 25*10**6:
		chunk = int(25*10**6 / len(A))
	nchunk = len(C) / chunk
	print('>>>>>>>>>>>>>Modified Number of Chunks**2: %s*%s' % (nchunk, nchunk))
	# def Chunk_Multiply(C=None, chunk=None, nchunk=None, i=0):
	# 	ltm = time.time()
	# 	C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
	# 	if expected_time >= 1.:
	# 		print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
	# 		sys.stdout.flush()
	# 	return C[i * chunk:(i + 1) * chunk]
	
	pool = Pool()
	try:
		from psutil import Process as process
		print('>>>>>>>> Cpu_Affinity: %s. <<<<<<<<<' %str(process.cpu_affinity()))
	except:
		pass
	# C_mp = [pool.apply_async(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i)) for i in range(nchunk)]
	C_mp = [pool.apply_async(Chunk_Multiply_p, args=(A, Ni, chunk, nchunk, expected_time, i, dot)) for i in range(nchunk)]
	C[:chunk * nchunk] = np.array([p.get() for p in C_mp]).reshape(chunk * nchunk, A.shape[1])
	
	# C_mp = [[pool.apply_async(Chunk_Multiply_p, args=(A, Ni, chunk, nchunk, expected_time, i, j)) for j in range(nchunk)] for i in range(nchunk)]
	# C[:chunk * nchunk, :chunk * nchunk] = np.array([np.array([p.get() for p in C_mp[k]]).reshape(chunk, chunk * nchunk) for k in range(nchunk)]).reshape(chunk * nchunk, chunk * nchunk)
	
	# C[:chunk * nchunk] = np.array([pool.apply(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i)) for i in range(nchunk)]).reshape(chunk * nchunk, A.shape[1])
	# try:
	# 	del(C_mp)
	# except:
	# 	print('No C_map to delete.')
	
	# # pool.apply(Chunk_Multiply, args=(i,)) for i in range(nchunk)
	# for i in range(nchunk):
	# 	pool.apply(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i))
		# pool.apply(Chunk_Multiply, args=(i,))
	# 	ltm = time.time()
	# 	C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
	# 	if expected_time >= 1.:
	# 		print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
	# 		sys.stdout.flush()
	if chunk * nchunk < len(C):
		if dot:
			C[chunk * nchunk:] = np.dot((A[:, chunk * nchunk:] * Ni[:, None]).transpose(), A)
		else:
			C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)
		# C[:, chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, :] * Ni[:, None], A[:, chunk * nchunk:])
	print('Processes End at: %s' % str(datetime.datetime.now()))
	return C


def ATNIA_small_parallel(A, Ni, C, nchunk=6, dot=True):  # C=AtNiA
	if A.ndim != 2 or C.ndim != 2 or Ni.ndim != 1:
		raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i" % (A.ndim, C.ndim, Ni.ndim))
	
	print('Processes Starts at: %s' % str(datetime.datetime.now()))
	expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1]) ** 2
	print " >>>>>>>>>>>>> Estimated time for A %i by %i <<<<<<<<<<<<<<" % (A.shape[0], A.shape[1]), expected_time, "minutes",
	sys.stdout.flush()
	
	# nchunk = np.min((nchunk, multiprocessing.cpu_count()))
	print('>>>>>>>>>>>>>Number of Chunks**2: %s*%s' % (nchunk, nchunk))
	chunk = len(C) / nchunk
	if chunk * len(A) > 25*10**6:
		chunk = int(25*10**6 / len(A))
	nchunk = len(C) / chunk
	print('>>>>>>>>>>>>>Modified Number of Chunks**2: %s*%s' % (nchunk, nchunk))
	# def Chunk_Multiply(C=None, chunk=None, nchunk=None, i=0):
	# 	ltm = time.time()
	# 	C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
	# 	if expected_time >= 1.:
	# 		print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
	# 		sys.stdout.flush()
	# 	return C[i * chunk:(i + 1) * chunk]
	
	pool = Pool()
	try:
		from psutil import Process as process
		print('>>>>>>>> Cpu_Affinity: %s. <<<<<<<<<' %str(process.cpu_affinity()))
	except:
		pass
	# C_mp = [pool.apply_async(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i)) for i in range(nchunk)]
	# C_mp = [pool.apply_async(Chunk_Multiply_p, args=(A, Ni, chunk, nchunk, expected_time, i)) for i in range(nchunk)]
	# C[:chunk * nchunk] = np.array([p.get() for p in C_mp]).reshape(chunk * nchunk, A.shape[1])
	
	C_mp = [[pool.apply_async(Chunk_Multiply_small_p, args=(A[:, i * chunk:(i + 1) * chunk], A[:, j * chunk:(j + 1) * chunk], Ni, chunk, nchunk, expected_time, i, j, dot)) for j in range(nchunk)] for i in range(nchunk)]
	C[:chunk * nchunk, :chunk * nchunk] = np.array([np.array([p.get() for p in C_mp[k]]).transpose((1, 0, 2)).reshape(chunk, chunk * nchunk) for k in range(nchunk)]).reshape(chunk * nchunk, chunk * nchunk)
	
	# C[:chunk * nchunk] = np.array([pool.apply(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i)) for i in range(nchunk)]).reshape(chunk * nchunk, A.shape[1])
	# try:
	# 	del(C_mp)
	# except:
	# 	print('No C_map to delete.')
	
	# # pool.apply(Chunk_Multiply, args=(i,)) for i in range(nchunk)
	# for i in range(nchunk):
	# 	pool.apply(Chunk_Multiply_p, args=(C, A, Ni, chunk, nchunk, expected_time, i))
	# pool.apply(Chunk_Multiply, args=(i,))
	# 	ltm = time.time()
	# 	C[i * chunk:(i + 1) * chunk] = np.einsum('ji,jk->ik', A[:, i * chunk:(i + 1) * chunk] * Ni[:, None], A)
	# 	if expected_time >= 1.:
	# 		print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
	# 		sys.stdout.flush()
	if chunk * nchunk < len(C):
		if dot:
			C[chunk * nchunk:] = np.dot((A[:, chunk * nchunk:] * Ni[:, None]).transpose(), A)
			C[:, chunk * nchunk:] = np.dot((A[:, :] * Ni[:, None]).transpose(), A[:, chunk * nchunk:])
		else:
			C[chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, chunk * nchunk:] * Ni[:, None], A)
			C[:, chunk * nchunk:] = np.einsum('ji,jk->ik', A[:, :] * Ni[:, None], A[:, chunk * nchunk:])
	print('Processes End at: %s' % str(datetime.datetime.now()))
	return C

def test_para(x):
	print('Process Starts at: %s' % str(datetime.datetime.now()))
	return x**(1.51564864368484684621354), x**(-1.51564864368484684621354)


Test_Parallel = False
if Test_Parallel:
	pool = Pool()
	xpow_list = {}
	for i, x in enumerate(np.arange(12, 32, 1)):
		xpow_list[i] = pool.apply_async(test_para, args=(x,))
	# xpow_list = [pool.apply_async(test_para, args=(x,)) for x in range(12, 32, 1)]
	xpow_list = np.array([np.array([np.array(xpow_list[p].get()) * k for k in range(4)]) for p in range(len(xpow_list))])
	print(xpow_list)
	
	AtNiA_small_pr = np.zeros((A.shape[1], A.shape[1]), dtype=precision)
	AtNiA_pr = np.zeros((A.shape[1], A.shape[1]), dtype=precision)
	AtNiA = np.zeros((A.shape[1], A.shape[1]), dtype=precision)
	print "Computing AtNiA...", datetime.datetime.now()
	sys.stdout.flush()
	AtNiA_small_pr = ATNIA_small_parallel(A, Ni, AtNiA_small_pr)
	AtNiA_pr = ATNIA_parallel(A, Ni, AtNiA_pr)
	ATNIA(A, Ni, AtNiA)
	print "%f minutes used" % (float(time.time() - timer) / 60.)
	sys.stdout.flush()


def ATNIA_acurate(A, Ni, C, nchunk=24):  # C=AtNiA
	if A.ndim != 2 or C.ndim != 2 or Ni.ndim != 2:
		raise ValueError("A, AtNiA and Ni not all have correct dims: %i %i %i" % (A.ndim, C.ndim, Ni.ndim))
	
	expected_time = 1.3e-11 * (A.shape[0]) * (A.shape[1]) ** 2
	print "Estimated time for A %i by %i" % (A.shape[0], A.shape[1]), expected_time, "minutes",
	sys.stdout.flush()
	
	chunk = len(C) / nchunk
	for i in range(nchunk):
		ltm = time.time()
		C[i * chunk:(i + 1) * chunk] = np.einsum('ij,jk->ik', np.outer(A[:, i * chunk:(i + 1) * chunk].transpose(), Ni), A)
		if expected_time >= 1.:
			print "%i/%i: %.5fmins" % (i, nchunk, (time.time() - ltm) / 60.),
			sys.stdout.flush()
	if chunk * nchunk < len(C):
		C[chunk * nchunk:] = np.einsum('ij,jk->ik', np.outer(A[:, chunk * nchunk:].transpose(), Ni), A)


def get_A(additive_A=None, A_path='', force_recompute=False, Only_AbsData=False, nUBL_used=None, nt_used=None, valid_npix=None, thetas=None, phis=None, used_common_ubls=None, beam_heal_equ_x=None, beam_heal_equ_y=None, lsts=None, freq=None):
	if os.path.isfile(A_path) and not force_recompute:
		print "Reading A matrix from %s" % A_path
		sys.stdout.flush()
		if fit_for_additive:
			A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used))
		else:
			A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used, 2, nt_used, valid_npix))
	else:
		
		print "Computing A matrix..."
		sys.stdout.flush()
		A = np.empty((nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used), dtype='complex128')
		timer = time.time()
		for n in range(valid_npix):
			ra = phis[n]
			dec = PI / 2 - thetas[n]
			print "\r%f%% completed, %f minutes left" % (
				100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.)),
			sys.stdout.flush()
			
			A[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_x, tlist=lsts) / 2  # xx and yy are each half of I
			A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=lsts) / 2
		
		print "%f minutes used" % (float(time.time() - timer) / 60.)
		sys.stdout.flush()
		A.tofile(A_path)
	
	# #put in autocorr regardless of whats saved on disk
	# for i in range(nUBL_used):
	#     for p in range(2):
	#         A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
	#         A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]
	
	A.shape = (nUBL_used * 2 * nt_used, A.shape[-1])
	if not fit_for_additive:
		A = A[:, :valid_npix]
	else:
		A[:, valid_npix:] = additive_A[:, 1:]
	# Merge A
	try:
		if not Only_AbsData:
			return np.concatenate((np.real(A), np.imag(A))).astype('float64')
		else:
			return  A
	except MemoryError:
		print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
		sys.stdout.flush()
		Ashape = list(A.shape)
		Ashape[0] = Ashape[0] * 2
		np.real(A).tofile(A_path + 'tmpre')
		np.imag(A).tofile(A_path + 'tmpim')
		del (A)
		os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))
		
		os.system("rm %s" % (A_path + 'tmpim'))
		A = np.fromfile(A_path + 'tmpre', dtype='float64').reshape(Ashape)
		os.system("rm %s" % (A_path + 'tmpre'))
		print "done."
		sys.stdout.flush()
		return A.astype('float64')


def get_complex_data(real_data, nubl=None, nt=None):
	if len(real_data.flatten()) != 2 * nubl * 2 * nt:
		raise ValueError("Incorrect dimensions: data has length %i where nubl %i and nt %i together require length of %i." % (len(real_data), nubl, nt, 2 * nubl * 2 * nt))
	input_shape = real_data.shape
	real_data.shape = (2, nubl, 2, nt)
	result = real_data[0] + 1.j * real_data[1]
	real_data.shape = input_shape
	return result


def stitch_complex_data(complex_data):
	return np.concatenate((np.real(complex_data.flatten()), np.imag(complex_data.flatten()))).astype('float64')


def get_vis_normalization(data, clean_sim_data, data_shape=None):
	a = np.linalg.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
	b = np.linalg.norm(clean_sim_data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]), axis=0).flatten()
	return a.dot(b) / b.dot(b)


def sol2map(sol, valid_npix=None, npix=None, valid_pix_mask=None, final_index=None, sizes=None):
	solx = sol[:valid_npix]
	full_sol = np.zeros(npix)
	full_sol[valid_pix_mask] = solx / sizes
	return full_sol[final_index]


def sol2additive(sol, valid_npix=None, nUBL_used=None):
	return np.transpose(sol[valid_npix:].reshape(nUBL_used, 2, 2), (1, 0, 2))  # ubl by pol by re/im before transpose


def solve_phase_degen(data_xx, data_yy, model_xx, model_yy, ubls, plot=False):  # data should be time by ubl at single freq. data * phasegensolution = model
	if data_xx.shape != data_yy.shape or data_xx.shape != model_xx.shape or data_xx.shape != model_yy.shape or data_xx.shape[1] != ubls.shape[0]:
		raise ValueError("Shapes mismatch: %s %s %s %s, ubl shape %s" % (data_xx.shape, data_yy.shape, model_xx.shape, model_yy.shape, ubls.shape))
	A = np.zeros((len(ubls) * 2, 2))
	b = np.zeros(len(ubls) * 2)
	
	nrow = 0
	for p, (data, model) in enumerate(zip([data_xx, data_yy], [model_xx, model_yy])):
		for u, ubl in enumerate(ubls):
			amp_mask = (np.abs(data[:, u]) > (np.median(np.abs(data[:, u])) / 2.))
			A[nrow] = ubl[:2]
			b[nrow] = omni.medianAngle(np.angle(model[:, u] / data[:, u])[amp_mask])
			nrow += 1
	phase_cal = omni.solve_slope(np.array(A), np.array(b), 1)
	if plot:
		plt.hist((np.array(A).dot(phase_cal) - b + PI) % TPI - PI)
		plt.title('phase fitting error')
		plt.show()
	
	# sooolve
	return phase_cal


class LastUpdatedOrderedDict(odict):
	'Store items in the order the keys were last added'
	
	def __setitem__(self, key, value):
		if key in self:
			del self[key]
		odict.__setitem__(self, key, value)


def S_casa_v_t(v, t=2015.5):
	S_0 = 2190.294  # S_casA_1GHz
	alpha = 0.725
	belta = 0.0148
	tau = 6.162 * 1.e-5
	
	a = -0.00633  # +-0.00024 year-1
	b = 0.00039  # +-0.00008 year -1
	c = 1.509 * 1.e-7  # +-0.162*1.e-7 year-1
	
	v *= 1.e-3
	
	# print(v) # from MHz to GHz
	# print(t) # in decimal year
	
	S_casa_v = S_0 * v ** (-alpha + belta * np.log(v)) * np.exp(-tau * v ** (-2.1))  # S_0: 2015.5
	
	d_speed_log_v = a + b * np.log(v) + c * v ** (-2.1)  # a,b,c: 2005.0
	
	S_casa_v_t = np.exp((t - 2015.5) * d_speed_log_v + np.log(S_casa_v))
	
	# print(d_speed_log_v)
	
	return S_casa_v_t


def S_cyga_v(v, t=2005):
	S_cyga_v = 3.835 * 1.e5 * v ** (-0.718) * np.exp(-0.342 * (21.713 / v) ** 2.1)
	
	return S_cyga_v

def UVData2AbsCalDict(datanames, pol_select=None, pop_autos=True, return_meta=False, filetype='miriad',
					  pick_data_ants=True):
	"""
	turn a list of pyuvdata.UVData objects or a list of miriad or uvfits file paths
	into the datacontainer dictionary form that AbsCal requires. This format is
	keys as antennas-pair + polarization format, Ex. (1, 2, 'xx')
	and values as 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

	Parameters:
	-----------
	datanames : list of either strings of data file paths or list of UVData instances
				to concatenate into a single dictionary

	pol_select : list of polarization strings to keep

	pop_autos : boolean, if True: remove autocorrelations

	return_meta : boolean, if True: also return antenna and unique frequency and LST arrays

	filetype : string, filetype of data if datanames is a string, options=['miriad', 'uvfits']
				can be ingored if datanames contains UVData objects.

	pick_data_ants : boolean, if True and return_meta=True, return only antennas in data

	Output:
	-------
	if return_meta is True:
		(data, flags, antpos, ants, freqs, times, lsts, pols)
	else:
		(data, flags)

	data : dictionary containing baseline-pol complex visibility data
	flags : dictionary containing data flags
	antpos : dictionary containing antennas numbers as keys and position vectors
	ants : ndarray containing unique antennas
	freqs : ndarray containing frequency channels (Hz)
	times : ndarray containing julian date bins of data
	lsts : ndarray containing LST bins of data (radians)
	pols : ndarray containing list of polarization index integers
	"""
	# check datanames is not a list
	if type(datanames) is not list and type(datanames) is not np.ndarray:
		if type(datanames) is str:
			# assume datanames is a file path
			uvd = UVData()
			suffix = os.path.splitext(datanames)[1]
			if filetype == 'uvfits' or suffix == '.uvfits':
				uvd.read_uvfits(datanames)
				uvd.unphase_to_drift()
			elif filetype == 'miriad':
				uvd.read_miriad(datanames)
		else:
			# assume datanames is a UVData instance
			uvd = datanames
	else:
		# if datanames is a list, check data types of elements
		if type(datanames[0]) is str:
			# assume datanames contains file paths
			uvd = UVData()
			suffix = os.path.splitext(datanames[0])[1]
			if filetype == 'uvfits' or suffix == '.uvfits':
				uvd.read_uvfits(datanames)
				uvd.unphase_to_drift()
			elif filetype == 'miriad':
				uvd.read_miriad(datanames)
		else:
			# assume datanames contains UVData instances
			uvd = reduce(operator.add, datanames)
	
	# load data
	d, f = firstcal.UVData_to_dict([uvd])
	
	# pop autos
	if pop_autos:
		for i, k in enumerate(d.keys()):
			if k[0] == k[1]:
				d.pop(k)
				f.pop(k)
	
	# turn into datacontainer
	data, flags = DataContainer(d), DataContainer(f)
	
	# get meta
	if return_meta:
		freqs = np.unique(uvd.freq_array)
		times = np.unique(uvd.time_array)
		lsts = np.unique(uvd.lst_array)
		antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
		antpos = odict(zip(ants, antpos))
		pols = uvd.polarization_array
		return data, flags, antpos, ants, freqs, times, lsts, pols
	else:
		return data, flags


def UVData2AbsCalDict_Auto(datanames, pol_select=None, pop_autos=True, return_meta=False, filetype='miriad',
					  pick_data_ants=True, svmemory=True, Time_Average=1, Frequency_Average=1, Dred=False, inplace=True, tol=5.e-4, Select_freq=False, Select_time=False, Badants=[], Parallel_Files=False):
	"""
	turn a list of pyuvdata.UVData objects or a list of miriad or uvfits file paths
	into the datacontainer dictionary form that AbsCal requires. This format is
	keys as antennas-pair + polarization format, Ex. (1, 2, 'xx')
	and values as 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

	Parameters:
	-----------
	datanames : list of either strings of data file paths or list of UVData instances
				to concatenate into a single dictionary

	pol_select : list of polarization strings to keep

	pop_autos : boolean, if True: remove autocorrelations

	return_meta : boolean, if True: also return antenna and unique frequency and LST arrays

	filetype : string, filetype of data if datanames is a string, options=['miriad', 'uvfits']
				can be ingored if datanames contains UVData objects.

	pick_data_ants : boolean, if True and return_meta=True, return only antennas in data

	Output:
	-------
	if return_meta is True:
		(data, flags, antpos, ants, freqs, times, lsts, pols, autocorr, autocorr_flags)
	else:
		(data, flags, autocorr, autocorr_flags))

	data : dictionary containing baseline-pol complex visibility data
	flags : dictionary containing data flags
	antpos : dictionary containing antennas numbers as keys and position vectors
	ants : ndarray containing unique antennas
	freqs : ndarray containing frequency channels (Hz)
	times : ndarray containing julian date bins of data
	lsts : ndarray containing LST bins of data (radians)
	pols : ndarray containing list of polarization index integers
	"""
	# check datanames is not a list
	if type(datanames) is not list and type(datanames) is not np.ndarray:
		if type(datanames) is str:
			# assume datanames is a file path
			# uvd = UVData()
			uvd = UVData_HR() # Self-Contain Module form pyuvdata
			suffix = os.path.splitext(datanames)[1]
			if filetype == 'uvfits' or suffix == '.uvfits':
				uvd.read_uvfits(datanames)
				uvd.unphase_to_drift()
			elif filetype == 'miriad':
				uvd.read_miriad(datanames, Time_Average=Time_Average, Frequency_Average=Frequency_Average, Dred=Dred, inplace=inplace, tol=tol, Select_freq=Select_freq, Select_time=Select_time, Badants=Badants, Parallel_Files=Parallel_Files)
		else:
			# assume datanames is a UVData instance
			uvd = datanames
	else:
		# if datanames is a list, check data types of elements
		if type(datanames[0]) is str:
			# assume datanames contains file paths
			# uvd = UVData()
			uvd = UVData_HR()  # Self-Contain Module form pyuvdata
			suffix = os.path.splitext(datanames[0])[1]
			if filetype == 'uvfits' or suffix == '.uvfits':
				uvd.read_uvfits(datanames)
				uvd.unphase_to_drift()
			elif filetype == 'miriad':
				uvd.read_miriad(datanames, Time_Average=Time_Average, Frequency_Average=Frequency_Average, Dred=Dred, inplace=inplace, tol=tol, Select_freq=Select_freq, Select_time=Select_time, Badants=Badants, Parallel_Files=Parallel_Files)
		else:
			# assume datanames contains UVData instances
			uvd = reduce(operator.add, datanames)
	
	if return_meta:
		# freqs = np.unique(uvd.freq_array)
		# times = np.unique(uvd.time_array)
		# lsts = np.unique(uvd.lst_array)
		freqs = uvd.freq_array.squeeze()
		times = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
		lsts = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
		antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
		antpos = odict(zip(ants, antpos))
		pols = uvd.polarization_array
		redundancy_temp = uvd.redundancy
		redundancy = np.zeros(0, np.int)
		if len(times) != len(np.unique(uvd.time_array)):
			print ('Times Overlapping.')
		else:
			print ('No Time Overlapping.')
		if len(lsts) != len(np.unique(uvd.lst_array)):
			print ('Lsts Overlapping.')
		else:
			print ('No Lst Overlapping.')
		if len(freqs) != len(np.unique(uvd.freq_array)):
			print ('Frequencies Overlapping.')
		else:
			print ('No Frequency Overlapping.')
	
	# load data
	if not svmemory:
		d, f = firstcal.UVData_to_dict([uvd])
	else:
		d, f = UVData_to_dict_svmemory([uvd], svmemory=svmemory)
	autos = {}
	autos_flags = {}
	
	# pop autos
	if pop_autos:
		for i, k in enumerate(d.keys()):
			if k[0] == k[1]:
				autos[k] = d[k]
				autos_flags[k] = f[k]
				# redundancy = np.append(redundancy[:i], redundancy[i+1:])
				d.pop(k)
				f.pop(k)
				print('Index of Autocorr popped out: %s.'%(str(i) + ': ' + str(k)))
			else:
				redundancy = np.append(redundancy, redundancy_temp[i])
				print('Index of Baselines not popped out: %s'%(str(i) + ': ' + str(k)))
	# turn into datacontainer
	try:
		data, flags = DataContainer(d), DataContainer(f)
		autos_pro, autos_flags_pro = DataContainer(autos), DataContainer(autos_flags)
		print('Data_Contained.')
	except:
		print('Not Data_Contained.')
	
	# get meta
	if return_meta:
		# freqs = np.unique(uvd.freq_array)
		# times = np.unique(uvd.time_array)
		# lsts = np.unique(uvd.lst_array)
		# freqs = uvd.freq_array
		# times = uvd.time_array
		# lsts = uvd.lst_array
		# antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
		# antpos = odict(zip(ants, antpos))
		# pols = uvd.polarization_array
		# if len(times) != len(np.unique(uvd.time_array)):
		# 	print ('Times Overlapping.')
		# else:
		# 	print ('No Time Overlapping.')
		# if len(lsts) != len(np.unique(uvd.lst_array)):
		# 	print ('Lsts Overlapping.')
		# else:
		# 	print ('No Lst Overlapping.')
		# if len(freqs) != len(np.unique(uvd.freq_array)):
		# 	print ('Frequencies Overlapping.')
		# else:
		# 	print ('No Frequency Overlapping.')
		return data, flags, antpos, ants, freqs, times, lsts, pols, autos_pro, autos_flags_pro, redundancy
	else:
		return data, flags, autos_pro, autos_flags_pro, redundancy


def set_lsts_from_time_array_hourangle(times, lon='21:25:41.9', lat='-30:43:17.5'):
	"""Set the lst_array based from the time_array."""
	lsts = []
	lst_array = np.zeros(len(np.unique(times)))
	# latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
	for ind, jd in enumerate(np.unique(times)):
		t = Time(jd, format='jd', location=(lon, lat))
		lst_array[np.where(np.isclose(
			jd, times, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').hourangle
	return lst_array

def set_lsts_from_time_array_radian(times, lon='21:25:41.9', lat='-30:43:17.5'):
	"""Set the lst_array based from the time_array."""
	lsts = []
	lst_array = np.zeros(len(np.unique(times)))
	# latitude, longitude, altitude = self.telescope_location_lat_lon_alt_degrees
	for ind, jd in enumerate(np.unique(times)):
		t = Time(jd, format='jd', location=(lon, lat))
		lst_array[np.where(np.isclose(
			jd, times, atol=1e-6, rtol=1e-12))] = t.sidereal_time('apparent').radian
	return lst_array

def UVData_to_dict(uvdata_list, filetype='miriad'):
	""" Turn a list of UVData objects or filenames in to a data and flag dictionary.

        Make dictionary with blpair key first and pol second key from either a
        list of UVData objects or a list of filenames with specific file_type.

        Args:
            uvdata_list: list of UVData objects or strings of filenames.
            filetype (string, optional): type of file if uvdata_list is
                a list of filenames

        Return:
            data (dict): dictionary of data indexed by pol and antenna pairs
            flags (dict): dictionary of flags indexed by pol and antenna pairs
        """
	
	# d, f = {}, {}
	d, f = odict(), odict()
	# d, f = LastUpdatedOrderedDict(), LastUpdatedOrderedDict()
	for uv_in in uvdata_list:
		if type(uv_in) == str:
			fname = uv_in
			uv_in = UVData()
			# read in file without multiple if statements
			getattr(uv_in, 'read_' + filetype)(fname)
		
		# iterate over unique baselines
		for nbl, (i, j) in enumerate(map(uv_in.baseline_to_antnums, np.unique(uv_in.baseline_array))):
			if (i, j) not in d:
				d[i, j] = {}
				f[i, j] = {}
			for ip, pol in enumerate(uv_in.polarization_array):
				pol = pol2str[pol]
				new_data = copy.copy(uv_in.get_data((i, j, pol)))
				new_flags = copy.copy(uv_in.get_flags((i, j, pol)))
				
				if pol not in d[(i, j)]:
					d[(i, j)][pol] = new_data
					f[(i, j)][pol] = new_flags
				else:
					d[(i, j)][pol] = np.concatenate(
						[d[(i, j)][pol], new_data ])
					f[(i, j)][pol] = np.concatenate(
						[f[(i, j)][pol], new_flags ])
	return d, f


def UVData_to_dict_svmemory(uvdata_list, filetype='miriad', svmemory=True):
	""" Turn a list of UVData objects or filenames in to a data and flag dictionary.

        Make dictionary with blpair key first and pol second key from either a
        list of UVData objects or a list of filenames with specific file_type.

        Args:
            uvdata_list: list of UVData objects or strings of filenames.
            filetype (string, optional): type of file if uvdata_list is
                a list of filenames

        Return:
            data (dict): dictionary of data indexed by pol and antenna pairs
            flags (dict): dictionary of flags indexed by pol and antenna pairs
        """
	
	# d, f = {}, {}
	d, f = odict(), odict()
	# d, f = LastUpdatedOrderedDict(), LastUpdatedOrderedDict()
	print (len(uvdata_list))
	for id_uv, uv_in in enumerate(uvdata_list):
		if type(uv_in) == str:
			fname = uv_in
			uv_in = UVData()
			# read in file without multiple if statements
			getattr(uv_in, 'read_' + filetype)(fname)
		
		# iterate over unique baselines
		for nbl, (i, j) in enumerate(map(uv_in.baseline_to_antnums, uv_in.baseline_array.reshape(uv_in.Ntimes, uv_in.Nbls)[0, :])):
			print('Pair in UVData_to_dict_svmemory: %s(%s, %s)'%(nbl, i, j))
			if (i, j) not in d:
				# d[i, j] = {}
				# f[i, j] = {}
				d[i, j] = odict()
				f[i, j] = odict()
			for ip, pol in enumerate(uv_in.polarization_array):
				pol = pol2str[pol]
				new_data = copy.copy(uv_in.get_data((i, j, pol)))
				new_flags = copy.copy(uv_in.get_flags((i, j, pol)))
				
				if pol not in d[(i, j)]:
					d[(i, j)][pol] = new_data
					f[(i, j)][pol] = new_flags
				else:
					d[(i, j)][pol] = np.concatenate(
						[d[(i, j)][pol], new_data])
					f[(i, j)][pol] = np.concatenate(
						[f[(i, j)][pol], new_flags])
		if svmemory:
			uvdata_list[id_uv] = []
			print ('Blank uvdata: %s'%id_uv)
			
	return d, f


def Compress_Data_by_Average(data=None, dflags=None, Time_Average=1, Frequency_Average=1, data_freqs=None, data_times=None, data_lsts=None, Contain_Autocorr=True, autocorr_data_mfreq=None, DicData=False, pol=None, use_select_time=False, use_select_freq=False, use_RFI_AlmostFree_Freq=False, Freq_RFI_AlmostFree_bool=None):
	if np.mod(data[data.keys()[0]].shape[0], Time_Average) != 0:
		if (data[data.keys()[0]].shape[0] / Time_Average) < 1.:
			# Time_Average = 1
			Time_Average = np.min((data[data.keys()[0]].shape[0], Time_Average))
	if np.mod(data[data.keys()[0]].shape[1], Frequency_Average) != 0:
		if (data[data.keys()[0]].shape[1] / Frequency_Average) < 1.:
			# Frequency_Average = 1
			Frequency_Average = np.min((data[data.keys()[0]].shape[1], Frequency_Average))
	
	remove_times = np.mod(data[data.keys()[0]].shape[0], Time_Average)
	remove_freqs = np.mod(data[data.keys()[0]].shape[1], Frequency_Average)
	if remove_times == 0:
		remove_times = -data[data.keys()[0]].shape[0]
	if remove_freqs == 0:
		remove_freqs = -data[data.keys()[0]].shape[1]
	print ('Time_Average: %s; Frequency_Average: %s.' % (Time_Average, Frequency_Average))
	print ('Remove_Times: %s; Remove_Freqs: %s.' % (remove_times, remove_freqs))
	
	# data_ff = {}
	# dflags_ff = {}
	# autocorr_data_mfreq_ff = {}
	# data_freqs_ff = {}
	# data_times_ff = {}
	# data_lsts_ff = {}
	
	# for i in range(2):
	timer = time.time()
	data_ff = LastUpdatedOrderedDict()
	dflags_ff = LastUpdatedOrderedDict()
	# autocorr_data_mfreq_ff[i] = LastUpdatedOrderedDict()
	# data_freqs_ff[i] = LastUpdatedOrderedDict()
	# data_times_ff[i] = LastUpdatedOrderedDict()
	# data_lsts_ff[i] = LastUpdatedOrderedDict()
	
	data_freqs = data_freqs[: -remove_freqs]
	if use_RFI_AlmostFree_Freq:
		try:
			data_freqs = data_freqs[Freq_RFI_AlmostFree_bool]
		except:
			print('data_freqs not use_RFI_AlmostFree_Freq or already and not again.')
	data_times = data_times[: -remove_times]
	data_lsts = data_lsts[: -remove_times]
	
	if not Select_freq:
		data_freqs_ff = np.mean(data_freqs.reshape(len(data_freqs) / Frequency_Average, Frequency_Average), axis=-1)
	else:
		data_freqs_ff = data_freqs.reshape(len(data_freqs) / Frequency_Average, Frequency_Average)[:, 0]
		
	if not Select_time:
		data_times_ff = np.mean(data_times.reshape(len(data_times) / Time_Average, Time_Average), axis=-1)
		data_lsts_ff = np.mean(data_lsts.reshape(len(data_lsts) / Time_Average, Time_Average), axis=-1)
	else:
		data_times_ff = data_times.reshape(len(data_times) / Time_Average, Time_Average)[:, 0]
		data_lsts_ff = data_lsts.reshape(len(data_lsts) / Time_Average, Time_Average)[:, 0]
	
	for id_key, key in enumerate(data.keys()):
		
		data[key] = data[key][: -remove_times, : -remove_freqs]
		# autocorr_data_mfreq[i] = autocorr_data_mfreq[i][: -remove_times, : -remove_freqs]
		dflags[key] = dflags[key][: -remove_times, : -remove_freqs]
		# data_freqs[i] = data_freqs[i][: -remove_freqs]
		# data_times[i] = data_times[i][: -remove_times]
		# data_lsts[i] = data_lsts[i][: -remove_times]
		
		if use_RFI_AlmostFree_Freq:
			data[key] = data[key][:, Freq_RFI_AlmostFree_bool]
			dflags[key] = dflags[key][:, Freq_RFI_AlmostFree_bool]
		
		if id_key == 0:
			print ('rawData_Shape-%s: %s' % (key, data[key].shape))
			print ('rawDflags_Shape-%s: %s' % (key, dflags[key].shape))
			print ('rawAutocorr_Shape: (%s, %s)' % autocorr_data_mfreq.shape)
			print ('rawData_Freqs: %s' % (len(data_freqs)))
			print ('rawData_Times: %s' % (len(data_times)))
			print ('rawData_Lsts: %s' % (len(data_lsts)))
			
		data_ff[key] = np.mean(data[key].reshape(data[key].shape[0] / Time_Average, Time_Average, data[key].shape[1]), axis=1) if use_select_time else data[key].reshape(data[key].shape[0] / Time_Average, Time_Average, data[key].shape[1])[:, 0, :]
		data_ff[key] = np.mean(data_ff[key].reshape(data[key].shape[0] / Time_Average, data[key].shape[1] / Frequency_Average, Frequency_Average), axis=-1) if use_select_freq else data_ff[key].reshape(data[key].shape[0] / Time_Average, data[key].shape[1] / Frequency_Average, Frequency_Average)[:, :, 0]
		if DicData:
			data.pop(key)
		else:
			data.__delitem__(key)
		
		dflags_ff[key] = np.mean(dflags[key].reshape(dflags[key].shape[0] / Time_Average, Time_Average, dflags[key].shape[1]), axis=1) if use_select_time else dflags[key].reshape(dflags[key].shape[0] / Time_Average, Time_Average, dflags[key].shape[1])[:, 0, :]
		dflags_ff[key] = (np.mean(dflags_ff[key].reshape(dflags[key].shape[0] / Time_Average, dflags[key].shape[1] / Frequency_Average, Frequency_Average), axis=-1) == 1)  if use_select_freq else (dflags_ff[key].reshape(dflags[key].shape[0] / Time_Average, dflags[key].shape[1] / Frequency_Average, Frequency_Average)[:, :, 0] == 1)
		if DicData:
			dflags.pop(key)
		else:
			dflags.__delitem__(key)
		
		
	print('compress_Pol_%s is done. %s seconds used.' % (pol, time.time() - timer))
	
	data = copy.deepcopy(data_ff)
	dflags = copy.deepcopy(dflags_ff)
	# autocorr_data_mfreq = copy.deepcopy(autocorr_data_mfreq_ff)
	data_freqs = copy.deepcopy(data_freqs_ff)
	data_times = copy.deepcopy(data_times_ff)
	data_lsts = copy.deepcopy(data_lsts_ff)
	
	if Contain_Autocorr:
		autocorr_data_mfreq = autocorr_data_mfreq[: -remove_times, : -remove_freqs]
		if use_RFI_AlmostFree_Freq:
			autocorr_data_mfreq = autocorr_data_mfreq[:, Freq_RFI_AlmostFree_bool]
		autocorr_data_mfreq_ff = np.mean(autocorr_data_mfreq.reshape(autocorr_data_mfreq.shape[0] / Time_Average, Time_Average, autocorr_data_mfreq.shape[1]), axis=1) if use_select_time else autocorr_data_mfreq.reshape(autocorr_data_mfreq.shape[0] / Time_Average, Time_Average, autocorr_data_mfreq.shape[1])[:, 0, :]
		autocorr_data_mfreq_ff = np.mean(autocorr_data_mfreq_ff.reshape(autocorr_data_mfreq.shape[0] / Time_Average, autocorr_data_mfreq.shape[1] / Frequency_Average, Frequency_Average), axis=-1) if use_select_freq else autocorr_data_mfreq_ff.reshape(autocorr_data_mfreq.shape[0] / Time_Average, autocorr_data_mfreq.shape[1] / Frequency_Average, Frequency_Average)[:, :, 0]
		autocorr_data_mfreq = copy.deepcopy(autocorr_data_mfreq_ff)
		del (autocorr_data_mfreq_ff)
		try:
			print ('Autocorr_Shape: (%s, %s)' % autocorr_data_mfreq.shape)
		except:
			print('Shape of autocorr results printing not complete.')
	
	del (data_ff)
	del (dflags_ff)
	# del (autocorr_data_mfreq_ff)
	del (data_freqs_ff)
	del (data_times_ff)
	del (data_lsts_ff)
	
	try:
		print ('Data_Shape-%s: %s' % (key, data[key].shape))
		print ('Dflags_Shape-%s: %s' % (key, dflags[key].shape))
		# print ('Autocorr_Shape: (%s, %s)' % autocorr_data_mfreq[i].shape)
		print ('Data_Freqs: %s' % (len(data_freqs)))
		print ('Data_Times: %s' % (len(data_times)))
		print ('Data_Lsts: %s' % (len(data_lsts)))
	except:
		print('Shape of results printing not complete.')
	
	if Contain_Autocorr:
		return data, dflags, autocorr_data_mfreq, data_freqs, data_times, data_lsts
	else:
		return data, dflags, data_freqs, data_times, data_lsts


def De_Redundancy(dflags=None, antpos=None, ants=None, SingleFreq=True, MultiFreq=True, Conjugate_CertainBSL=False, Conjugate_CertainBSL2=False, Conjugate_CertainBSL3=False, data_freqs=None, Nfreqs=64, data_times=None, Ntimes=None, FreqScaleFactor=1.e6, Frequency_Select=None, Frequency_Select_Index=None, vis_data_mfreq=None, vis_data=None,
                  tol=5.e-4, Badants=[], freq=None, nside_standard=None, C=299.792458, 	baseline_safety_low = 30., baseline_safety_factor = 0.5, baseline_safety_xx = 10., baseline_safety_yy = 30., baseline_safety_xx_max = 70., baseline_safety_yy_max = 70.):
	
	antloc = {}
	
	if SingleFreq:
		flist = {}
		index_freq = {}
		if Frequency_Select_Index is not None:
			for i in range(2):
				index_freq[i] = Frequency_Select_Index[i]
		elif data_freqs is not None:
			for i in range(2):
				flist[i] = np.array(data_freqs[i]) / FreqScaleFactor
				try:
					# index_freq[i] = np.where(flist[i] == 150)[0][0]
					#		index_freq = 512
					index_freq[i] = np.abs(Frequency_Select - flist[i]).argmin()
				except:
					index_freq[i] = len(flist[i]) / 2
		
		dflags_sf = {}  # single frequency
		for i in range(2):
			dflags_sf[i] = LastUpdatedOrderedDict()
			for key in dflags[i].keys():
				dflags_sf[i][key] = dflags[i][key][:, index_freq[i]]
		
		if vis_data_mfreq is not None and vis_data is None:
			vis_data = {}
			for i in range(2):
				vis_data[i] = vis_data_mfreq[i][index_freq[i], :, :]  # [pol][ freq, time, bl]
		elif vis_data is not None:
			vis_data = vis_data
		else:
			raise ValueError('No vis_data provided or calculated from vis_data_mfreq.')
	
	# ant locations
	for i in range(2):
		antloc[i] = np.array(map(lambda k: antpos[i][k], ants[i]))
	
	# bls = [[], []]
	# for i in range(2):
	# 	bls[i] = odict([(x, np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])) for x in data[i].keys() if ])
	# 	# bls[i] = odict([(x, antpos[i][x[0]] - antpos[i][x[1]]) for x in dflags[i].keys()])
	# 	# bls[1] = odict([(y, antpos_yy[y[0]] - antpos_yy[y[1]]) for y in data_yy.keys()])
	# 	bls = np.array(bls)
	
	bls = [[], []]
	# bls_test = [[], []]
	for i in range(2):
		bls[i] = odict()
		for x in data[i].keys():
			if (la.norm((antpos[i][x[0]] - antpos[i][x[1]])) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor) and (la.norm((antpos[i][x[0]] - antpos[i][x[1]])) / (C / freq) >= baseline_safety_low) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0] >= baseline_safety_xx) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] >= baseline_safety_yy) \
					and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0] <= baseline_safety_xx_max) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] <= baseline_safety_yy_max):
				if Conjugate_CertainBSL:
					bls[i][x] = np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])
				elif Conjugate_CertainBSL2:
					bls[i][x] = np.sign(np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] - np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0]) * (antpos[i][x[0]] - antpos[i][x[1]])
				elif Conjugate_CertainBSL3:
					if np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) == 1:
						bls[i][x] = np.sign(np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] - np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0]) * (antpos[i][x[0]] - antpos[i][x[1]])
					elif np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) == -1:
						bls[i][x] = np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])
				else:
					bls[i][x] = (antpos[i][x[0]] - antpos[i][x[1]])
		# bls[i] = odict([(x, np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])) for x in data[i].keys()])
	# bls[1] = odict([(y, antpos_yy[y[0]] - antpos_yy[y[1]]) for y in data_yy.keys()])
	bls = np.array(bls)
	
	bsl_coord = [[], []]
	bsl_coord[0] = np.array([bls[0][index] for index in bls[0].keys()])
	bsl_coord[1] = np.array([bls[1][index] for index in bls[1].keys()])
	# bsl_coord_x=bsl_coord_y=bsl_coord
	bsl_coord = np.array(bsl_coord)
	
	Ubl_list_raw = [[], []]
	Ubl_list = [[], []]
	# ant_pos = [[], []]
	
	Nubl_raw = np.zeros(2, dtype=int)
	times_raw = np.zeros(2, dtype=int)
	redundancy_pro = [[], []]
	redundancy_pro_mfreq = [[], []]
	bsl_coord_dred = [[], []]
	bsl_coord_dred_mfreq = [[], []]
	vis_data_dred = [[], []]
	vis_data_dred_mfreq = [[], []]
	
	for i in range(2):
		Ubl_list_raw[i] = np.array(mmvs.arrayinfo.compute_reds_total(antloc[i], tol=tol))  ## Note that a new function has been added into omnical.arrayinfo as "compute_reds_total" which include all ubls not only redundant ones.
		try:
			print('Length of Ubl_list_raw[%s] with Badants: %s' %(i, len(Ubl_list_raw[i])))
		except:
			print('No Ubl_list_raw with Badants printing.')
		try:
			Ubl_list_raw[i] = mmvs.arrayinfo.filter_reds_total(Ubl_list_raw[i], ex_ants=map(lambda k:antpos[i].keys().index(k), Badants))
		except:
			print('Badants not in the ant-list.')
		try:
			print('Length of Ubl_list_raw[%s]: %s' %(i, len(Ubl_list_raw[i])))
		except:
			print('No Ubl_list_raw printing.')
	try:
		print('Bandants: %s'%str(Badants))
		print('Bandants Index: %s'%str(map(lambda k:antpos[i].keys().index(k), Badants)))
	except:
		print('Bandants, Index not printed.')
	# ant_pos[i] = antpos[i]
	
	for i in range(2):
		for i_ubl in range(len(Ubl_list_raw[i])):
			list_bsl = []
			for i_ubl_pair in range(len(Ubl_list_raw[i][i_ubl])):
				try:
					list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], '%s' % ['xx', 'yy'][i])))
				except:
					try:
						list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], '%s' % ['xx', 'yy'][1 - i])))
					except:
						try:
							list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], '%s' % ['xx', 'yy'][i])))
						except:
							try:
								list_bsl.append(bls[i].keys().index((antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], '%s' % ['xx', 'yy'][1 - i])))
							except:
								# print('Baseline:%s%s not in bls[%s]'%(antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][0]], antpos[i].keys()[Ubl_list_raw[i][i_ubl][i_ubl_pair][1]], i))
								continue
			
			if len(list_bsl) >= 1:
				Ubl_list[i].append(list_bsl)
			else:
				pass
	
	for i in range(2):
		Nubl_raw[i] = len(Ubl_list[i])
		times_raw[i] = len(data_times[i]) if data_times is not None else Ntimes
		bsl_coord_dred[i] = np.zeros((Nubl_raw[i], 3))
		bsl_coord_dred_mfreq[i] = np.zeros((Nubl_raw[i], 3))
		vis_data_dred[i] = np.zeros((times_raw[i], Nubl_raw[i]), dtype='complex128')
		vis_data_dred_mfreq[i] = np.zeros((len(data_freqs[i]), times_raw[i], Nubl_raw[i]), dtype='complex128') if data_freqs is not None else np.zeros((Nfreqs, times_raw[i], Nubl_raw[i]), dtype='complex128')
	
	if SingleFreq:
		dflags_dred = {}
		
		for i in range(2):
			dflags_dred[i] = LastUpdatedOrderedDict()
			pol = ['xx', 'yy'][i]
			
			for i_ubl in range(Nubl_raw[0]):
				vis_data_dred[i][:, i_ubl] = np.mean(vis_data[i].transpose()[Ubl_list[i][i_ubl]].transpose(), axis=1)
				bsl_coord_dred[i][i_ubl] = np.mean(bsl_coord[i][Ubl_list[i][i_ubl]], axis=0)
				dflags_dred[i][dflags_sf[i].keys()[Ubl_list[i][i_ubl][0]]] = np.mean(np.array([dflags_sf[i][dflags_sf[i].keys()[Ubl_list[i][i_ubl][k]]] for k in range(len(Ubl_list[i][i_ubl]))]), axis=0) == 1
				redundancy_pro[i].append(len(Ubl_list[i][i_ubl]))
	
	if MultiFreq:
		dflags_dred_mfreq = {}
		
		for i in range(2):
			dflags_dred_mfreq[i] = LastUpdatedOrderedDict()
			pol = ['xx', 'yy'][i]
			
			for i_ubl in range(Nubl_raw[0]):
				vis_data_dred_mfreq[i][:, :, i_ubl] = np.mean(vis_data_mfreq[i][:, :, Ubl_list[i][i_ubl]], axis=-1)
				bsl_coord_dred_mfreq[i][i_ubl] = np.mean(bsl_coord[i][Ubl_list[i][i_ubl]], axis=0)
				dflags_dred_mfreq[i][dflags[i].keys()[Ubl_list[i][i_ubl][0]]] = np.mean(np.array([dflags[i][dflags[i].keys()[Ubl_list[i][i_ubl][k]]] for k in range(len(Ubl_list[i][i_ubl]))]), axis=0) == 1
				redundancy_pro_mfreq[i].append(len(Ubl_list[i][i_ubl]))
	
	if SingleFreq and MultiFreq:
		if (la.norm(np.array(redundancy_pro[0]) - np.array(redundancy_pro_mfreq[0])) + la.norm(np.array(redundancy_pro[1]) - np.array(redundancy_pro_mfreq[1]))) != 0:
			raise ValueError('redundancy_pro doesnot match redundancy_pro_mfreq')
	elif MultiFreq:
		redundancy_pro = redundancy_pro_mfreq
	elif not SingleFreq:
		print ('No De-Redundancy Done.')
	
	if SingleFreq and MultiFreq:
		if (la.norm(bsl_coord_dred[0] - bsl_coord_dred_mfreq[0]) + la.norm(bsl_coord_dred[1] - bsl_coord_dred_mfreq[1])) != 0:
			raise ValueError('bsl_coord_dred doesnot match bsl_coord_dred_mfreq')
	elif MultiFreq:
		bsl_coord_dred = bsl_coord_dred_mfreq
	elif not SingleFreq:
		print ('No De-Redundancy Done.')
	
	if SingleFreq and MultiFreq:
		return vis_data_dred, vis_data_dred_mfreq, redundancy_pro, dflags_dred, dflags_dred_mfreq, bsl_coord_dred, Ubl_list
	elif MultiFreq:
		return vis_data_dred_mfreq, redundancy_pro, dflags_dred_mfreq, bsl_coord_dred, Ubl_list
	elif SingleFreq:
		return vis_data_dred, redundancy_pro, dflags_dred, bsl_coord_dred, Ubl_list
	else:
		return None


def Calculate_pointsource_visibility(ra, dec, d, freq, beam_healpix_hor=None, beam_heal_equ=None, nt=None, tlist=None, verbose=False):
	
	return vs.calculate_pointsource_visibility(ra, dec, d, freq, beam_healpix_hor=beam_healpix_hor, beam_heal_equ=beam_heal_equ, nt=nt, tlist=tlist, verbose=verbose)


def get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=True, Compute_beamweight=False, A_path='', A_RE_path='', A_got=None, A_version=1.0, AllSky=True, MaskedSky=False, Synthesize_MultiFreq=True, Flist_select_index=None, Flist_select=None, flist=None, Reference_Freq_Index=None, Reference_Freq=None,
                    equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=None, thresh=2., valid_pix_thresh = 1.e-4, Use_BeamWeight=False, Only_AbsData=False, Del_A=False, valid_npix=None,
                    beam_weight=None, ubls=None, C=299.792458, used_common_ubls=None, nUBL_used=None, nUBL_used_mfreq=None, nt_used=None, nside_standard=None, nside_start=None, nside_beamweight=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=None, Parallel_A=False):
	
	print('flist: %s' % str(flist))
	
	if Synthesize_MultiFreq:
		if flist is None:
			raise ValueError('No flist provided.')
		if Flist_select_index is None and Flist_select is not None:
			Flist_select_index = {}
			try:
				for i in range(2):
					Flist_select_index[i] = np.zeros_like(Flist_select[i], dtype='int')
					for k in range(len(Flist_select[i])):
						Flist_select_index[i][k] = np.abs(Flist_select[i][k] - flist[i]).argmin()
			except:
				raise ValueError('Flist_select cannot come from flist.')
		elif Flist_select_index is not None:
			for i in range(2):
				Flist_select[i] = flist[i][Flist_select_index[i]]
		else:
			raise ValueError('No Flist_select or Flist_select_index provided.')
		
		# if len(Flist_select) != 2:
		# 	raise ValueError('Please Specify Flist_select for each polarization.')
		
		if Reference_Freq_Index is None:
			Reference_Freq_Index = [[], []]
			for i in range(2):
				try:
					Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
				except:
					Reference_Freq_Index[i] = len(flist[i]) / 2
		Reference_Freq = [[], []]
		for i in range(2):
			Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
		print ('Reference_Freq_Index: x-%s; y-%s' % (Reference_Freq_Index[0], Reference_Freq_Index[1]))
		print ('Reference_Freq: x-%s; y-%s' % (Reference_Freq[0], Reference_Freq[1]))
		print ('Flist_select_index: %s' % (str(Flist_select_index)))
		print ('Flist_select: %s' % (str(Flist_select)))
	
	else:
		# if Flist_select is None:
		try:
			if flist is None:
				Flist_select = [[Reference_Freq[0]], [Reference_Freq[1]]]
			
			else:
				if Reference_Freq_Index is not None:
					for i in range(2):
						try:
							Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
							Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
						except:
							Reference_Freq_Index[i] = len(flist[i]) / 2
							Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
				else:
					Reference_Freq_Index = [[], []]
					for i in range(2):
						try:
							Reference_Freq_Index[i] = np.abs(Reference_Freq[i] - flist[i]).argmin()
							Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
						except:
							Reference_Freq_Index[i] = len(flist[i]) / 2
							Reference_Freq[i] = flist[i][Reference_Freq_Index[i]]
				Flist_select = [[Reference_Freq[0]], [Reference_Freq[1]]]
				Flist_select_index = {}
				try:
					for i in range(2):
						Flist_select_index[i] = np.zeros_like(Flist_select[i], dtype='int')
						for k in range(len(Flist_select[i])):
							Flist_select_index[i][k] = np.abs(Flist_select[i][k] - flist[i]).argmin()
				except:
					raise ValueError('Flist_select cannot come from flist.')
				print ('Reference_Freq_Index: x-%s; y-%s' % (Reference_Freq_Index[0], Reference_Freq_Index[1]))
		except:
			raise ValueError('Please specify Reference_Freq for each polarization. ')
		
		print ('Reference_Freq: x-%s; y-%s' % (Reference_Freq[0], Reference_Freq[1]))
		if flist is not None:
			print ('Flist_select_index: %s' % (str(Flist_select_index)))
		print ('Flist_select: %s' % (str(Flist_select)))
	
	if len(Flist_select[0]) != len(Flist_select[1]):
		raise ValueError('Lengths of Flist_select for two pols are different.')
	
	if nUBL_used is not None and nUBL_used is not None:
		if nUBL_used != len(used_common_ubls):
			raise ValueError('len(used_common_ubls)%s != nUBL_used%s'%(len(used_common_ubls), nUBL_used))
			
	if nUBL_used_mfreq is not None:
		if nUBL_used_mfreq != len(used_common_ubls) * len(Flist_select[0]):
			print('len(used_common_ubls) * len(Flist_select[0])%s != nUBL_used_mfreq%s' % (len(used_common_ubls) * len(Flist_select[0]), nUBL_used_mfreq))
	
	if lsts is not None and nt_used is not None:
		if len(lsts) != nt_used:
			raise ValueError('number of lsts%s doesnot match nt_used%s.' % (len(lsts), nt_used))
		
	nUBL_used = len(used_common_ubls)
	nUBL_used_mfreq = len(used_common_ubls) * len(Flist_select[0])
	print('nUBL_used: %s\nnUBL_used_mfreq: %s'%(nUBL_used, nUBL_used_mfreq))
	
	if AllSky:
		
		A_version = A_version
		A = {}
		if equatorial_GSM_standard is None and Synthesize_MultiFreq:
			equatorial_GSM_standard = equatorial_GSM_standard_mfreq[Reference_Freq_Index[0]]  # choose x freq.
		
		if Parallel_A:
			timer = time.time()
			pool = Pool()
			if not Synthesize_MultiFreq:
				if beam_heal_equ_x is None:
					try:
						beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
					except:
						raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
				
				if beam_heal_equ_y is None:
					try:
						beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
					except:
						raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
				beam_heal_equ = {0: beam_heal_equ_x, 1: beam_heal_equ_y}
				
				A_multiprocess_list = np.array([[[pool.apply_async(Calculate_pointsource_visibility, args=(hpf.pix2ang(nside_beamweight, n)[1], (PI / 2. - hpf.pix2ang(nside_beamweight, n)[0]), used_common_ubls, f, None, beam_heal_equ[id_p], None, lsts)) for n in range(12 * nside_beamweight ** 2)] for f in Flist_select[id_p]] for id_p in range(2)])
				A_list = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2. for n in range(12 * nside_beamweight ** 2)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((0, 1, 3, 4, 2))
			else:
				beam_heal_equ = {0: beam_heal_equ_x_mfreq, 1: beam_heal_equ_y_mfreq}
				
				A_multiprocess_list = np.array([[[pool.apply_async(Calculate_pointsource_visibility, args=(hpf.pix2ang(nside_beamweight, n)[1], (PI / 2. - hpf.pix2ang(nside_beamweight, n)[0]), used_common_ubls, f, None, beam_heal_equ[id_p][Flist_select_index[id_p][id_f]], None, lsts)) for n in range(12 * nside_beamweight ** 2)] for id_f, f in enumerate(Flist_select[id_p])] for id_p in range(2)])
				A_list = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2. * (equatorial_GSM_standard_mfreq[Flist_select_index[id_p][id_f], n] / equatorial_GSM_standard[n]) for n in range(12 * nside_beamweight ** 2)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((0, 1, 3, 4, 2))
			print('Shape of A_list: %s'%(str(A_list.shape)))
			
			A = {}
			for id_p, p in enumerate(['x', 'y']):
				A[p] = A_list[id_p].reshape(len(Flist_select[id_p]) * len(used_common_ubls) * nt_used , 12 * nside_beamweight ** 2)
				print('Shape of A[%s]: %s' % (p, str(A[p].shape)))
			del(A_multiprocess_list)
			del(A_list)
			print('%s minutes used for parallel_computing A' % ((float(time.time() - timer) / 60.)))
			
		else:
			for id_p, p in enumerate(['x', 'y']):
				pol = p + p
				try:
					print "%i UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(ubls[p]), np.max(np.linalg.norm(ubls[p], axis=1)) / (C / Reference_Freq[id_p]), pol)
					print "%i Used-Common-UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(used_common_ubls), np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / Reference_Freq[id_p]), pol)
				except:
					try:
						print "%i Used-Common-UBLs to include, longest baseline is %i wavelengths for Pol: %s" % (len(used_common_ubls), np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / Reference_Freq[id_p]), pol)
					except:
						pass
					
				A[p] = np.zeros((len(Flist_select[id_p]), len(used_common_ubls), nt_used , 12 * nside_beamweight ** 2), dtype='complex128')
				
				for id_f, f in enumerate(Flist_select[id_p]):
					if not Synthesize_MultiFreq:
						if beam_heal_equ_x is None:
							try:
								beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
							except:
								raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
						
						if beam_heal_equ_y is None:
							try:
								beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
							except:
								raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
						
						if p == 'x':
							beam_heal_equ = beam_heal_equ_x
						elif p == 'y':
							beam_heal_equ = beam_heal_equ_y
					else:
						if p == 'x':
							beam_heal_equ = beam_heal_equ_x_mfreq[Flist_select_index[id_p][id_f]]
						elif p == 'y':
							beam_heal_equ = beam_heal_equ_y_mfreq[Flist_select_index[id_p][id_f]]
					
					# beam
					
					print "Computing sky weighting A matrix for pol: %s, for freq: %s" % (p, f)
					sys.stdout.flush()
					
					# A[p] = np.zeros((nt_used * len(used_common_ubls), 12 * nside_beamweight ** 2), dtype='complex128')
					
					timer = time.time()
					for i in np.arange(12 * nside_beamweight ** 2):
						dec, ra = hpf.pix2ang(nside_beamweight, i)  # gives theta phi
						dec = PI / 2 - dec
						print "\r%.1f%% completed" % (100. * float(i) / (12. * nside_beamweight ** 2)),
						sys.stdout.flush()
						if abs(dec - lat_degree * PI / 180) <= PI / 2:
							if Synthesize_MultiFreq:
								A[p][id_f, :, :, i] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts)) * (equatorial_GSM_standard_mfreq[Flist_select_index[id_p][id_f], i] / equatorial_GSM_standard[i])
							else:
								A[p][id_f, :, :, i] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts))
					
					print "%f minutes used for pol: %s, freq: %s" % ((float(time.time() - timer) / 60.), pol, f)
					sys.stdout.flush()
				# A[p] = A[p].reshape(len(Flist_select[id_p]) * len(used_common_ubls),  nt_used , 12 * nside_beamweight ** 2)
				# print('Shape of A[%s]: %s' % (p, str(A[p].shape)))
				A[p] = A[p].reshape(len(Flist_select[id_p]) * len(used_common_ubls) * nt_used, 12 * nside_beamweight ** 2)
				print('Shape of A[%s]: %s' % (p, str(A[p].shape)))
		
		if Compute_beamweight:
			print "Computing beam weight...",
			sys.stdout.flush()
			beam_weight = ((la.norm(A['x'], axis=0) ** 2 + la.norm(A['y'], axis=0) ** 2) ** .5)[hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
			beam_weight = beam_weight / np.mean(beam_weight)
			thetas_standard, phis_standard = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
			beam_weight = hpf.get_interp_val(beam_weight, thetas_standard, phis_standard, nest=True)  # np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
			print "done."
			sys.stdout.flush()
			return A, beam_weight
		else:
			return A
	
	elif MaskedSky:
		if os.path.isfile(A_RE_path) and not force_recompute:
			print "Reading A matrix from %s" % A_path
			sys.stdout.flush()
			if fit_for_additive:
				A = np.fromfile(A_path, dtype='complex128').reshape((2 * nUBL_used * len(Flist_select[0]), 2, valid_npix + nt_used, 4 * nUBL_used * len(Flist_select[0])))
			else:
				A = np.fromfile(A_path, dtype='complex128').reshape((2 * nUBL_used * len(Flist_select[0]), 2, nt_used, valid_npix))
			return A
		elif os.path.isfile(A_path) and not force_recompute:
			print "Reading A matrix from %s" % A_path
			sys.stdout.flush()
			if fit_for_additive:
				A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used * len(Flist_select[0]), 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0])))
				A.shape = (len(Flist_select[0]) * nUBL_used * 2 * nt_used, A.shape[-1])
			else:
				A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used * len(Flist_select[0]), 2, nt_used, valid_npix))
				A.shape = (len(Flist_select[0]) * nUBL_used * 2 * nt_used, A.shape[-1])
			if not fit_for_additive:
				A = A[:, :valid_npix]
			else:
				A[:, valid_npix:] = additive_A[:, 1:]
			try:
				print('>>>>>>>>>>>>>>>>> Shape of A after fit_for_additive: %s' % (str(A.shape)))
			# print('>>>>>>>>>>>>>>>>> Shape of A after Real/Imag Seperation: %s' % (str(np.concatenate((np.real(A), np.imag(A))).shape)))
			except:
				print('No printing A.')
			
			# Merge A
			try:
				if not Only_AbsData:
					return np.concatenate((np.real(A), np.imag(A))).astype('float64')
				else:
					return A

			except MemoryError:
				print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
				sys.stdout.flush()
				Ashape = list(A.shape)
				Ashape[0] = Ashape[0] * 2
				np.real(A).tofile(A_path + 'tmpre')
				np.imag(A).tofile(A_path + 'tmpim')
				del (A)
				os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))
				
				os.system("rm %s" % (A_path + 'tmpim'))
				A = np.fromfile(A_path + 'tmpre', dtype='float64').reshape(Ashape)
				os.system("rm %s" % (A_path + 'tmpre'))
				print "done."
				sys.stdout.flush()
				return A.astype('float64')

			# return A
		else:
			
			if equatorial_GSM_standard is None:
				try:
					equatorial_GSM_standard = equatorial_GSM_standard_mfreq[Reference_Freq_Index[0]]  # choose x freq.
				except:
					print('No equatorial_GSM_standard calculated.')
			if beam_weight is None:
				if A_got is None:
					A_got = get_A_multifreq(additive_A=None, A_path=None, A_got=None, A_version=1.0, AllSky=True, MaskedSky=False, Synthesize_MultiFreq=False, flist=flist, Flist_select=None, Flist_select_index=None, Reference_Freq_Index=Reference_Freq_Index, Reference_Freq=Reference_Freq, equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
					                        used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=None, nside_start=None, nside_beamweight=nside_beamweight, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
				print "Computing beam weight...",
				sys.stdout.flush()
				beam_weight = ((la.norm(A_got['x'], axis=0) ** 2 + la.norm(A_got['y'], axis=0) ** 2) ** .5)[hpf.nest2ring(nside_beamweight, range(12 * nside_beamweight ** 2))]
				beam_weight = beam_weight / np.mean(beam_weight)
				thetas_standard, phis_standard = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
				beam_weight = hpf.get_interp_val(beam_weight, thetas_standard, phis_standard, nest=True)  # np.array([beam_weight for i in range(nside_standard ** 2 / nside_beamweight ** 2)]).transpose().flatten()
				try:
					del (A_got)
					print('A_got has been successfully deleted.')
				except:
					print('No A_got to be deleted.')
				print "done."
				sys.stdout.flush()
			
			gsm_beamweighted = equatorial_GSM_standard * beam_weight
			
			
			nside_distribution = np.zeros(12 * nside_standard ** 2)
			final_index = np.zeros(12 * nside_standard ** 2, dtype=int)
			thetas, phis, sizes = [], [], []
			abs_thresh = np.mean(gsm_beamweighted) * thresh
			pixelize(gsm_beamweighted, nside_distribution, nside_standard, nside_start, abs_thresh,
			         final_index, thetas, phis, sizes)
			npix = len(thetas)
			if Use_BeamWeight:
				valid_pix_mask = hpf.get_interp_val(beam_weight, thetas, phis, nest=True) > valid_pix_thresh * max(beam_weight)
			else:
				valid_pix_mask = hpf.get_interp_val(gsm_beamweighted, thetas, phis, nest=True) > valid_pix_thresh * max(gsm_beamweighted)
			valid_npix = np.sum(valid_pix_mask)
			print '>>>>>>VALID NPIX =', valid_npix
			
			fake_solution_map = np.zeros_like(thetas)
			for i in range(len(fake_solution_map)):
				fake_solution_map[i] = np.sum(equatorial_GSM_standard[final_index == i])
			fake_solution_map = fake_solution_map[valid_pix_mask]
			
			if Synthesize_MultiFreq:
				fake_solution_map_mfreq_temp = np.zeros((len(Flist_select[0]), npix))
				fake_solution_map_mfreq = np.zeros((len(Flist_select[0]), valid_npix))
				for id_f, f in enumerate(Flist_select_index[0]):
					for i in range(npix):
						fake_solution_map_mfreq_temp[id_f, i] = np.sum(equatorial_GSM_standard_mfreq[f, final_index == i])
					fake_solution_map_mfreq[id_f] = fake_solution_map_mfreq_temp[id_f, valid_pix_mask]
			
			try:
				del(equatorial_GSM_standard)
				# del(beam_weight)
				print('equatorial_GSM_standard and beam_weight have been successfully deleted.')
			except:
				print('No equatorial_GSM_standard or beam_weight to be deleted.')
				
			try:
				del(equatorial_GSM_standard_mfreq)
				del(fake_solution_map_mfreq_temp)
				print('equatorial_GSM_standard_mfreq and fake_solution_map_mfreq_temp have been successfully deleted.')
			except:
				print('No equatorial_GSM_standard_mfreq or fake_solution_map_mfreq_temp to be deleted.')
				
			sizes = np.array(sizes)[valid_pix_mask]
			thetas = np.array(thetas)[valid_pix_mask]
			phis = np.array(phis)[valid_pix_mask]
			try:
				np.savez(pixel_directory + '/../Output/' + 'pixel_scheme_%i_%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.npz' % (valid_npix, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none',
				                                                                                                                      bnside, nside_standard), gsm=fake_solution_map, thetas=thetas, phis=phis, sizes=sizes, nside_distribution=nside_distribution, final_index=final_index,
				         n_fullsky_pix=npix, valid_pix_mask=valid_pix_mask, thresh=thresh)
			except:
				print('Not Saving to pixel_directory.')
			
			if not fit_for_additive:
				fake_solution = np.copy(fake_solution_map)
			else:
				fake_solution = np.concatenate((fake_solution_map, np.zeros(4 * nUBL_used)))
			
			if not Compute_A:
				return beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution
			
			# if os.path.isfile(A_path) and not force_recompute:
			# 	print "Reading A matrix from %s" % A_path
			# 	sys.stdout.flush()
			# 	A = np.fromfile(A_path, dtype='complex128').reshape((nUBL_used * len(Flist_select[0]), 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0])))
			# 	try:
			# 		del(beam_heal_equ_x_mfreq)
			# 		del(beam_heal_equ_y_mfreq)
			# 		del(equatorial_GSM_standard_mfreq)
			# 	except:
			# 		pass
			# else:
			
			if Parallel_A:
				print('Parallel Computing A matrix...')
			else:
				print ("Computing A matrix...")
			sys.stdout.flush()
			if fit_for_additive:
				A = np.empty((len(Flist_select[0]), nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0])), dtype='complex128')
			else:
				A = np.empty((len(Flist_select[0]), nUBL_used, 2, nt_used, valid_npix), dtype='complex128')
			timer = time.time()
			if Parallel_A:
				pool = Pool()
				if not Synthesize_MultiFreq:
					if beam_heal_equ_x is None:
						try:
							beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
						except:
							raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
					
					if beam_heal_equ_y is None:
						try:
							beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
						except:
							raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
					beam_heal_equ = {0: beam_heal_equ_x, 1: beam_heal_equ_y}
					
					A_multiprocess_list = np.array([[[pool.apply_async(Calculate_pointsource_visibility, args=(phis[n], (PI / 2. - thetas[n]), used_common_ubls, f, None, beam_heal_equ[id_p], None, lsts)) for n in range(valid_npix)] for f in Flist_select[id_p]] for id_p in range(2)])
					A[:, :, :, :, :valid_npix] = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2. for n in range(valid_npix)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((1, 3, 0, 4, 2))
				else:
					beam_heal_equ = {0: beam_heal_equ_x_mfreq, 1: beam_heal_equ_y_mfreq}
					
					A_multiprocess_list = np.array([[[pool.apply_async(Calculate_pointsource_visibility, args=(phis[n], (PI / 2. - thetas[n]), used_common_ubls, f, None, beam_heal_equ[id_p][Flist_select_index[id_p][id_f]], None, lsts)) for n in range(valid_npix)] for id_f, f in enumerate(Flist_select[id_p])] for id_p in range(2)])
					A[:, :, :, :, :valid_npix] = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2. * (fake_solution_map_mfreq[id_f, n] / fake_solution_map[n]) for n in range(valid_npix)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((1, 3, 0, 4, 2))
				print('A shape: %s' %str(A.shape))
				print('%s minutes used for parallel_computing A' % ((float(time.time() - timer) / 60.)))
				
				del(A_multiprocess_list)
				
				if fit_for_additive:
					A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0]))
				else:
					A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix)
				print('>>>>>>>>>>>>>>>>> Shape of A: %s' % (str(A.shape)))
				# if Del_A:
				# 	try:
				# 		A.tofile(A_path)
				# 	except:
				# 		print('A not saved.')
			else:
				for id_p, p in enumerate(['x', 'y']):
					for id_f, f in enumerate(Flist_select[id_p]):
						if not Synthesize_MultiFreq:
							if beam_heal_equ_x is None:
								try:
									beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
								except:
									raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
							
							if beam_heal_equ_y is None:
								try:
									beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
								except:
									raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
							
							if p == 'x':
								beam_heal_equ = beam_heal_equ_x
							elif p == 'y':
								beam_heal_equ = beam_heal_equ_y
						else:
							if p == 'x':
								beam_heal_equ = beam_heal_equ_x_mfreq[Flist_select_index[id_p][id_f]]
							elif p == 'y':
								beam_heal_equ = beam_heal_equ_y_mfreq[Flist_select_index[id_p][id_f]]
						
						for n in range(valid_npix):
							ra = phis[n]
							dec = PI / 2. - thetas[n]
							print "\r%f%% completed, %f minutes left for %s-%s" % (
								100. * float(n) / (valid_npix), float(valid_npix - n) / (n + 1) * (float(time.time() - timer) / 60.), id_f, f),
							sys.stdout.flush()
							if Synthesize_MultiFreq:
								A[id_f, :, id_p, :, n] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2) * (fake_solution_map_mfreq[id_f, n] / fake_solution_map[n])  # xx and yy are each half of I
							else:
								A[id_f, :, id_p, :, n] = (vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2)  # xx and yy are each half of I
						# A[:, -1, :, n] = vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ_y, tlist=lsts) / 2
						
						print ("%f minutes used for pol: %s, freq: %s" % ((float(time.time() - timer) / 60.), p, f))
					print('Shape of A[%s]: %s' % (p, str(A[:, :, id_p, :, :].shape)))
					sys.stdout.flush()
				
				if fit_for_additive:
					A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0]))
				else:
					A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix)
				print('>>>>>>>>>>>>>>>>> Shape of A: %s' % (str(A.shape)))

			if Del_A:
				try:
					A.tofile(A_path)
				except:
					print('A not saved.')
		
			# vs.calculate_pointsource_visibility(self, ra, dec, d, freq, beam_healpix_hor=None, beam_heal_equ=None, nt=None, tlist=None, verbose=False)
			# if Parallel_A:
			# 	pool = Pool()
			# 	if not Synthesize_MultiFreq:
			# 		if beam_heal_equ_x is None:
			# 			try:
			# 				beam_heal_equ_x = beam_heal_equ_x_mfreq[Reference_Freq_Index[0]]
			# 			except:
			# 				raise ValueError('No beam_heal_equ_x can be loaded or calculated from mfreq version.')
			#
			# 		if beam_heal_equ_y is None:
			# 			try:
			# 				beam_heal_equ_y = beam_heal_equ_x_mfreq[Reference_Freq_Index[1]]
			# 			except:
			# 				raise ValueError('No beam_heal_equ_y can be loaded or calculated from mfreq version.')
			# 		beam_heal_equ = {0: beam_heal_equ_x, 1:beam_heal_equ_y}
			#
			# 		A_multiprocess_list = [[[pool.apply_async(vs.calculate_pointsource_visibility, args=(phis[n], (PI/2.-thetas[n]), used_common_ubls, f, None, beam_heal_equ[id_p], None, lsts)) for n in range(valid_npix)] for f in Flist_select[id_p]] for id_p in range(2)]
			# 		A = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2. for n in range(valid_npix)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((0, 3, 1, 4, 2))
			# 	else:
			# 		beam_heal_equ = {0: beam_heal_equ_x_mfreq, 1: beam_heal_equ_y_mfreq}
			#
			# 		A_multiprocess_list = [[[pool.apply_async(vs.calculate_pointsource_visibility, args=(phis[n], (PI / 2. - thetas[n]), used_common_ubls, f, None, beam_heal_equ[id_p][Flist_select_index[id_p][id_f]], None, lsts)) for n in range(valid_npix)] for id_f,f in enumerate(Flist_select[id_p])] for id_p in range(2)]
			# 		A = np.array([[[A_multiprocess_list[id_p][id_f][n].get() / 2.  * (fake_solution_map_mfreq[id_f, n] / fake_solution_map[n]) for n in range(valid_npix)] for id_f in range(len(Flist_select[id_p]))] for id_p in range(2)]).transpose((1, 3, 0, 4, 2))
			# 	print('%s minutes used for parallel_computing A'%((float(time.time() - timer) / 60.)))
			#
			# 	A = A.reshape(len(Flist_select[0]) * nUBL_used, 2, nt_used, valid_npix + 4 * nUBL_used * len(Flist_select[0]))
			# 	print('>>>>>>>>>>>>>>>>> Shape of A: %s' % (str(A.shape)))
			# 	try:
			# 		A.tofile(A_path)
			# 	except:
			# 		print('A not saved.')
			
			# #put in autocorr regardless of whats saved on disk
			# for i in range(nUBL_used):
			#     for p in range(2):
			#         A[i, p, :, valid_npix + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]
			#         A[i, p, :, valid_npix + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]
		
			A.shape = (len(Flist_select[0]) * nUBL_used * 2 * nt_used, A.shape[-1])
			if not fit_for_additive:
				A = A[:, :valid_npix]
			else:
				A[:, valid_npix:] = additive_A[:, 1:]
			try:
				print('>>>>>>>>>>>>>>>>> Shape of A after fit_for_additive: %s' % (str(A.shape)))
				# print('>>>>>>>>>>>>>>>>> Shape of A after Real/Imag Seperation: %s' % (str(np.concatenate((np.real(A), np.imag(A))).shape)))
			except:
				print('No printing A.')
			
			# Merge A
			try:
				if Synthesize_MultiFreq:
					if not Only_AbsData:
						return np.concatenate((np.real(A), np.imag(A))).astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution #, fake_solution_map_mfreq
					else:
						return A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution  # , fake_solution_map_mfreq
				else:
					if not Only_AbsData:
						return np.concatenate((np.real(A), np.imag(A))).astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution
					else:
						return A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution
			except MemoryError:
				print "Not enough memory, concatenating A on disk ", A_path + 'tmpre', A_path + 'tmpim',
				sys.stdout.flush()
				Ashape = list(A.shape)
				Ashape[0] = Ashape[0] * 2
				np.real(A).tofile(A_path + 'tmpre')
				np.imag(A).tofile(A_path + 'tmpim')
				del (A)
				os.system("cat %s >> %s" % (A_path + 'tmpim', A_path + 'tmpre'))
				
				os.system("rm %s" % (A_path + 'tmpim'))
				A = np.fromfile(A_path + 'tmpre', dtype='float64').reshape(Ashape)
				os.system("rm %s" % (A_path + 'tmpre'))
				print "done."
				sys.stdout.flush()
				if Synthesize_MultiFreq:
					return A.astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution#, fake_solution_map_mfreq
				else:
					return A.astype('float64'), beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution


		# return A, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map


def Simulate_Visibility_mfreq(script_dir='', INSTRUMENT='', full_sim_filename_mfreq='', sim_vis_xx_filename_mfreq='', sim_vis_yy_filename_mfreq='', Force_Compute_Vis=True, Get_beam_GSM=False, Force_Compute_beam_GSM=False, Multi_freq=False, Multi_Sin_freq=False, Fake_Multi_freq=False, crosstalk_type='',
                              flist=None, freq_index=None, freq=None, equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None,
                              beam_weight=None, C=299.792458, used_common_ubls=None, nUBL_used=None, nUBL_used_mfreq=None, nt_used=None, nside_standard=None, nside_start=None, nside_beamweight=None,
                              beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=None, tlist=None, tmask=True, Time_Expansion_Factor=1., Parallel_Mulfreq_Visibility = False, Parallel_Mulfreq_Visibility_deep=False):
	if Force_Compute_beam_GSM or Get_beam_GSM:
		beam_heal_hor_x_mfreq = np.array([local_beam_unpol(flist[0][i])[0] for i in range(len(flist[0]))])
		beam_heal_hor_y_mfreq = np.array([local_beam_unpol(flist[1][i])[1] for i in range(len(flist[1]))])
		beam_heal_equ_x_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_x_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(len(flist[0]))])
		beam_heal_equ_y_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_y_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(len(flist[1]))])
		
		pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
		pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
		pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
		components = np.loadtxt(script_dir + '/../data/components.dat')
		scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
		w1 = si.interp1d(components[:, 0], components[:, 2])
		w2 = si.interp1d(components[:, 0], components[:, 3])
		w3 = si.interp1d(components[:, 0], components[:, 4])
		gsm_standard = {}
		for i in range(2):
			gsm_standard[i] = np.exp(scale_loglog(np.log(freq[i]))) * (w1(freq[i]) * pca1 + w2(freq[i]) * pca2 + w3(freq[i]) * pca3)
		if Multi_freq:
			gsm_standard_mfreq = {}
			for p in range(2):
				gsm_standard_mfreq[p] = np.array([np.exp(scale_loglog(np.log(flist[p][i]))) * (w1(flist[p][i]) * pca1 + w2(flist[p][i]) * pca2 + w3(flist[p][i]) * pca3) for i in range(len(flist[p]))])
		
		# rotate sky map and converts to nest
		equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
		print "Rotating GSM_standard and converts to nest...",
		
		if INSTRUMENT == 'miteor':
			DecimalYear = 2013.58  # 2013, 7, 31, 16, 47, 59, 999998)
			JulianEpoch = 2013.58
		elif 'hera47' in INSTRUMENT:
			DecimalYear = Time(tlist_JD[0], format='jd').decimalyear + (np.mean(Time(tlist_JD, format='jd').decimalyear) - Time(tlist_JD[0], format='jd').decimalyear) * Time_Expansion_Factor
			JulianEpoch = Time(tlist_JD[0], format='jd').jyear + (np.mean(Time(tlist_JD, format='jd').jyear) - Time(tlist_JD[0], format='jd').jyear) * Time_Expansion_Factor  # np.mean(Time(data_times[0], format='jd').jyear)
		print('JulianEpoch: %s' % (str(JulianEpoch)))
		
		sys.stdout.flush()
		equ_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=JulianEpoch))
		ang0, ang1 = hp.rotator.rotateDirection(equ_to_gal_matrix,
		                                        hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
		equatorial_GSM_standard = {}
		for i in range(2):
			equatorial_GSM_standard[i] = hpf.get_interp_val(gsm_standard[i], ang0, ang1)
		equatorial_GSM_standard_xx = equatorial_GSM_standard[0]
		equatorial_GSM_standard_yy = equatorial_GSM_standard[1]
		del (equatorial_GSM_standard)
		if Multi_freq:
			equatorial_GSM_standard_mfreq = {}
			for p in range(2):
				equatorial_GSM_standard_mfreq[p] = np.array([hpf.get_interp_val(gsm_standard_mfreq[p][i], ang0, ang1) for i in range(len(flist[p]))])
			equatorial_GSM_standard_mfreq_xx = equatorial_GSM_standard_mfreq[0]
			equatorial_GSM_standard_mfreq_yy = equatorial_GSM_standard_mfreq[1]
			del (equatorial_GSM_standard_mfreq)
		
		print "done."
	
	# if Get_beam_GSM:
	# 	return
	
	if not Multi_freq and not Fake_Multi_freq:
		if flist is None and freq_index is None and freq is None:
			raise valueerror('no frequency can be specified.')
		elif freq is not None:
			flist = [[freq[0]], [freq[1]]]
			if flist is not None:
				freq_index = {}
				freq_index[0] = np.abs(freq[0] - flist[0]).argmin()
				freq_index[1] = np.abs(freq[1] - flist[1]).argmin()
		elif flist is not None and freq_index is not None:
			flist = [[flist[0][freq_index[0]]], [flist[1][freq_index[1]]]]
		elif flist is not None and freq_index is None:
			flist = [[flist[0][len(flist[0]) / 2]], [flist[1][len(flist[1]) / 2]]]
			freq_index = [len(flist[0]) / 2, len(flist[0]) / 2]
			print ('choose the middle of flist for each pol as default since none has been specified.')
	
	else:
		if flist is None:
			if Fake_Multi_freq and (nf_used is None or freq_index is None):
				raise ValueError('Cannot do fake-mfreq simulation without flist provided.')
			elif Fake_Multi_freq:
				flist = np.ones((2, nf_used))
			else:
				raise ValueError('Cannot do mfreq simulation without flist provided.')
		elif Multi_Sin_freq or Fake_Multi_freq:
			if freq_index is not None or freq is not None:
				if freq_index is None:
					freq_index = {}
					for i in range(2):
						freq_index[i] = np.abs(freq[i] - flist[i]).argmin()
				elif freq is not None:
					for i in range(2):
						if freq_index[i] != np.abs(freq[i] - flist[i]).argmin():
							print('freq not match freq_index from flist, use freq_index from flist for pol-%s.' % ['xx', 'yy'][i])
			for i in range(2):
				freq[i] = flist[i][freq_index[i]]
				print('Sinfreq from multifreq: %s-%s' % (freq_index[i], freq[i]))
	
	if len(flist[0]) != len(flist[1]):
		raise ValueError('Two pol nf_used not same: %s != %s' % (len(flist[0]), len(flist[1])))
	nf_used = len(flist[0])
	
	try:
		print('flist: %s MHz;\nnf_used: %s' % (str(flist), nf_used))
	except:
		raise ValueError('No flist information successfully processed and printed.')
	
	if used_common_ubls is not None and nUBL_used is not None:
		if len(used_common_ubls) != nUBL_used:
			raise ValueError('number of used_common_ubls%s doesnot match nUBL_used%s.' % (len(used_common_ubls), nUBL_used))
	nUBL_used = len(used_common_ubls)
	
	if lsts is not None and nt_used is not None:
		if len(lsts) != nt_used:
			raise ValueError('number of lsts%s doesnot match nt_used%s.' % (len(lsts), nt_used))
	nt_used = len(lsts)
	
	if not Multi_freq or Fake_Multi_freq:
		if beam_heal_equ_x is None and beam_heal_equ_x_mfreq is not None:
			beam_heal_equ_x = beam_heal_equ_x_mfreq[freq_index[0]]
		elif beam_heal_equ_x is None and beam_heal_equ_x_mfreq is None:
			raise ValueError('No x beam data.')
		if beam_heal_equ_y is None and beam_heal_equ_y_mfreq is not None:
			beam_heal_equ_y = beam_heal_equ_y_mfreq[freq_index[1]]
		elif beam_heal_equ_y is None and beam_heal_equ_y_mfreq is None:
			raise ValueError('No y beam data either from sinfreq or multifreq.')
		beam_heal_equ_x_mfreq = [beam_heal_equ_x]
		beam_heal_equ_y_mfreq = [beam_heal_equ_y]
	
	else:
		if beam_heal_equ_x_mfreq is None or beam_heal_equ_y_mfreq is None:
			raise ValueError('No multifreq beam data.')
	
	if not Multi_freq or Fake_Multi_freq:
		if equatorial_GSM_standard_xx is None and equatorial_GSM_standard_mfreq_xx is not None:
			equatorial_GSM_standard_xx = equatorial_GSM_standard_mfreq_xx[freq_index[0]]
		elif equatorial_GSM_standard_xx is None and equatorial_GSM_standard_mfreq_xx is None:
			raise ValueError('No equatorial_GSM_standard_xx data.')
		if equatorial_GSM_standard_yy is None and equatorial_GSM_standard_mfreq_yy is not None:
			equatorial_GSM_standard_yy = equatorial_GSM_standard_mfreq_yy[freq_index[1]]
		elif equatorial_GSM_standard_yy is None and equatorial_GSM_standard_mfreq_yy is None:
			raise ValueError('No equatorial_GSM_standard data.')
		
		equatorial_GSM_standard_mfreq = np.array([[equatorial_GSM_standard_xx], [equatorial_GSM_standard_yy]])
	
	else:
		if beam_heal_equ_x_mfreq is None or beam_heal_equ_y_mfreq is None:
			raise ValueError('No multifreq beam data.')
		equatorial_GSM_standard_mfreq = np.array([equatorial_GSM_standard_mfreq_xx, equatorial_GSM_standard_mfreq_yy])
	
	if os.path.isfile(full_sim_filename_mfreq) and not Force_Compute_Vis:
		fullsim_vis_mfreq = np.fromfile(full_sim_filename_mfreq, dtype='complex128').reshape((2, nUBL_used + 1, nt_used, nf_used))
		fullsim_vis_mfreq[0][:-1].astype('complex128').tofile(sim_vis_xx_filename_mfreq)
		fullsim_vis_mfreq[1][:-1].astype('complex128').tofile(sim_vis_yy_filename_mfreq)
	
	else:
		if Parallel_Mulfreq_Visibility or Parallel_Mulfreq_Visibility_deep:
			pool = Pool()
		if not Parallel_Mulfreq_Visibility_deep:
			fullsim_vis_mfreq = np.zeros((2, nUBL_used + 1, nt_used, nf_used), dtype='complex128')  # since its going to accumulate along the pixels it needs to start with complex128. significant error if start with complex64
		if Fake_Multi_freq:
			print('>>>>Freq_index selected not fake before: %s' % (str(freq_index)))
			freq_index_fakemfreq = copy.deepcopy(freq_index)
			fullsim_vis, autocorr_vis = Simulate_Visibility_mfreq(full_sim_filename_mfreq='', sim_vis_xx_filename_mfreq='', sim_vis_yy_filename_mfreq='', Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_ubls,
			                                                      flist=flist, freq_index=freq_index_fakemfreq, freq=freq, equatorial_GSM_standard_xx=equatorial_GSM_standard_xx, equatorial_GSM_standard_yy=equatorial_GSM_standard_yy, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq_xx, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq_yy, beam_weight=beam_weight,
			                                                      C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
			                                                      nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
			for id_p in range(2):
				fullsim_vis_mfreq[id_p, :-1, :, freq_index[id_p]] = fullsim_vis[:, id_p, :]
				fullsim_vis_mfreq[id_p, -1, :, freq_index[id_p]] = autocorr_vis[id_p]
			# freq_index = freq_index_fakemfreq
			print('>>>>Freq_index selected not fake: %s' % (str(freq_index)))
		
		else:
			full_sim_ubls = np.concatenate((used_common_ubls, [[0, 0, 0]]), axis=0)  # tag along auto corr
			full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
			full_decs = PI / 2 - full_thetas
			full_ras = full_phis
			full_sim_mask = hpf.get_interp_val(beam_weight, full_thetas, full_phis, nest=True) > 0
			# fullsim_vis_DBG = np.zeros((2, len(used_common_ubls), nt_used, np.sum(full_sim_mask)), dtype='complex128')
			
			print "Deep Parallel Simulating visibilities, %s, expected time %f min" % (datetime.datetime.now(), 14.6 * nf_used * (nUBL_used / 78.) * (nt_used / 193.) * (np.sum(full_sim_mask) / 1.4e5)),
			sys.stdout.flush()
			masked_equ_GSM_mfreq = equatorial_GSM_standard_mfreq[:, :, full_sim_mask]
			timer = time.time()
			if Parallel_Mulfreq_Visibility_deep:
				beam_heal_equ_mfreq = {0: beam_heal_equ_x_mfreq, 1: beam_heal_equ_y_mfreq}
				Visibility_multiprocess_list = np.array([[[pool.apply_async(Calculate_pointsource_visibility, args=(full_ras[full_sim_mask][n], full_decs[full_sim_mask][n], full_sim_ubls, f, None, beam_heal_equ_mfreq[id_p][id_f], None, lsts, False)) for n in range(len(full_ras[full_sim_mask]))] for id_f,f in enumerate(flist[id_p])] for id_p in range(2)])
				fullsim_vis_mfreq = np.array([[np.dot(np.array([Visibility_multiprocess_list[id_p][id_f][n].get() for n in range(len(full_ras[full_sim_mask]))]).transpose(1, 2, 0), masked_equ_GSM_mfreq[id_p, id_f, :]) for id_f,f in enumerate(flist[id_p])] for id_p in range(2)]).transpose(0, 2, 3, 1)
				print('Shape of fullsim_vis_mfreq parallel computed: %s' %(str(fullsim_vis_mfreq.shape)))
				
			else:
				for id_f, f in enumerate(flist[0]):
					for p, beam_heal_equ in enumerate([beam_heal_equ_x_mfreq[id_f], beam_heal_equ_y_mfreq[id_f]]):
						f = flist[p][id_f]
						for i, (ra, dec) in enumerate(zip(full_ras[full_sim_mask], full_decs[full_sim_mask])):
							# if Parallel_Mulfreq_Visibility:
							# 	res = vs.calculate_pointsource_visibility(ra, dec, full_sim_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
							res = vs.calculate_pointsource_visibility(ra, dec, full_sim_ubls, f, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
							fullsim_vis_mfreq[p, :, :, id_f] += masked_equ_GSM_mfreq[p, id_f, i] * res
				# fullsim_vis_DBG[p, ..., i] = res[:-1]
				# autocorr = ~16*la.norm, ~80*np.std, ~1.e-5*np.corrrelate
			print "simulated visibilities in %f minutes." % ((time.time() - timer) / 60.)
			try:
				fullsim_vis_mfreq.astype('complex128').tofile(full_sim_filename_mfreq)
				fullsim_vis_mfreq[0][:-1, :, :].astype('complex128').tofile(sim_vis_xx_filename_mfreq)
				fullsim_vis_mfreq[1][:-1, :, :].astype('complex128').tofile(sim_vis_yy_filename_mfreq)
			except:
				print('>>>>>>>>>>>>> Not Saved.')
	
	autocorr_vis_mfreq = np.abs(np.squeeze(fullsim_vis_mfreq[:, -1])) # (Pol, Times, Freqs)
	fullsim_vis_mfreq = np.squeeze(fullsim_vis_mfreq[:, :-1].transpose((1, 0, 2, 3)))  # (uBL, Pol, Times, Freqs)
	
	# autocorr_vis_mfreq_fulltime = copy.deepcopy(autocorr_vis_mfreq)
	# fullsim_vis_mfreq_fulltime = copy.deepcopy(fullsim_vis_mfreq)
    #
	# autocorr_vis_mfreq = autocorr_vis_mfreq[:, tmask, :]
	# fullsim_vis_mfreq = fullsim_vis_mfreq[:, :, tmask, :]
	
	if crosstalk_type == 'autocorr':
		autocorr_vis_mfreq_normalized = np.array([autocorr_vis[p, :, id_f] / (la.norm(autocorr_vis[p, :, id_f]) / la.norm(np.ones_like(autocorr_vis[p, :, id_f]))) for id_f in range(autocorr_vis_mfreq.shape[2]) for p in range(2)]).transpose(0, 2, 1)
	else:
		autocorr_vis_mfreq_normalized = np.ones_like(autocorr_vis_mfreq) #((2, nt_used, nf_used))
	
	if Multi_Sin_freq and Multi_freq:
		autocorr_vis = np.concatenate((autocorr_vis_mfreq[0:1, :, freq_index[0]], autocorr_vis_mfreq[1:2, :, freq_index[1]]), axis=0)
		autocorr_vis_normalized = np.concatenate((autocorr_vis_mfreq_normalized[0:1, :, freq_index[0]], autocorr_vis_mfreq_normalized[1:2, :, freq_index[1]]), axis=0)
		fullsim_vis = np.concatenate((fullsim_vis_mfreq[:, 0:1, :, freq_index[0]], fullsim_vis_mfreq[:, 1:2, :, freq_index[1]]), axis=1)
	
	if Multi_Sin_freq and Multi_freq:
		print('Shape of Autocorr_vis at %sMHz: %s' % (str(freq), str(autocorr_vis.shape)))
		print('Shape of Autocorr_vis_normalized at %sMHz: %s' % (str(freq), str(autocorr_vis_normalized.shape)))
		print('Shape of Fullsim_vis at %sMHz: %s' % (str(freq), str(fullsim_vis.shape)))
		print('Shape of Autocorr_vis_mfreq: %s' % (str(autocorr_vis_mfreq.shape)))
		print('Shape of Autocorr_vis_mfreq_normalized: %s' % (str(autocorr_vis_mfreq_normalized.shape)))
		print('Shape of Fullsim_vis_mfreq: %s' % (str(fullsim_vis_mfreq.shape)))
		return fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized, fullsim_vis, autocorr_vis, autocorr_vis_normalized
	
	else:
		print('Shape of Autocorr_vis_mfreq: %s' % (str(autocorr_vis_mfreq.shape)))
		print('Shape of Autocorr_vis_mfreq_normalized: %s' % (str(autocorr_vis_mfreq_normalized.shape)))
		print('Shape of Fullsim_vis_mfreq: %s' % (str(fullsim_vis_mfreq.shape)))
		return fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized


def Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq=False, Absolute_Calibration_dred=False, Fake_wgts_dred_mfreq=False, re_cal_times=1, antpos=None, Bandpass_Constrain=True,
                            Mocal_time_bin_temp=None, nt_used=None, lsts=None, Mocal_freq_bin_temp=None, flist=None, fullsim_vis_mfreq=None, vis_data_dred_mfreq=None, dflags_dred_mfreq=None, add_Autobsl=False, autocorr_vis_mfreq=None, autocorr_data_mfreq=None, bl_dred_mfreq_select=8,
                            INSTRUMENT=None, used_common_ubls=None, freq=None, nUBL_used=None, bnside=None, nside_standard=None, AmpCal_Pro=False, Index_Freq=0):
	if nt_used is not None:
		if nt_used != len(lsts):
			raise ValueError('nt_used doesnot match len(lsts).')
	nt_used = len(lsts)
	
	# Mocal_time_bin_temp = 5
	mocal_time_bin = np.min([Mocal_time_bin_temp, nt_used])
	mocal_time_bin_num = nt_used / mocal_time_bin if np.mod(nt_used, mocal_time_bin) == 0 else (nt_used / mocal_time_bin + 1)
	print('Mocal_time_bin_temp: %s; mocal_time_bin: %s; mocal_time_bin_num: %s' % (Mocal_time_bin_temp, mocal_time_bin, mocal_time_bin_num))
	
	# Mocal_freq_bin_temp = 64
	mocal_freq_bin = 1 if not Absolute_Calibration_dred_mfreq else np.min([Mocal_freq_bin_temp, len(flist[0])])
	mocal_freq_bin_num = len(flist[0]) / mocal_freq_bin if np.mod(len(flist[0]), mocal_freq_bin) == 0 else (len(flist[0]) / mocal_freq_bin + 1)
	print('Mocal_freq_bin_temp: %s; mocal_freq_bin: %s; mocal_freq_bin_num: %s' % (Mocal_freq_bin_temp, mocal_freq_bin, mocal_freq_bin_num))
	
	model_dred_mfreq = {}
	data_dred_mfreq = {}
	cdflags_dred_mfreq = {}
	abs_corr_data_dred_mfreq = {}
	vis_data_dred_mfreq_abscal = [[], []]
	autocorr_data_dred_mfreq_abscal = [[], []]
	vis_data_dred_abscal = [[], []]
	autocorr_data_dred_abscal = [[], []]
	interp_flags_dred_mfreq = {}
	AC_dred_mfreq = {}
	DAC_dred_mfreq = {}
	DPAC_dred_mfreq = {}
	dly_phs_corr_data_dred_mfreq = {}
	auto_select_dred_mfreq = {}
	delay_corr_data_dred_mfreq = {}
	
	# try:
	# 	cdflags_dred_mfreq = copy.deepcopy(dflags_dred_mfreq)
	# except:
	# 	pass
	
	wgts_dred_mfreq = {}
	for i in range(2):
		pol = ['xx', 'yy'][i]
		re_cal = 0
		model_dred_mfreq[i] = LastUpdatedOrderedDict()
		data_dred_mfreq[i] = LastUpdatedOrderedDict()
		cdflags_dred_mfreq[i] = LastUpdatedOrderedDict()
		
		vis_data_dred_mfreq_abscal[i] = np.zeros_like(vis_data_dred_mfreq[i], dtype='complex128')
		autocorr_data_dred_mfreq_abscal[i] = np.zeros_like(autocorr_vis_mfreq[i])
		vis_data_dred_abscal[i] = np.zeros_like(vis_data_dred_mfreq_abscal[i][index_freq[i], :, :])
		autocorr_data_dred_abscal[i] = np.zeros_like(autocorr_vis_mfreq[i][:, index_freq[i]])
		
		for id_t_bin in range(mocal_time_bin_num):
			nt_mocal_used = mocal_time_bin if (id_t_bin + 1) * mocal_time_bin <= nt_used else (nt_used - id_t_bin * mocal_time_bin)
			
			for id_f_bin in range(mocal_freq_bin_num):
				nf_mocal_used = mocal_freq_bin if (id_f_bin + 1) * mocal_freq_bin <= len(flist[0]) else (len(flist[0]) - id_f_bin * mocal_freq_bin)
				
				keys = dflags_dred_mfreq[i].keys()
				for key_index, key in enumerate(keys):
					model_dred_mfreq[i][key] = fullsim_vis_mfreq[key_index, i, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
					# data_dred_mfreq[i][key] = np.real(vis_data_dred_mfreq[i][:, :, key_index].transpose()) + np.abs(np.imag(vis_data_dred_mfreq[i][:, :, key_index].transpose()))*1j #[pol][freq,time,ubl_index].transpose()
					data_dred_mfreq[i][key] = vis_data_dred_mfreq[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, key_index].transpose()  # [pol][freq,time,ubl_index].transpose()
					cdflags_dred_mfreq[i][key] = dflags_dred_mfreq[i][key][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
				if add_Autobsl:
					model_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = autocorr_vis_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]  # not lose generality, choose the first anntena in the first UBL for autocorrelation calibraiton.
					data_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]  # add the autocorrelation of first antenna in the first UBL as the last line in visibility.
					cdflags_dred_mfreq[i][keys[0][0], keys[0][0], keys[0][2]] = np.array([[False] * autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used].shape[1]] * autocorr_data_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used].shape[0])
					auto_select_dred_mfreq[i] = (keys[0][0], keys[0][0], keys[0][2])
				print(dflags_dred_mfreq[i].keys())
				print(dflags_dred_mfreq[i].keys()[0][0])
				print('(id_t_bin: %s, id_f_bin: %s) data_shape[%s][%s]: (%s) \n' % (id_t_bin, id_f_bin, ['xx', 'yy'][i], key, data_dred_mfreq[i][key].shape))
				
				wgts_dred_mfreq[i] = copy.deepcopy(cdflags_dred_mfreq[i])
				for k in wgts_dred_mfreq[i].keys():
					if not Fake_wgts_dred_mfreq:
						wgts_dred_mfreq[i][k] = (~wgts_dred_mfreq[i][k]).astype(np.float)
					else:
						wgts_dred_mfreq[i][k] = (((~wgts_dred_mfreq[i][k]).astype(np.float) + 1).astype(bool)).astype(np.float)
				
				lsts_binned = lsts[id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used]
				flist_binned = flist[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
				
				for re_cal in range(re_cal_times):  # number of times of absolute calibration
					if re_cal == 0:
						if not Absolute_Calibration_dred_mfreq:
							# Skip Delay_Lincal
							# instantiate class
							DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
							print('>>>>>>>>Fake mfreq used.')
						else:
							# instantiate class
							print('>>>>>>>>True mfreq used.')
							try:
								model_dred_mfreq[i], interp_flags_dred_mfreq[i] = hc.abscal.interp2d_vis(model_dred_mfreq[i], lsts_binned, flist_binned, lsts_binned, flist_binned)
							except:
								print('No Interp')
							AC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
							if Bandpass_Constrain:
								# kernel is median filter kernel, chosen to produce time-smooth output delays for this particular dataset
								AC_dred_mfreq[i].delay_slope_lincal(kernel=(1, ((np.min([nf_mocal_used, 3]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True)
								# apply to data
								delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].dly_slope_gain))
							else:
								AC_dred_mfreq[i].delay_lincal(kernel=(1, ((np.min([nf_mocal_used, 11]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True, solve_offsets=True)
								# apply to data
								delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].ant_dly_gain, AC_dred_mfreq[i].ant_dly_phi_gain))
							# instantiate class
							DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
					else:
						if not Absolute_Calibration_dred_mfreq:
							# delay_corr_data_dred_mfreq[i] = abs_corr_data_dred_mfreq[i]
							DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], abs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
						else:
							# instantiate class
							AC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], abs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
							if Bandpass_Constrain:
								# kernel is median filter kernel, chosen to produce time-smooth output delays for this particular dataset
								AC_dred_mfreq[i].delay_slope_lincal(kernel=(1, ((np.min([nf_mocal_used, 3]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True)
								# apply to data
								delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].dly_slope_gain))
							else:
								AC_dred_mfreq[i].delay_lincal(kernel=(1, ((np.min([nf_mocal_used, 11]) - 1) / 2 * 2 + 1)), medfilt=True, time_avg=True, solve_offsets=True)
								# apply to data
								delay_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(AC_dred_mfreq[i].data, (AC_dred_mfreq[i].ant_dly_gain, AC_dred_mfreq[i].ant_dly_phi_gain))
							# instantiate class
							DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
					
					# # instantiate class
					# DAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], delay_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
					# avg phase solver
					DAC_dred_mfreq[i].phs_logcal(avg=True)
					# apply to data
					dly_phs_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(DAC_dred_mfreq[i].data, (DAC_dred_mfreq[i].ant_phi_gain))
					# instantiate class
					DPAC_dred_mfreq[i] = hc.abscal.AbsCal(model_dred_mfreq[i], dly_phs_corr_data_dred_mfreq[i], antpos=antpos[i], wgts=wgts_dred_mfreq[i], freqs=flist_binned)
					
					if Bandpass_Constrain:
						# run amp linsolve
						DPAC_dred_mfreq[i].abs_amp_logcal()
						# run phs linsolve
						DPAC_dred_mfreq[i].TT_phs_logcal(zero_psi=False, four_pol=False)
						# apply to data
						abs_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(DPAC_dred_mfreq[i].data, (DPAC_dred_mfreq[i].abs_psi_gain, DPAC_dred_mfreq[i].TT_Phi_gain, DPAC_dred_mfreq[i].abs_eta_gain), gain_convention='divide')
					else:
						# run amp logcal
						DPAC_dred_mfreq[i].amp_logcal()
						# run phase calibration
						DPAC_dred_mfreq[i].phs_logcal()
						# apply to data
						abs_corr_data_dred_mfreq[i] = hc.abscal.apply_gains(DPAC_dred_mfreq[i].data, (DPAC_dred_mfreq[i].ant_eta_gain, DPAC_dred_mfreq[i].ant_phi_gain))
				
				for key_id, key in enumerate(dflags_dred_mfreq[i].keys()):
					if not AmpCal_Pro:
						vis_data_dred_mfreq_abscal[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, key_id] = abs_corr_data_dred_mfreq[i][key].transpose()
					else:
						vis_data_dred_mfreq_abscal[i][id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used, id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, key_id] = abs_corr_data_dred_mfreq[i][key].transpose() * (np.sum(np.abs(model_dred_mfreq[i][key].transpose()), axis=-1) / np.sum(np.abs(abs_corr_data_dred_mfreq[i][key].transpose()), axis=-1)).reshape(nf_mocal_used, 1)
					# vis_data_dred_mfreq_abscal[i][:, :, key_id] = np.real(abs_corr_data_dred_mfreq[i][key].transpose()) + np.abs(np.imag(abs_corr_data_dred_mfreq[i][key].transpose()))*1j
				if add_Autobsl:
					autocorr_data_dred_mfreq_abscal[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used] = abs_corr_data_dred_mfreq[i][auto_select_dred_mfreq[i]]
				else:
					autocorr_data_dred_mfreq_abscal[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used] = autocorr_vis_mfreq[i][id_t_bin * mocal_time_bin:id_t_bin * mocal_time_bin + nt_mocal_used, id_f_bin * mocal_freq_bin:id_f_bin * mocal_freq_bin + nf_mocal_used]
		
		vis_data_dred_abscal[i] = vis_data_dred_mfreq_abscal[i][index_freq[i], :, :]
		if add_Autobsl:
			autocorr_data_dred_abscal[i] = autocorr_data_dred_mfreq_abscal[i][:, index_freq[i]]
		else:
			autocorr_data_dred_abscal[i] = autocorr_vis_mfreq[i][:, index_freq[i]]
	
	try:
		bl_dred_mfreq = [dflags_dred_mfreq[0].keys()[bl_dred_mfreq_select], dflags_dred_mfreq[1].keys()[bl_dred_mfreq_select]]  # [(25, 37, 'xx'), (25, 37, 'yy')]
		fig3 = {}
		axes3 = {}
		fig3_data = {}
		axes3_data = {}
		fig3_data_abscorr = {}
		axes3_data_abscorr = {}
		for i in range(2):  # add another redundant 'for loop' for testing plotting.
			pol = ['xx', 'yy'][i]
			try:
				plt.figure(80000000 + 10 * i + Index_Freq)
				fig3[i], axes3[i] = plt.subplots(2, 1, figsize=(12, 8))
				plt.sca(axes3[i][0])
				uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_select, i, :, :], mode='log', mx=6, drng=4)
				plt.colorbar()
				plt.title(pol + ' model AMP {}'.format(bl_dred_mfreq[i]))
				plt.sca(axes3[i][1])
				uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_select, i, :, :], mode='phs', mx=np.pi, drng=2 * np.pi)
				plt.colorbar()
				plt.title(pol + ' model PHS {}'.format(bl_dred_mfreq[i]))
				plt.show(block=False)
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_model-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				# plt.cla()
				
				plt.figure(90000000 + 10 * i + Index_Freq)
				fig3_data[i], axes3_data[i] = plt.subplots(2, 1, figsize=(12, 8))
				plt.sca(axes3_data[i][0])
				uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_select].transpose(), mode='log', mx=2, drng=6)
				plt.colorbar()
				plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq[i]))
				plt.sca(axes3_data[i][1])
				uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
				plt.colorbar()
				plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq[i]))
				plt.show(block=False)
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				# plt.cla()
				
				####################### after ABS Calibration #########################
				
				plt.figure(8000000 + 10 * i + Index_Freq)
				fig3_data_abscorr[i], axes3_data_abscorr[i] = plt.subplots(2, 1, figsize=(12, 8))
				plt.sca(axes3_data_abscorr[i][0])
				uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_select].transpose(), mode='log', mx=6, drng=6)
				plt.colorbar()
				plt.title(pol + ' abs_caled data AMP {}'.format(bl_dred_mfreq[i]))
				plt.sca(axes3_data_abscorr[i][1])
				uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
				plt.colorbar()
				plt.title(pol + ' abs_caled data PHS {}'.format(bl_dred_mfreq[i]))
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Modcal_data-caled-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_select, 0], used_common_ubls[bl_dred_mfreq_select, 1], ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				plt.show(block=False)
			# plt.cla()
			except:
				print('Error when Plotting Mocal Results.')
	except:
		print('No Plotting for Model_Calibration Results.')
	
	return vis_data_dred_mfreq_abscal, autocorr_data_dred_mfreq_abscal, vis_data_dred_abscal, autocorr_data_dred_abscal, mocal_time_bin, mocal_freq_bin


def PointSource_Calibration(data_var_xx_filename_pscal='', data_var_yy_filename_pscal='', PointSource_AbsCal=False, PointSource_AbsCal_SingleFreq=False, Pt_vis=False, From_AbsCal=False, comply_ps2mod_autocorr=False, southern_points=None, phase_degen_niter_max=50,
                            index_freq=None, freq=None, flist=None, lsts=None, tlist=None, tmask=True, vis_data_dred_mfreq=None, vis_data_dred_mfreq_abscal=None, autocorr_data_mfreq=None, autocorr_data_dred_mfreq_abscal=None, equatorial_GSM_standard_mfreq=None, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None,
                            Integration_Time=None, Frequency_Bin=None, used_redundancy=None, nt=None, nUBL=None, ubls=None, bl_dred_mfreq_pscal_select=8, dflags_dred_mfreq=None, INSTRUMENT=None, used_common_ubls=None, nUBL_used=None, nt_used=None, bnside=None, nside_standard=None, scale_noise=False, scale_noise_ratio=1.):
	for source in southern_points.keys():
		southern_points[source]['body'] = ephem.FixedBody()
		southern_points[source]['body']._ra = southern_points[source]['ra']
		southern_points[source]['body']._dec = southern_points[source]['dec']
	
	flux_func = {}
	# flux_func['cas'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/casA2013.5out')[:,2])
	# flux_func['cyg'] = si.interp1d(np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,1], np.loadtxt('/home/omniscope/data/point_source_flux/cygA2006out')[:,2])
	flux_func['cas'] = si.interp1d(flist[0], np.array([S_casa_v_t(flist[0][i], DecimalYear) for i in range(len(flist[0]))]))
	flux_func['cyg'] = si.interp1d(flist[0], np.array([S_cyga_v(flist[0][i], DecimalYear) for i in range(len(flist[0]))]))
	
	full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
	
	if Pt_vis:
		flux_raw_gsm_ps = {}
		flux_gsm_ps = {}
		flux_raw_dis_gsm_ps = {}
		flux_dis_gsm_ps = {}
		pix_index_gsm_ps = {}
		pix_raw_index_gsm_ps = {}
		pix_max_index_gsm_ps = {}
		pt_sources = southern_points.keys()
		for source in pt_sources:
			flux_raw_gsm_ps[source] = 0
			flux_gsm_ps[source] = 0
			flux_raw_dis_gsm_ps[source] = []
			flux_dis_gsm_ps[source] = []
			pix_raw_index_gsm_ps[source] = []
			pix_index_gsm_ps[source] = []
			# pix_max_index_gsm_ps[source] = []
			for i in range(len(equatorial_GSM_standard)):
				if la.norm(np.array([full_phis[i] - southern_points[source]['body']._ra,
				                     (PI / 2 - full_thetas[i]) - southern_points[source]['body']._dec])) <= 0.1:
					flux_raw_gsm_ps[source] += equatorial_GSM_standard[i]
					flux_raw_dis_gsm_ps[source].append(equatorial_GSM_standard[i])
					pix_raw_index_gsm_ps[source].append(i)
			
			pix_max_index_gsm_ps[source] = pix_raw_index_gsm_ps[source][flux_raw_dis_gsm_ps[source].index(np.array(flux_raw_dis_gsm_ps[source]).max())]
			for j in range(len(flux_raw_dis_gsm_ps[source])):
				if flux_raw_dis_gsm_ps[source][j] >= 0.4 * equatorial_GSM_standard[pix_max_index_gsm_ps[source]]:
					flux_gsm_ps[source] += equatorial_GSM_standard[pix_raw_index_gsm_ps[source][j]]
					flux_dis_gsm_ps[source].append(equatorial_GSM_standard[pix_raw_index_gsm_ps[source][j]])
					pix_index_gsm_ps[source].append(pix_raw_index_gsm_ps[source][j])
			
			print('total flux of %s' % source, flux_gsm_ps[source])
			print('total raw flux of %s' % source, flux_raw_gsm_ps[source])
			print('maximum pix flux of %s' % source, equatorial_GSM_standard[pix_max_index_gsm_ps[source]])
			print('pix-index with maximum flux of %s' % source, pix_max_index_gsm_ps[source])
			print('raw-pix-indexes of %s' % source, pix_raw_index_gsm_ps[source])
			print('pix-indexes of %s' % source, pix_index_gsm_ps[source])
			print('\n')
		
		# pt_sources = ['cyg', 'cas']
		pt_sources = southern_points.keys()
		pt_vis = np.zeros((len(pt_sources), 2, nUBL_used, nt_used), dtype='complex128')
		if INSTRUMENT == 'miteor':
			print "Simulating cyg casvisibilities, %s, expected time %.1f min" % (datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (2. / 1.4e5)),
			sys.stdout.flush()
			timer = time.time()
			for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
				for i, source in enumerate(pt_sources):
					ra = southern_points[source]['body']._ra
					dec = southern_points[source]['body']._dec
					# 			pt_vis[i, p] = jansky2kelvin * flux_func[source](freq) * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
					pt_vis[i, p] = flux_gsm_ps[source] * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
		elif 'hera47' in INSTRUMENT:
			print "Simulating cyg casvisibilities, %s, expected time %.1f min" % (datetime.datetime.now(), 14.6 * (nUBL_used / 78.) * (nt_used / 193.) * (2. / 1.4e5)),
			sys.stdout.flush()
			timer = time.time()
			for p, beam_heal_equ in enumerate([beam_heal_equ_x, beam_heal_equ_y]):
				for i, source in enumerate(pt_sources):
					ra = southern_points[source]['body']._ra
					dec = southern_points[source]['body']._dec
					# 			pt_vis[i, p] = jansky2kelvin * flux_func[source](freq) * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
					pt_vis[i, p] = flux_gsm_ps[source] * vs.calculate_pointsource_visibility(ra, dec, used_common_ubls, freq, beam_heal_equ=beam_heal_equ, tlist=lsts) / 2
	
	vis_freq = {}
	
	autocorr_data_dred_mfreq_pscal = {}
	vis_data_dred_mfreq_pscal = {}
	
	if PointSource_AbsCal_SingleFreq:
		pscal_freqstart = index_freq[0]
		pscal_freqend = index_freq[0] + 1
	else:
		pscal_freqstart = 0
		pscal_freqend = np.min([len(flist[0]), len(flist[1])])
	
	for j, p in enumerate(['x', 'y']):
		pol = p + p
		vis_data_dred_mfreq_pscal[j] = np.zeros_like(vis_data_dred_mfreq[j])
		autocorr_data_dred_mfreq_pscal[j] = np.zeros_like(autocorr_data_mfreq[j])
	
	for id_f in range(pscal_freqstart, pscal_freqend, 1):
		vis_freq[0] = flist[0][id_f]
		vis_freq[1] = flist[1][id_f]
		# cal_lst_range = np.array([5, 6]) / TPI * 24.
		# 		cal_lst_range = np.array([tlist[15], tlist[-15]])
		# cal_lst_range = np.array([tlist[len(tlist) / 3], tlist[-len(tlist) / 3]])
		calibrate_ubl_length = 2600 / np.mean([vis_freq[0], vis_freq[1]])  # 10.67
		# cal_time_mask = tmask	 #(tlist>cal_lst_range[0]) & (tlist<cal_lst_range[1])#a True/False mask on all good data to get good data in cal time range
		cal_time_mask = tmask #(tlist >= cal_lst_range[0]) & (tlist <= cal_lst_range[1]) &
		# cal_ubl_mask = np.linalg.norm(ubls[p], axis=1) >= calibrate_ubl_length
		
		print('%i times used' % len(lsts[cal_time_mask]))
		
		flux_raw_gsm_ps = {}
		flux_gsm_ps = {}
		flux_raw_dis_gsm_ps = {}
		flux_dis_gsm_ps = {}
		pix_index_gsm_ps = {}
		pix_raw_index_gsm_ps = {}
		pix_max_index_gsm_ps = {}
		pt_sources = southern_points.keys()
		for source in pt_sources:
			flux_raw_gsm_ps[source] = 0
			flux_gsm_ps[source] = 0
			flux_raw_dis_gsm_ps[source] = []
			flux_dis_gsm_ps[source] = []
			pix_raw_index_gsm_ps[source] = []
			pix_index_gsm_ps[source] = []
			# pix_max_index_gsm_ps[source] = []
			for i in range(len(equatorial_GSM_standard_mfreq[id_f])):
				if la.norm(np.array([full_phis[i] - southern_points[source]['body']._ra,
				                     (PI / 2 - full_thetas[i]) - southern_points[source]['body']._dec])) <= 0.1:
					flux_raw_gsm_ps[source] += equatorial_GSM_standard_mfreq[id_f, i]
					flux_raw_dis_gsm_ps[source].append(equatorial_GSM_standard_mfreq[id_f, i])
					pix_raw_index_gsm_ps[source].append(i)
			
			pix_max_index_gsm_ps[source] = pix_raw_index_gsm_ps[source][flux_raw_dis_gsm_ps[source].index(np.array(flux_raw_dis_gsm_ps[source]).max())]
			for j in range(len(flux_raw_dis_gsm_ps[source])):
				if flux_raw_dis_gsm_ps[source][j] >= 0.5 * equatorial_GSM_standard_mfreq[id_f, pix_max_index_gsm_ps[source]]:
					flux_gsm_ps[source] += equatorial_GSM_standard_mfreq[id_f, pix_raw_index_gsm_ps[source][j]]
					flux_dis_gsm_ps[source].append(equatorial_GSM_standard_mfreq[id_f, pix_raw_index_gsm_ps[source][j]])
					pix_index_gsm_ps[source].append(pix_raw_index_gsm_ps[source][j])
			
			print('total flux of %s' % source, flux_gsm_ps[source])
			print('total raw flux of %s' % source, flux_raw_gsm_ps[source])
			print('maximum pix flux of %s' % source, equatorial_GSM_standard_mfreq[id_f, pix_max_index_gsm_ps[source]])
			print('pix-index with maximum flux of %s' % source, pix_max_index_gsm_ps[source])
			print('raw-pix-indexes of %s' % source, pix_raw_index_gsm_ps[source])
			print('pix-indexes of %s' % source, pix_index_gsm_ps[source])
			print('\n')
		
		Ni = {}
		cubls = copy.deepcopy(ubls)
		ubl_sort = {}
		noise_data_pscal = {}
		N_data_pscal = {}
		vis_data_dred_pscal = {}
		
		From_AbsCal = False
		
		for i, p in enumerate(['x', 'y']):
			pol = p + p
			cal_ubl_mask = np.linalg.norm(ubls[p], axis=1) >= calibrate_ubl_length
			# get Ni (1/variance) and data
			# var_filename = datadir + tag + '_%s%s_%i_%i%s.var'%(p, p, nt, nUBL, vartag)
			# noise_data_pscal['y'] = np.array([(np.random.normal(0,autocorr_data[1][t_index]/(Integration_Time*Frequency_Bin)**0.5,nUBL_used) ) for t_index in range(len(autocorr_data[1]))],dtype='float64').flatten()
			
			if From_AbsCal:
				vis_data_dred_pscal[i] = vis_data_dred_mfreq_abscal[i][id_f][np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
				noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_abscal[i][t_index, id_f] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(autocorr_data_dred_mfreq_abscal[0].shape[0])], dtype='float64').flatten()  # Absolute Calibrated
			else:
				vis_data_dred_pscal[i] = vis_data_dred_mfreq[i][id_f][np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
				noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_mfreq[i][t_index, id_f] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(autocorr_data_mfreq[0].shape[0])], dtype='float64').flatten()  # Absolute Calibrated
			
			N_data_pscal[p] = noise_data_pscal[p] * noise_data_pscal[p]
			# N_data_pscal[p] = N_data[p]
			# N_data_pscal['y'] = noise_data_pscal['y'] * noise_data_pscal['y']
			Ni[p] = 1. / N_data_pscal[p].reshape((nt, nUBL))[np.ix_(cal_time_mask, cal_ubl_mask)].transpose()
			ubls[p] = ubls[p][cal_ubl_mask]
			ubl_sort[p] = np.argsort(la.norm(ubls[p], axis=1))
			
			print "%i UBLs to include" % len(ubls[p])
		
		del (noise_data_pscal)
		
		print "Computing UNpolarized point sources matrix..."
		sys.stdout.flush()
		# cal_sources = ['cyg', 'cas']
		cal_sources = southern_points.keys()
		Apol = np.empty((np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask), len(cal_sources)), dtype='complex128')
		timer = time.time()
		for n, source in enumerate(cal_sources):
			ra = southern_points[source]['body']._ra
			dec = southern_points[source]['body']._dec
			
			Apol[:, 0, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq[0], beam_heal_equ=beam_heal_equ_x_mfreq[id_f], tlist=lsts[cal_time_mask])
			Apol[:, 1, :, n] = vs.calculate_pointsource_visibility(ra, dec, ubls[p], vis_freq[1], beam_heal_equ=beam_heal_equ_y_mfreq[id_f], tlist=lsts[cal_time_mask])
		
		Apol = np.conjugate(Apol).reshape((np.sum(cal_ubl_mask), 2 * np.sum(cal_time_mask), len(cal_sources)))
		Ni = np.transpose([Ni['x'], Ni['y']], (1, 0, 2))
		
		realA = np.zeros((2 * Apol.shape[0] * Apol.shape[1], 1 + 2 * np.sum(cal_ubl_mask) * 2), dtype='float64')
		# 		realA[:, 0] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2])))), axis=0).dot([jansky2kelvin_mfreq[0][id_f] * flux_func[source](vis_freq[0]) for source in cal_sources])
		realA[:, 0] = np.concatenate((np.real(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2]))), np.imag(Apol.reshape((Apol.shape[0] * Apol.shape[1], Apol.shape[2])))), axis=0).dot([flux_gsm_ps[source] for source in cal_sources])
		vis_scale = la.norm(realA[:, 0]) / len(realA) ** .5
		for coli, ncol in enumerate(range(1, realA.shape[1])):
			realA[coli * np.sum(cal_time_mask): (coli + 1) * np.sum(cal_time_mask), ncol] = vis_scale
		
		realNi = np.concatenate((np.real(Ni.flatten()), np.imag(Ni.flatten())))
		realAtNiAinv = np.linalg.pinv(np.einsum('ji,j,jk->ik', realA, realNi, realA))
		
		b = np.transpose([vis_data_dred_pscal[0], vis_data_dred_pscal[1]], (1, 0, 2))
		phase_degen_niter = 0
		phase_degen2 = {'x': np.zeros(2), 'y': np.zeros(2)}
		phase_degen_iterative_x = np.zeros(2)
		phase_degen_iterative_y = np.zeros(2)
		
		def tocomplex(realdata):
			reshapedata = realdata.reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
			return reshapedata[0] + reshapedata[1] * 1.j
		
		# phase_degen_niter_max = 100
		while (phase_degen_niter < phase_degen_niter_max and max(np.linalg.norm(phase_degen_iterative_x), np.linalg.norm(phase_degen_iterative_y)) > 1e-5) or phase_degen_niter == 0:
			phase_degen_niter += 1
			b[:, 0] = b[:, 0] * np.exp(1.j * ubls['x'][:, :2].dot(phase_degen_iterative_x))[:, None]
			b[:, -1] = b[:, -1] * np.exp(1.j * ubls['y'][:, :2].dot(phase_degen_iterative_y))[:, None]
			realb = np.concatenate((np.real(b.flatten()), np.imag(b.flatten())))
			
			psol = realAtNiAinv.dot(np.transpose(realA).dot(realNi * realb))
			realb_fit = realA.dot(psol)
			perror = ((realb_fit - realb) * (realNi ** .5)).reshape((2, np.sum(cal_ubl_mask), 2, np.sum(cal_time_mask)))
			
			realbfit_noadditive = realA[:, 0] * psol[0]
			realbfit_additive = realb_fit - realbfit_noadditive
			realb_noadditive = realb - realbfit_additive
			bfit_noadditive = tocomplex(realbfit_noadditive)
			b_noadditive = tocomplex(realb_noadditive)
			if phase_degen_niter == phase_degen_niter_max:
				phase_degen_iterative_x = solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'])  # , [3, 3, 1e3])
				phase_degen_iterative_y = solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'])  # , [3, 3, 1e3])
			
			else:
				phase_degen_iterative_x = solve_phase_degen(np.transpose(b_noadditive[:, 0]), np.transpose(b_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), np.transpose(bfit_noadditive[:, 0]), ubls['x'])
				phase_degen_iterative_y = solve_phase_degen(np.transpose(b_noadditive[:, -1]), np.transpose(b_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), np.transpose(bfit_noadditive[:, -1]), ubls['y'])
			phase_degen2['x'] += phase_degen_iterative_x
			phase_degen2['y'] += phase_degen_iterative_y
			print phase_degen_niter, phase_degen2['x'], phase_degen2['y'], np.linalg.norm(perror)
		
		renorm = 1 / (2 * psol[0])
		
		print (renorm, vis_freq[0], phase_degen2['x'], vis_freq[1], phase_degen2['y'])
		
		# freqs[fi] = vis_freq
		
		################################# apply to data and var and output unpolarized version ####################################
		# data_var_xx_filename_pscal = script_dir + '/../Output/%s_%s_p2_u%i_t%i_nside%i_bnside%i_var_data_xx_pscal.simvis' % (INSTRUMENT, freq, nUBL, nt, nside_standard, bnside)
		# data_var_yy_filename_pscal = script_dir + '/../Output/%s_%s_p2_u%i_t%i_nside%i_bnside%i_var_data_yy_pscal.simvis' % (INSTRUMENT, freq, nUBL, nt, nside_standard, bnside)
		
		######### recover ubls and ubl_sort ##########
		ubls = cubls
		# ubl_sort = cubl_sort
		
		if Keep_Red:
			nUBL = len(bsl_coord_x)
			for p in ['x', 'y']:
				# ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
				ubls[p] = globals()['bsl_coord_' + p]
			common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
		
		else:
			nUBL = len(bsl_coord_dred[0])
			nUBL_yy = len(bsl_coord_dred[1])
			for i in range(2):
				p = ['x', 'y'][i]
				ubls[p] = bsl_coord_dred[i]
			common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
		
		# get data and var and apply change
		
		for j, p in enumerate(['x', 'y']):
			pol = p + p
			
			if From_AbsCal:
				vis_data_dred_mfreq_pscal[j][id_f] = vis_data_dred_mfreq_abscal[j][id_f] * np.exp(1.j * ubls[p][:, :2].dot(phase_degen2[p])) * renorm
				if comply_ps2mod_autocorr:
					autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_vis_mfreq[j][:, id_f]
				else:
					autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_data_dred_mfreq_abscal[j][:, id_f] * np.abs(renorm)  # Absolute Calibrated
			else:
				vis_data_dred_mfreq_pscal[j][id_f] = vis_data_dred_mfreq[j][id_f] * np.exp(1.j * ubls[p][:, :2].dot(phase_degen2[p])) * renorm
				if comply_ps2mod_autocorr:
					autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_vis_mfreq[j][:, id_f]
				else:
					autocorr_data_dred_mfreq_pscal[j][:, id_f] = autocorr_data_mfreq[j][:, id_f] * np.abs(renorm)  # Absolute Calibrated
	
	noise_data_pscal = {}
	N_data_pscal = {}
	vis_data_dred_pscal = {}
	for i, p in enumerate(['x', 'y']):
		pol = p + p
		vis_data_dred_pscal[i] = vis_data_dred_mfreq_pscal[i][index_freq[i], tmask, :]
		if not scale_noise:
			noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_pscal[i][t_index, index_freq[i]] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[i]) ** 0.5) for t_index in range(len(autocorr_data_dred_mfreq_pscal[i]))], dtype='float64').flatten()  # Absolute Calibrated
			# noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_pscal[i][t_index, index_freq[i]] * np.exp(np.random.normal(0, 2 * np.pi, 1) * 1.j)  / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[i]) ** 0.5) for t_index in range(len(autocorr_data_dred_mfreq_pscal[i]))], dtype='complex128').flatten()  # Absolute Calibrated
		else:
			noise_data_pscal[p] = np.array([(np.random.normal(0, autocorr_data_dred_mfreq_pscal[i][t_index, index_freq[i]] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL) / np.array(used_redundancy[i]) ** 0.5) for t_index in range(len(autocorr_data_dred_mfreq_pscal[i]))], dtype='float64').flatten()  # Absolute Calibrated
			# noise_data_pscal[p] = np.array([(np.random.normal(0, np.mean(np.abs(vis_data_dred_pscal[i]), axis=1)[t_index] * scale_noise_ratio, nUBL) * np.exp(np.random.normal(0, np.pi, nUBL_used) * 1.j) / np.array(used_redundancy[i]) ** 0.5) for t_index in range(len(autocorr_data_dred_mfreq_pscal[i]))], dtype='complex128').flatten()  # Absolute Calibrated
			
		N_data_pscal[p] = noise_data_pscal[p] * noise_data_pscal[p]
		N_data_pscal[p] = N_data_pscal[p].reshape((nt, nUBL))
		
		if not os.path.isfile(globals()['data_var_' + pol + '_filename_pscal']):
			try:
				N_data_pscal[p].astype('float64').tofile(globals()['data_var_' + pol + '_filename_pscal'])
			except:
				print('N_data_pscal not saved.')
		else:
			print('N_data_pscal already exists on disc.')
	# (new_var * 100.).astype('float32').tofile(op_var100_filename)
	del (noise_data_pscal)
	
	try:
		bl_dred_mfreq_pscal = [dflags_dred_mfreq[0].keys()[bl_dred_mfreq_pscal_select], dflags_dred_mfreq[1].keys()[bl_dred_mfreq_pscal_select]]  # [(25, 37, 'xx'), (25, 37, 'yy')]
		fig4 = {}
		axes4 = {}
		fig4_data = {}
		axes4_data = {}
		fig4_data_abscorr = {}
		axes4_data_abscorr = {}
		for i in range(2):  # add another redundant 'for loop' for testing plotting.
			pol = ['xx', 'yy'][i]
			try:
				plt.figure(80000000 + 10 * i)
				fig4[i], axes4[i] = plt.subplots(2, 1, figsize=(12, 8))
				plt.sca(axes4[i][0])
				uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_pscal_select, i, :, :], mode='log', mx=6, drng=4)
				plt.colorbar()
				plt.title(pol + ' model AMP {}'.format(bl_dred_mfreq_pscal[i]))
				plt.sca(axes4[i][1])
				uvt.plot.waterfall(fullsim_vis_mfreq[bl_dred_mfreq_pscal_select, i, :, :], mode='phs', mx=np.pi, drng=2 * np.pi)
				plt.colorbar()
				plt.title(pol + ' model PHS {}'.format(bl_dred_mfreq_pscal[i]))
				plt.show(block=False)
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_model-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				# plt.cla()
				
				if From_AbsCal:
					plt.figure(90000000 + 10 * i)
					fig4_data[i], axes4_data[i] = plt.subplots(2, 1, figsize=(12, 8))
					plt.sca(axes4_data[i][0])
					uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=1.5, drng=5)
					plt.colorbar()
					plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq_pscal[i]))
					plt.sca(axes4_data[i][1])
					uvt.plot.waterfall(vis_data_dred_mfreq_abscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
					plt.colorbar()
					plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq_pscal[i]))
					plt.show(block=False)
					plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				
				else:
					plt.figure(90000000 + 10 * i)
					fig4_data[i], axes4_data[i] = plt.subplots(2, 1, figsize=(12, 8))
					plt.sca(axes4_data[i][0])
					uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=1.5, drng=5)
					plt.colorbar()
					plt.title(pol + ' data AMP {}'.format(bl_dred_mfreq_pscal[i]))
					plt.sca(axes4_data[i][1])
					uvt.plot.waterfall(vis_data_dred_mfreq[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
					plt.colorbar()
					plt.title(pol + ' data PHS {}'.format(bl_dred_mfreq_pscal[i]))
					plt.show(block=False)
					plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				# plt.cla()
				
				####################### after ABS Calibration #########################
				
				plt.figure(8000000 + 10 * i)
				fig4_data_abscorr[i], axes4_data_abscorr[i] = plt.subplots(2, 1, figsize=(12, 8))
				plt.sca(axes4_data_abscorr[i][0])
				uvt.plot.waterfall(vis_data_dred_mfreq_pscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='log', mx=6, drng=4)
				plt.colorbar()
				plt.title(pol + ' abs_caled data AMP {}'.format(bl_dred_mfreq_pscal[i]))
				plt.sca(axes4_data_abscorr[i][1])
				uvt.plot.waterfall(vis_data_dred_mfreq_pscal[i][:, :, bl_dred_mfreq_pscal_select].transpose(), mode='phs', mx=np.pi, drng=2 * np.pi)
				plt.colorbar()
				plt.title(pol + ' abs_caled data PHS {}'.format(bl_dred_mfreq_pscal[i]))
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-Pscal-%s_data-caled-%s-%.2fMHz-nubl%s-nt%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[bl_dred_mfreq_pscal_select, 0], used_common_ubls[bl_dred_mfreq_pscal_select, 1], 'SinFreq' if PointSource_AbsCal_SingleFreq else 'MulFreq', ['xx', 'yy'][i], freq, nUBL_used, nt_used, bnside, nside_standard))
				plt.show(block=False)
			# plt.cla()
			except:
				print('Error when Plotting Pscal Results')
	except:
		print('No Plotting for Pscal Results.')
	
	if Pt_vis:
		return vis_data_dred_mfreq_pscal, autocorr_data_dred_mfreq_pscal, vis_data_dred_pscal, pt_vis, pt_sources
	else:
		return vis_data_dred_mfreq_pscal, autocorr_data_dred_mfreq_pscal, vis_data_dred_pscal

def echo(message, type=0, verbose=True):
	if verbose:
		if type == 0:
			print(message)
		elif type == 1:
			print('\n{}\n{}'.format(message, '-'*70))


def Pre_Calibration(pre_calibrate=False, pre_ampcal=False, pre_phscal=False, pre_addcal=False, comply_ps2mod_autocorr=False, Use_PsAbsCal=False, Use_AbsCal=False, Use_Fullsim_Noise=False, Precal_time_bin_temp=None, nt_used=None, nUBL_used=None, data_shape=None, cal_times=1, niter_max=50, antpairs=None, ubl_index=None, Only_AbsData=False,
                    autocorr_vis_normalized=None, fullsim_vis=None, data=None, Ni=None, pt_vis=None, pt_sources=None, used_common_ubls=None, freq=None, lsts=None, lst_offset=None, INSTRUMENT=None, Absolute_Calibration_dred_mfreq=False, mocal_time_bin=None, mocal_freq_bin=None, bnside=None, nside_standard=None, scale_noise=False, ubl_sort=None):
	if nt_used is not None:
		if nt_used != len(lsts):
			raise ValueError('nt_used doesnot match len(lsts).')
	nt_used = len(lsts)
	
	if nUBL_used is not None:
		if nUBL_used != len(used_common_ubls):
			raise ValueError('nUBL_used doesnot match len(used_common_ubls).')
	nUBL_used = len(used_common_ubls)
	
	if Only_AbsData:
		data = np.concatenate((np.real(data), np.imag(data))).astype('complex128')
		Ni = np.concatenate((Ni * 2, Ni * 2))
	#####1. antenna based calibration#######
	precal_time_bin = np.min([Precal_time_bin_temp, nt_used])
	precal_time_bin_num = (data_shape['xx'][1] / precal_time_bin) if np.mod(data_shape['xx'][1], precal_time_bin) == 0 else (data_shape['xx'][1] / precal_time_bin) + 1
	print ("Precal_time_bin: %s; \nPrecal_time_bin_num: %s" % (precal_time_bin, precal_time_bin_num))
	raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
	
	try:
		if antpairs is not None:
			used_antpairs = antpairs[abs(ubl_index['x']) - 1]
			n_usedants = np.unique(used_antpairs)
	except:
		pass
	
	#####2. re-phasing and crosstalk#######
	additive_A = np.zeros((nUBL_used, 2, nt_used, 1 + 4 * nUBL_used)).astype('complex128')
	
	# put in autocorr regardless of whats saved on disk
	for p in range(2):
		additive_A[:, p, :, 0] = fullsim_vis[:, p]
		for i in range(nUBL_used):
			additive_A[i, p, :, 1 + 4 * i + 2 * p] = 1. * autocorr_vis_normalized[p]  # [id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
			additive_A[i, p, :, 1 + 4 * i + 2 * p + 1] = 1.j * autocorr_vis_normalized[p]  # [id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
	additive_A.shape = (nUBL_used * 2 * nt_used, 1 + 4 * nUBL_used)
	
	additive_term = np.zeros_like(data)
	additive_term_incr = np.zeros_like(data)
	
	for id_t_bin in range(precal_time_bin_num):
		nt_precal_used = precal_time_bin if ((id_t_bin + 1) * precal_time_bin) <= data_shape['xx'][1] else (data_shape['xx'][1] - id_t_bin * precal_time_bin)
		print ('Nt_precal_used: %s' % nt_precal_used)
		
		additive_A_tbin = additive_A.reshape(nUBL_used, 2, nt_used, 1 + 4 * nUBL_used)[:, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used), :].reshape(nUBL_used * 2 * nt_precal_used, 1 + 4 * nUBL_used)
		
		for cal_index in range(cal_times):
			
			if pre_calibrate:
				# import omnical.calibration_omni as omni
				# raw_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
				# raw_Ni = np.copy(Ni)
				
				real_additive_A = np.concatenate((np.real(additive_A_tbin), np.imag(additive_A_tbin)), axis=0).astype('complex128')
				if pre_ampcal:  # if pre_ampcal, allow xx and yy to fit amp seperately
					n_prefit_amp = 2
					real_additive_A.shape = (2 * nUBL_used, 2, nt_precal_used, 1 + 4 * nUBL_used)
					real_additive_A_expand = np.zeros((2 * nUBL_used, 2, nt_precal_used, n_prefit_amp + 4 * nUBL_used), dtype='complex128')
					for i in range(n_prefit_amp):
						real_additive_A_expand[:, i, :, i] = real_additive_A[:, i, :, 0]
					real_additive_A_expand[..., n_prefit_amp:] = real_additive_A[..., 1:]
					real_additive_A = real_additive_A_expand
					real_additive_A.shape = (2 * nUBL_used * 2 * nt_precal_used, n_prefit_amp + 4 * nUBL_used)
				else:
					n_prefit_amp = 1
				
				additive_AtNiA = np.empty((n_prefit_amp + 4 * nUBL_used, n_prefit_amp + 4 * nUBL_used), dtype='complex128')
				if pre_addcal:
					ATNIA(real_additive_A, Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), additive_AtNiA)
					additive_AtNiAi = sla.inv(additive_AtNiA)
				else:
					real_additive_A[..., n_prefit_amp:] = 0.
					ATNIA(real_additive_A, Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), additive_AtNiA)
					additive_AtNiAi = sla.pinv(additive_AtNiA)
				
				niter = 0
				rephases = np.zeros((2, 2))
				# additive_term = np.zeros_like(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten())
				# additive_term_incr = np.zeros_like(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten())
				while (niter == 0 or la.norm(rephases) > .001 or la.norm(additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) / la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) > .001) and niter < niter_max:
					niter += 1
					
					if pre_phscal:
						cdata = get_complex_data(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used)
						for p, pol in enumerate(['xx', 'yy']):
							# rephase = omni.solve_phase_degen_fast(cdata[:, p].transpose(), cdata[:, p].transpose(), fullsim_vis[:, p].transpose(), fullsim_vis[:, p].transpose(), used_common_ubls)
							rephase = solve_phase_degen(cdata[:, p].transpose(), cdata[:, p].transpose(), fullsim_vis[:, p, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].transpose(), fullsim_vis[:, p, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].transpose(), used_common_ubls)
							rephases[p] = rephase
							if p == 0:
								print 'pre process rephase', pol, rephase,
							else:
								print pol, rephase
							cdata[:, p] *= np.exp(1.j * used_common_ubls[:, :2].dot(rephase))[:, None]
						data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(cdata).reshape(2, data_shape['xx'][0], 2, nt_precal_used).astype('complex128')
					
					additive_sol = additive_AtNiAi.dot(np.transpose(real_additive_A).dot(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten() * Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()))
					print ('>>>>>>>>>>>>>additive fitting amp', additive_sol[:n_prefit_amp])
					# additive_term_incr_tbin = additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1]))[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()
					additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = real_additive_A[:, n_prefit_amp:].dot(additive_sol[n_prefit_amp:]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
					data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] -= additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
					additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] += additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)]
					try:
						print ("additive fraction", la.norm(additive_term_incr.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()) / la.norm(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten()))
					except:
						print('No additive fraction printed.')
				
				# cadd = get_complex_data(additive_term)
				
				if pre_ampcal:
					data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(data.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) / additive_sol[:n_prefit_amp, None]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
					if comply_ps2mod_autocorr or Use_AbsCal or Use_Fullsim_Noise and not Use_PsAbsCal:
						pass
					elif not (Use_PsAbsCal and comply_ps2mod_autocorr):
						Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(Ni.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) * additive_sol[:n_prefit_amp, None] ** 2).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
					additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)] = stitch_complex_data(get_complex_data(additive_term.reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])[:, :, :, id_t_bin * precal_time_bin:(id_t_bin * precal_time_bin + nt_precal_used)].flatten(), nubl=nUBL_used, nt=nt_precal_used) / additive_sol[:n_prefit_amp, None]).reshape(2, data_shape['xx'][0], 2, nt_precal_used)
					
					print('Additive_sol: %s' % additive_sol[:n_prefit_amp])
	
	# try:
	# 	if Use_AbsCal or Use_Fullsim_Noise or (Use_PsAbsCal and comply_ps2mod_autocorr) and scale_noise:
	# 		vis_normalization = get_vis_normalization(data, stitch_complex_data(fullsim_vis), data_shape=data_shape)
	# 		Ni /= vis_normalization ** 2
	# 		print('Ni rescaled after pre_calibration, divided by %s.'%vis_normalization)
	# except:
	# 	print('Could not scale noise after pre_calibration toward data amplitude.')
	
	cadd = get_complex_data(additive_term, nubl=nUBL_used, nt=nt_used)
	
	try:
		print 'saving data to', os.path.dirname(data_filename) + '/' + INSTRUMENT + tag + datatag + vartag + '_gsmcal_n%i_bn%i_nubl%s_nt%s-mtbin%s-mfbin%s-tbin%s.npz' % (nside_standard, bnside, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none')
		np.savez(os.path.dirname(data_filename) + '/' + INSTRUMENT + tag + datatag + vartag + '_gsmcal_n%i_bn%i_%s_%s-mtbin%s-mfbin%s-tbin%s.npz' % (nside_standard, bnside, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'),
		         data=data,
		         simdata=stitch_complex_data(fullsim_vis),
		         psdata=[stitch_complex_data(vis) for vis in pt_vis],
		         pt_sources=pt_sources,
		         ubls=used_common_ubls,
		         tlist=lsts,
		         Ni=Ni,
		         freq=freq)
	except:
		print('Error when Saving Calibrated Results Package.')
	
	try:
		if plot_data_error:
			# plt.clf()
			
			cdata = get_complex_data(data, nubl=nUBL_used, nt=nt_used)
			crdata = get_complex_data(raw_data, nubl=nUBL_used, nt=nt_used)  # / (additive_sol[0] * (pre_ampcal) + (not pre_ampcal))
			cNi = get_complex_data(Ni, nubl=nUBL_used, nt=nt_used)
			
			fun = np.abs
			srt = sorted((lsts - lst_offset) % 24. + lst_offset)
			asrt = np.argsort((lsts - lst_offset) % 24. + lst_offset)
			# pncol = min(int(60. / (srt[-1] - srt[0])), 12) if nt_used > 1 else (len(ubl_sort['x']) / 2)
			# us = ubl_sort['x'][::len(ubl_sort['x']) / pncol] if len(ubl_sort['x']) / pncol >= 1 else ubl_sort['x']
			us = ubl_sort['x'][::len(ubl_sort['x']) / 24]
			figure = {}
			for p in range(2):
				for nu, u in enumerate(us):
					plt.figure(5000 + 100 * p + nu)
					# plt.subplot(5, (len(us) + 4) / 5, nu + 1)
					figure[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
					figure[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
					figure[3], = plt.plot(srt, fun(crdata[u, p][asrt]), '+', label='raw_data')
					figure[4], = plt.plot(srt, fun(cNi[u, p][asrt]) ** -.5, label='Ni')
					if pre_calibrate:
						figure[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
						data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p]))),
					else:
						data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
					plt.yscale('log')
					plt.title("%s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
					plt.ylim([-1.05 * data_range, 1.05 * data_range])
					if pre_calibrate:
						plt.legend(handles=[figure[1], figure[2], figure[3], figure[4], figure[5]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise', 'additive'], loc=0)
					else:
						plt.legend(handles=[figure[1], figure[2], figure[3], figure[4]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise'], loc=0)
					plt.savefig(
						script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Abs_Full_vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard))
					plt.show(block=False)
			
			fun = np.angle
			for p in range(2):
				for nu, u in enumerate(us):
					plt.figure(50000 + 100 * p + nu)
					# plt.subplot(5, (len(us) + 4) / 5, nu + 1)
					figure[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
					figure[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
					figure[3], = plt.plot(srt, fun(crdata[u, p][asrt]), '+', label='raw_data')
					figure[4], = plt.plot(srt, fun(cNi[u, p][asrt]), label='Ni')
					if pre_calibrate:
						figure[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
						data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p]))),
					else:
						data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(crdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
					# plt.yscale('log')
					plt.title("%s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
					plt.ylim([-1.05 * data_range, 1.05 * data_range])
					if pre_calibrate:
						plt.legend(handles=[figure[1], figure[2], figure[3], figure[4], figure[5]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise', 'additive'], loc=0)
					else:
						plt.legend(handles=[figure[1], figure[2], figure[3], figure[4]], labels=['calibrated_data', 'fullsim_vis', 'raw_data', 'noise'], loc=0)
					plt.savefig(
						script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Angle_Full_vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (INSTRUMENT, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard))
					plt.show(block=False)
			# plt.gcf().clear()
			# plt.clf()
			# plt.close()
	except:
		print('Error when Plotting Calibrated Results.')
	
	if pre_calibrate:
		if not Only_AbsData:
			return data, Ni, additive_A, additive_term, additive_sol, precal_time_bin
		else:
			return cdata, cNi, additive_A, additive_term, additive_sol, precal_time_bin
	else:
		return additive_A, additive_term, precal_time_bin


def source2file(ra, lon=21.428305555, lat=-30.72152, duration=2.0, offset=0.0, start_jd=None,
                jd_files=None, get_filetimes=False, verbose=False):
	"""
    """
	# get LST of source
	lst = RA2LST(ra, lon, lat, start_jd)
	
	# offset
	lst += offset / 60.
	
	echo("source LST (offset by {} minutes) = {} Hours".format(offset, lst), type=1, verbose=verbose)
	
	jd = None
	utc_range = None
	utc_center = None
	source_files = None
	source_utc_range = None
	
	# get JD when source is at zenith
	jd = utils.LST2JD(lst * np.pi / 12., start_jd, longitude=lon)
	echo("JD closest to zenith (offset by {} minutes): {}".format(offset, jd), type=1, verbose=verbose)
	
	# print out UTC time
	jd_duration = duration / (60. * 24 + 4.0)
	time1 = Time(jd - jd_duration / 2, format='jd').to_datetime()
	time2 = Time(jd + jd_duration / 2, format='jd').to_datetime()
	time3 = Time(jd, format='jd').to_datetime()
	utc_range = '"{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}~{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}"' \
	            ''.format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second,
	                      time2.year, time2.month, time2.day, time2.hour, time2.minute, time2.second)
	utc_center = '{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}'.format(time3.year, time3.month, time3.day,
	                                                                time3.hour, time3.minute, time3.second)
	echo('UTC time range of {} minutes is:\n{}\ncentered on {}' \
	     ''.format(duration, utc_range, utc_center), type=1, verbose=verbose)
	
	if jd_files is not None:
		
		# get files
		files = jd_files
		if len(files) == 0:
			raise AttributeError("length of jd_files is zero")
		
		# keep files with start_JD in them
		file_jds = []
		for i, f in enumerate(files):
			if str(start_jd) not in f:
				files.remove(f)
			else:
				fjd = os.path.basename(f).split('.')
				findex = fjd.index(str(start_jd))
				file_jds.append(float('.'.join(fjd[findex:findex + 2])))
		files = np.array(files)[np.argsort(file_jds)]
		file_jds = np.array(file_jds)[np.argsort(file_jds)]
		
		# get file with closest jd1 that doesn't exceed it
		jd1 = jd - jd_duration / 2
		jd2 = jd + jd_duration / 2
		
		jd_diff = file_jds - jd1
		jd_before = jd_diff[jd_diff < 0]
		if len(jd_before) == 0:
			start_index = np.argmin(np.abs(jd_diff))
		else:
			start_index = np.argmax(jd_before)
		
		# get file closest to jd2 that doesn't exceed it
		jd_diff = file_jds - jd2
		jd_before = jd_diff[jd_diff < 0]
		if len(jd_before) == 0:
			end_index = np.argmin(np.abs(jd_diff))
		else:
			end_index = np.argmax(jd_before)
		
		source_files = files[start_index:end_index + 1]
		
		echo("file(s) closest to source (offset by {} minutes) over {} min duration:\n {}" \
		     "".format(offset, duration, source_files), type=1, verbose=verbose)
		
		if get_filetimes:
			# Get UTC timerange of source in files
			uvd = UVData()
			for i, sf in enumerate(source_files):
				if i == 0:
					uvd.read_miriad(sf)
				else:
					uv = UVData()
					uv.read_miriad(sf)
					uvd += uv
			file_jds = np.unique(uvd.time_array)
			file_delta_jd = np.median(np.diff(file_jds))
			file_delta_min = file_delta_jd * (60. * 24)
			num_file_times = int(np.ceil(duration / file_delta_min))
			file_jd_indices = np.argsort(np.abs(file_jds - jd))[:num_file_times]
			file_jd1 = file_jds[file_jd_indices].min()
			file_jd2 = file_jds[file_jd_indices].max()
			
			time1 = Time(file_jd1, format='jd').to_datetime()
			time2 = Time(file_jd2, format='jd').to_datetime()
			
			source_utc_range = '"{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}~{:04d}/{:02d}/{:02d}/{:02d}:{:02d}:{:02d}"' \
			                   ''.format(time1.year, time1.month, time1.day, time1.hour, time1.minute, time1.second,
			                             time2.year, time2.month, time2.day, time2.hour, time2.minute, time2.second)
			
			# source_utc_midyear = (time1.year + time2.year) / 2.
			
			source_utc_center = '{:04d}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}'.format((time1.year + time2.year) / 2, (time1.month + time2.month) / 2, (time1.day + time2.day) / 2, (time1.hour + time2.hour) / 2, (time1.minute + time2.minute) / 2, (time1.second + time2.second) / 2)
			
			echo('UTC time range of source in files above over {} minutes is:\n{}' \
			     ''.format(duration, source_utc_range), type=1, verbose=verbose)
	
	return (lst, jd, utc_range, utc_center, source_files, source_utc_range, source_utc_center)



INSTRUMENT = ''

#####commandline inputs#####
if len(sys.argv) == 1:
	INSTRUMENT = 'hera47'
else:
	INSTRUMENT = sys.argv[1]  # 'miteor'#'mwa'#'hera-47''paper'

INSTRUMENT = 'hera47'  # 'hera47'; 'miteor'
print INSTRUMENT

tag = '-ampcal-'  # '-ampcal-' #sys.argv[2]; if use real uncalibrated data, set tag = '-ampcal-' for amplitude calibration.

AtNiA_only = False
if len(sys.argv) > 3 and sys.argv[3][:5] == 'atnia':
	AtNiA_only = True
	pixel_scheme_number = int(sys.argv[3][5:])

simulation_opt = 1

plotcoord = 'CG'
baseline_safety_factor = 2. # max_ubl = 1.4*lambda*nside_standard/baseline_safety_factor
crosstalk_type = 'autocorr'
# pixel_directory = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'

plot_pixelization = True and not AtNiA_only
plot_projection = True and not AtNiA_only
plot_data_error = True and not AtNiA_only

force_recompute = False
force_recompute_AtNiAi_eig = False
force_recompute_AtNiAi = True
force_recompute_S = False
force_recompute_SEi = False

C = 299.792458
kB = 1.3806488 * 1.e-23

try:
	__file__
except NameError:
	# script_dir = '/Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/simulate_visibilities/scripts'
	# script_dir = os.path.join(DATA_PATH, '../../../HERA_MapMaking_VisibilitySimulation/scripts')
	script_dir = os.path.join(DATA_PATH, '../scripts')
	pixel_directory = script_dir
	print ('Run IPython: %s'%script_dir)
	
else:
	script_dir = os.path.dirname(os.path.realpath(__file__))
	pixel_directory = script_dir
	print ('Run Python： %s'%script_dir)

###########################################################
################ data file and load beam #################
##########################################################
if INSTRUMENT == 'miteor':
	# 	Simulation = True
	# 	Use_SimulatedData = False
	# 	Use_Simulation_noise = False
	
	sys.stdout.flush()
	Simulation = True
	Use_SimulatedData = False
	Use_Simulation_noise = True
	From_File_Data = True
	Keep_Red = False
	Absolute_Calibration = False
	Absolute_Calibration_red = False
	Absolute_Calibration_mfreq = False
	Absolute_Calibration_dred = False
	Absolute_Calibration_dred_mfreq = False
	PointSource_AbsCal = False
	Absolute_Calibration_dred_mfreq_pscal = False
	
	Use_AbsCal = False  # Use Model calibrated noise which is just fullsim autocorr calculated noise and Model calibrated data.
	Use_PsAbsCal = False  # higher priority over Use_AbsCal and Use_Fullsim_Noise. if comply_ps2mod_autocorr then become just fullsim autocorr calculated noise.
	comply_ps2mod_autocorr = False
	Use_Fullsim_Noise = False  # Use fullsim autocorr calculated noise.
	
	Replace_Data = True
	
	seek_optimal_threshs = False and not AtNiA_only
	dynamic_precision = .2  # .1#ratio of dynamic pixelization error vs data std, in units of data, so not power
	thresh = 2  # .2#2.#.03125#
	valid_pix_thresh = 1.e-4
	nside_start = 32
	nside_standard = 32  # Determine the resolution of GSM of sky
	
	pre_calibrate = True
	tag = '-ampcal-' if pre_calibrate else ''  # '-ampcal-' #sys.argv[2]; if use real uncalibrated data, set tag = '-ampcal-' for amplitude calibration.
	pre_ampcal = ('ampcal' in tag)
	pre_phscal = True
	pre_addcal = True
	fit_for_additive = False
	nside_beamweight = 16  # Determin shape of A matrix
	
	Erase = True
	
	Add_S_diag = False
	Add_Rcond = True
	Data_Deteriorate = False
	
	S_type = 'dyS_lowadduniform_lowI'  # 'dyS_lowadduniform_lowI'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
	# 	S_type = 'dyS_lowadduniform_min18I' if Add_S_diag else 'no_use' #'dyS_lowadduniform_minI', 'dyS_lowadduniform_I', 'dyS_lowadduniform_lowI', 'dyS_lowadduniform_lowI'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
	rcond_list = 10. ** np.arange(-17., -0., 1.)
	if Data_Deteriorate:
		S_type += '-deteriorated-'
	else:
		pass
	
	Integration_Time = 2.7  # seconds
	Frequency_Bin = 0.5 * 1.e6  # Hz
	
	lat_degree = 45.2977
	lst_offset = 5.  # lsts will be wrapped around [lst_offset, 24+lst_offset]
	#	# tag = "q3AL_5_abscal"  #"q0AL_13_abscal"  #"q1AL_10_abscal"'q3_abscalibrated'#"q4AL_3_abscal"# L stands for lenient in flagging
	if 'ampcal' in tag:
		datatag = '_2016_01_20_avg'  # '_seccasa.rad'#
		vartag = '_2016_01_20_avgx100'  # ''#
	else:
		datatag = '_2016_01_20_avg2_unpollock'  # '_2016_01_20_avg_unpollock'#'_seccasa.rad'#
		vartag = '_2016_01_20_avg2_unpollock'  # '_2016_01_20_avg_unpollockx100'#''#
	#	datadir = '/home/omniscope/data/GSM_data/absolute_calibrated_data/'
	datadir = script_dir + '/../Output/'
	antpairs = None
	baseline_safety_factor = 2.  # max_ubl = 1.4*lambda*nside_standard/baseline_safety_factor
	# deal with beam: create a callable function of the form y(freq) in MHz and returns 2 by npix
	
	############################################ Load Beam and Visibility Data ###########################################
	flist = {}
	vis_freq_list = flist[0] = flist[1] = np.array([126.83333, 127.6667, 128.5000, 129.3333, 130.1667, 131.0000, 131.8333, 132.6667, 133.5000, 134.3333, 135.1667, 136.0000, 136.8333, 137.6667, 139.3333, 140.0000, 141.83333, 142.6667, 143.5000, 144.3333, 145.0000, 145.1667, 146.0000, 146.6667, 147.5000, 148.3333, 150.8333, 151.6667, 152.5000, 153.3333, 154.1667, 155.0000, 155.8333, 156.0000, 156.6667, 156.8333, 159.3333, 161.8333, 164.3333, 166.8333, 167.8333, 170.3333, 172.8333])
	freq = vis_freq_selected = 150.8333  # MHz
	
	if tag == '-ampcal-':
		tag = '%s-%f' % (INSTRUMENT, freq) + tag
	else:
		tag = '%s-%f' % (INSTRUMENT, freq)
	
	bnside = 64  # Depend on beam pattern data
	freqs = range(110, 200, 10)
	local_beam_unpol = si.interp1d(freqs, np.array([la.norm(np.loadtxt(
		script_dir + '/../data/MITEoR/beam/%s.txt' % (p), skiprows=0).reshape(
		(len(freqs), 12 * bnside ** 2, 4)), axis=-1) ** 2 for p in ['x', 'y']]).transpose(1, 0, 2), axis=0)
	Plot_Beam = True
	if Plot_Beam:
		plt.figure(0)
		# ind = np.where(beam_freqs == freq)[0][0]
		hp.mollview(10.0 * np.log10(local_beam_unpol(freq)[0, :]), title='HERA-Dipole Beam-East (%sMHz, bnside=%s)' % (freq, bnside),
					unit='dBi')
		#     hp.mollview(10.0 * np.log10(beam_E[ind,:]), title='HERA-Dipole Beam-East (%sMHz, bnside=%s)'%(beam_freqs[ind], bnside),
		#             unit='dBi')
		plt.savefig(script_dir + '/../Output/%s-dipole-Beam-east-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
		hp.mollview(10.0 * np.log10(local_beam_unpol(freq)[1, :]), title='HERA-Dipole Beam-North (%sMHz, bnside=%s)' % (freq, bnside),
					unit='dBi')
		#     hp.mollview(10.0 * np.log10(beam_N[ind,:]), title='HERA-Dipole Beam-North (%sMHz, bnside=%s)'%(beam_freqs[ind], bnside),
		#             unit='dBi')
		plt.savefig(script_dir + '/../Output/%s-dipole-Beam-north-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
		plt.show(block=False)
	# plt.gcf().clear()
	# plt.clf()
	# plt.close()
	
	time_vis_data = np.array([np.loadtxt(script_dir + '/../data/MITEoR/visibilities/%sMHz_%s%s_A.txt' % (vis_freq_selected, p, p)) for p in ['x', 'y']])
	
	vis_data = (time_vis_data[:, 1:, 1::3] + time_vis_data[:, 1:, 2::3] * 1j).astype('float64')
	var_data = time_vis_data[:, 1:, 3::3]
	
	tlist = time_vis_data[0, 1:, 0]
	tmasks = {}
	for p in ['x', 'y']:
		tmasks[p] = np.ones_like(tlist).astype(bool)
	
	tmask = tmasks['x'] & tmasks['y']
	tlist = tlist[tmask]
	nt = nt_used = len(tlist)
	jansky2kelvin = 1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * PI / (12 * nside_standard ** 2))
	nUBL = int(len(time_vis_data[0, 0, :]) / 3)
	
	ubls = {}
	for p in range(2):
		ubls[['x', 'y'][p]] = time_vis_data[p, 0, 1:].reshape((nUBL, 3))
	common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
	# manually filter UBLs
	used_common_ubls = common_ubls[la.norm(common_ubls, axis=-1) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor]  # [np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
	nUBL_used = len(used_common_ubls)
	ubl_index = {}  # stored index in each pol's ubl for the common ubls
	for p in ['x', 'y']:
		ubl_index[p] = np.zeros(nUBL_used, dtype='int')
		for i, u in enumerate(used_common_ubls):
			if u in ubls[p]:
				ubl_index[p][i] = np.argmin(la.norm(ubls[p] - u, axis=-1)) + 1
			elif -u in ubls[p]:
				ubl_index[p][i] = - np.argmin(la.norm(ubls[p] + u, axis=-1)) - 1
			else:
				raise Exception('Logical Error')
			
	redundancy = used_redundancy = {}
	for i in range(2):
		used_redundancy[i] = redundancy[i] = np.ones_like(ubl_index)
	
	print '>>>>>>Used nUBL = %i, nt = %i.' % (nUBL_used, nt_used)
	sys.stdout.flush()


elif 'hera47' in INSTRUMENT:
	Simulation = True
	Use_SimulatedData = False
	Use_Simulation_noise = False
	From_File_Data = True
	Keep_Red = False
	
	Absolute_Calibration = False
	Absolute_Calibration_red = False
	Absolute_Calibration_mfreq = False
	Absolute_Calibration_dred = False
	
	AbsCal_files = False
	Use_CASA_Calibrated_Data = True if AbsCal_files else False
	
	Absolute_Calibration_dred_mfreq = True  # The only working Model Calibration Method.
	Absolute_Calibration_dred_mfreq_byEachBsl = True # Absolute_Calibration_dred_mfreq once for AbsByUbl_bin ubls (multiprocessing).
	AbsByUbl_bin = 10 # Number of ubls for each abs_calibartion.
	AmpCal_Pro = True # ReCalibrate Amplitude over each time_bin for each frequency.
	Mocal_time_bin_temp = 10 #30; 600; (362)
	Mocal_freq_bin_temp = 600 #600; 22; 32; (64)
	Precal_time_bin_temp = 600
	mocal_time_bin = 0 # For titles, will be recalculated using mocal_time_bin_temp.
	mocal_freq_bin = 0 # For titiles, will be recalculated using mocal_freq_bin_temp.
	Fake_wgts_dred_mfreq = True  # Remove Flags for Model Calibration.
	Bandpass_Constrain = True # use constrained solver or not.
	re_cal_times = 1 # number of iteration of mfreq-calibration.
	INSTRUMENT = INSTRUMENT + ('-UB%s'%AbsByUbl_bin if Absolute_Calibration_dred_mfreq_byEachBsl else '') + ('-APro' if AmpCal_Pro else '')
	
	PointSource_AbsCal = False
	PointSource_AbsCal_SingleFreq = False if PointSource_AbsCal else False
	Absolute_Calibration_dred_mfreq_pscal = False
	
	Use_AbsCal = True if Absolute_Calibration_dred_mfreq else False  # Use Model calculated noise which is just fullsim autocorr calculated noise and data.
	Use_PsAbsCal = True if PointSource_AbsCal else False  # higher priority over Use_AbsCal and Use_Fullsim_Noise. if comply_ps2mod_autocorr then become just fullsim autocorr calculated noise. To use Pscaled data.
	comply_ps2mod_autocorr = True
	Use_Fullsim_Noise = False  # Use fullsim autocorr calculated noise.
	scale_noise = True  # rescale amplitude of noise to data amplitude.
	scale_noise_ratio = 10.**2 if scale_noise else 1.  # comparing to data amplitude.
	INSTRUMENT = INSTRUMENT + '-nsc%s'%int(scale_noise_ratio)
	
	Replace_Data = True
	
	pre_calibrate = False
	precal_time_bin = 0
	tag = '-ampcal-scale%s-'%scale_noise_ratio if pre_calibrate else '-scale%s-'%int(scale_noise_ratio)  # '-ampcal-' #sys.argv[2]; if use real uncalibrated data, set tag = '-ampcal-' for amplitude calibration.
	pre_ampcal = ('ampcal' in tag)
	pre_phscal = True
	pre_addcal = False
	fit_for_additive = False
	
	Erase = True
	
	Add_S_diag = False
	Add_Rcond = True
	
	Small_ModelData = False
	Model_Calibration = False
	
	Data_Deteriorate = False
	
	Time_Expansion_Factor = 1. if Use_SimulatedData else 1.
	Lst_Hourangle = True
	
	Conjugate_CertainBSL = False # Whether we conjugate baselines with different x and y coordinates. Different Sign.
	Conjugate_CertainBSL2 = False # Whether we conjugate baselines with different x and y coordinates. sign(Y - X)
	Conjugate_CertainBSL3 = False # Whether we conjugate baselines with different x and y coordinates. sign(Y-X) for same sign, while - for different sign.
	baseline_safety_low = 0. # Meters
	baseline_safety_factor = 0.5 # max_ubl = 1.4*lambda*nside_standard/baseline_safety_factor
	baseline_safety_xx = 0. # Meters
	baseline_safety_yy = 0. # Meters
	baseline_safety_xx_max = 200. # Meters
	baseline_safety_yy_max = 200. # Meters
	
	Real_Visibility = False # Only Use Real parts of Visibility to Calculate sky vector and only keep real part of matrixes.
	Only_AbsData = False # Whether to use Only Absolute value of the visibility.
	INSTRUMENT = INSTRUMENT + ('-AV' if Only_AbsData else '') + ('-RV' if Real_Visibility else '')
	
	Nfiles_temp = 20
	Specific_Files = True # Choose a list of Specific Data Sets.
	Specific_FileIndex_start = [51, 51] # Starting point of selected data sets. [51, 51]
	Specific_FileIndex_end = [52, 52] # Ending point of selected data sets. [51, 51]
	Parallel_Files = True # Parallel Computing for Loading Files
	Parallel_DataPolsLoad = True if not (Small_ModelData or Model_Calibration or Parallel_Files) else False # Parallel Computing for Loading Two Pols Data
	Parallel_Files = True if not Parallel_DataPolsLoad else False
	Parallel_Mulfreq_Visibility = True # Parallel Computing for Multi-Freq Visibility.
	Parallel_Mulfreq_Visibility_deep = True # Parallel Computing for Multi-Freq Visibility in functions, which is more efficient.
	Parallel_A = True # Parallel Computing for A matrix.
	Del_A = False # Whether to delete A and save A to disc or keep in memory, which can save time but cost memory.
	Parallel_AtNiA = False # Parallel Computing for AtNiA (Matrix Multiplication)
	nchunk = 1 # UseDot to Parallel but not Parallel_AtNiA.
	nchunk_AtNiA = 24 # nchunk starting number
	nchunk_AtNiA_maxcut = 6 # maximum nchunk nchunk_AtNiA_maxcut * nchunk_AtNiA
	nchunk_AtNiA_step = 0.5 # step from 0 to nchunk_AtNiA_maxcut
	UseDot = True # Whether to use numpy.dot(paralleled) to multiply matrix or numpy.einsum(not paralleled)
	
	Time_Average_preload = 1 #12 # Number of Times averaged before loaded for each file (keep tails)'
	Frequency_Average_preload = 1 #16 # Number of Frequencies averaged before loaded for each file (remove tails)'
	Select_freq = True # Use the first frequency as the selected one every Frequency_Average_preload freq-step.
	Select_time = False # Use the first time as the selected one every Time_Average_preload time-step.
	Dred_preload = False # Whether to de-redundancy before each file loaded
	inplace_preload = True # Change the self when given to the function select_average in uvdata.py.
	
	Compress_Average = True # Compress after files loaded.
	Time_Average_afterload = 1 if Compress_Average else 1
	Frequency_Average_afterload = 1 if Compress_Average else 1
	use_select_time = True
	use_select_freq = True
	
	Tmask_temp = True
	Tmask_temp_start = 21
	Tmask_temp_end = 31
	
	
	Time_Average = (Time_Average_preload if not Select_time else 1) * (Time_Average_afterload if not use_select_time else 1)
	Frequency_Average = (Frequency_Average_preload if not Select_freq else 1) * (Frequency_Average_afterload if not use_select_freq else 1)
	
	Frequency_Select = 150. # MHz, the single frequency as reference.
	RFI_Free_Thresh = 0.99 # Will be used for choosing good selected freq by ratio of RFI-Free items.
	RFI_AlmostFree_Thresh = 0.95 # Will be used for choosing good flist by ratio of RFI-Free items.
	Freq_Low = [130, 130]
	Freq_High = [185, 185]
	Bad_Freqs = [[137.5], [137.5]]
	Comply2RFI = True # Use RFI_Best as selected frequency.
	badants_append = [0, 2, 11, 14, 26, 50, 68, 84, 98, 121]
	
	Check_Dred_AFreq_ATime = False
	Tolerance = 5.e-3 # meter, Criterion for De-Redundancy
	
	Synthesize_MultiFreq = True
	Synthesize_MultiFreq_Nfreq = 17 if Synthesize_MultiFreq else 1  # temp
	Synthesize_MultiFreq_Step = 1 if Synthesize_MultiFreq else 1

	
	sys.stdout.flush()
	
	lat_degree = -30.72153  # lon='21:25:41.9' lat='-30:43:17.5' lon=21.428305555, lat=-30.72152
	lst_offset = 0.0  # lsts will be wrapped around [lst_offset, 24+lst_offset]
	
	Integration_Time = 11  # seconds
	Frequency_Bin = 101562.5 #1.625 * 1.e6  # Hz
	
	S_type = 'dyS_lowadduniform_min4I' if Add_S_diag else 'no_use'  # 'dyS_lowadduniform_minI', 'dyS_lowadduniform_I', 'dyS_lowadduniform_lowI', 'dyS_lowadduniform_lowI'#'none'#'dyS_lowadduniform_Iuniform'  #'none'# dynamic S, addlimit:additive same level as max data; lowaddlimit: 10% of max data; lowadduniform: 10% of median max data; Iuniform median of all data
	rcond_list = 10. ** np.arange(-10., 20., 1.)
	if Data_Deteriorate:
		S_type += '-deteriorated-'
	else:
		pass
	
	seek_optimal_threshs = False and not AtNiA_only
	dynamic_precision = .2  # .1#ratio of dynamic pixelization error vs data std, in units of data, so not power
	thresh = 2  # .2#2.#.03125#
	valid_pix_thresh = 10**(-4.)
	Use_BeamWeight = False # Use beam_weight for calculating valid_pix_mask.
	
	nside_start = 32  # starting point to calculate dynamic A
	nside_standard = 32  # resolution of sky, dynamic A matrix length of a row before masking.
	nside_beamweight = 32  # undynamic A matrix shape
	bnside = 64  # beam pattern data resolution
	
	#	# tag = "q3AL_5_abscal"  #"q0AL_13_abscal"  #"q1AL_10_abscal"'q3_abscalibrated'#"q4AL_3_abscal"# L stands for lenient in flagging
	if 'ampcal' in tag:
		datatag = '_2018'  # '_seccasa.rad'#
		vartag = '_2018'  # ''#
	else:
		datatag = '_2018_ampcaled'  # '_2016_01_20_avg_unpollock'#'_seccasa.rad'#
		vartag = '_2018_ampcaled'  # '_2016_01_20_avg_unpollockx100'#''#1
	datadir = script_dir + '/../Output/'
	antpairs = None
	
	#######################################################################################################
	##################################### Load Visibility Data ###########################################
	# specify model file and load into UVData, load into dictionary
	timer_loading = time.time()
	model_fname = {}
	model = {}
	mflags = {}
	mantpos = {}
	mants = {}
	model_freqs = {}
	model_times = {}
	model_lsts = {}
	model_pols = {}
	model_autos = {}
	model_autos_flags = {}
	
	data_fname = {}
	data_fname_full = {}
	dflags = {}
	data = {}
	antpos = {}
	ants = {}
	data_freqs = {}
	data_times = {}
	data_lsts = {}
	data_pols = {}
	data_autos = {}
	data_autos_flags = {}
	
	fulldflags = {}
	
	data_fnames = {}
	files_bases = {}
	file_times = {}
	file_JDays = {}
	
	autocorr_data_mfreq = {}  # np.zeros((2, Ntimes, Nfreqs)) /nfs/blender/data/jshu_li/anaconda3/envs/Cosmology_python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192287662/2458044/  /Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192114862/2458042
	autocorr_data = {}
	
	# /nfs/blender/data/jshu_li/anaconda3/envs/Cosmology_python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1193758938/2458061  /Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192114862/2458042
	Observing_Session = '/ObservingSession-1192201262/2458043/' #/nfs/blender/data/jshu_li/anaconda3/envs/Cosmology_python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192201262/2458043/  /Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192115507/2458042/
	Nfiles = min(Nfiles_temp, len(glob.glob("{0}/zen.*.*.xx.HH.uvOR".format(DATA_PATH + Observing_Session))), len(glob.glob("{0}/zen.*.*.yy.HH.uvOR".format(DATA_PATH + Observing_Session))))
	
	redundancy = [[],[]]
	model_redundancy = [[], []]
	
	flist = {}
	index_freq = {}
	
	try:
		if not Specific_Files:
			data_fnames[0] = xxfiles = sorted((glob.glob("{0}/zen.*.*.xx.HH.uvOR".format(DATA_PATH + Observing_Session))))[:Nfiles]
			data_fnames[1] = yyfiles = sorted((glob.glob("{0}/zen.*.*.yy.HH.uvOR".format(DATA_PATH + Observing_Session))))[:Nfiles]
			
			files_bases[0] = xxfile_bases = map(os.path.basename, xxfiles)
			files_bases[1] = yyfile_bases = map(os.path.basename, yyfiles)
			
			file_times[0] = xxfile_times = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:3]), xxfiles), np.float)
			file_times[1] = yyfile_times = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:3]), yyfiles), np.float)
			
			file_JDays[0] = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:2]), xxfiles), np.int)[0]
			file_JDays[1] = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:2]), yyfiles), np.int)[0]
			
			if file_JDays[0] != file_JDays[1]:
				raise ValueError('Two Pols from diffent days.')
			else:
				INSTRUMENT = INSTRUMENT + '-' + str(file_JDays[0]) + '-Nf%s'%Nfiles + ('-CjSmBSL' if Conjugate_CertainBSL else '') + ('-CjSmBSL2' if Conjugate_CertainBSL2 else '') + ('-CjBs3' if Conjugate_CertainBSL3 else '')
			print ('Nfiles: %s' % Nfiles)
			
		# elif Specific_FileIndex_end[0] <= (Nfiles + 1) and Specific_FileIndex_end[1] <= (Nfiles + 1):
		elif Specific_Files:
			data_fnames[0] = xxfiles = sorted((glob.glob("{0}/zen.*.*.xx.HH.uvOR".format(DATA_PATH + Observing_Session))))[Specific_FileIndex_start[0]:Specific_FileIndex_end[1]]
			data_fnames[1] = yyfiles = sorted((glob.glob("{0}/zen.*.*.yy.HH.uvOR".format(DATA_PATH + Observing_Session))))[Specific_FileIndex_start[1]:Specific_FileIndex_end[1]]
			
			files_bases[0] = xxfile_bases = map(os.path.basename, xxfiles)
			files_bases[1] = yyfile_bases = map(os.path.basename, yyfiles)
			
			file_times[0] = xxfile_times = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:3]), xxfiles), np.float)
			file_times[1] = yyfile_times = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:3]), yyfiles), np.float)
			
			file_JDays[0] = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:2]), xxfiles), np.int)[0]
			file_JDays[1] = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:2]), yyfiles), np.int)[0]
			
			if len(data_fnames[0]) != len(data_fnames[1]):
				raise ValueError('Two Pols have different number of data sets.')
			else:
				Nfiles = len(data_fnames[0])
				
			if file_JDays[0] != file_JDays[1]:
				raise ValueError('Two Pols from diffent days.')
			else:
				INSTRUMENT = INSTRUMENT + '-' + str(file_JDays[0]) + '-SDx%s%s'%(Specific_FileIndex_start[0], Specific_FileIndex_end[1]) + ('-CjBs' if Conjugate_CertainBSL else '') + ('-CjBs2' if Conjugate_CertainBSL2 else '') + ('-CjBs3' if Conjugate_CertainBSL3 else '')
			print ('Nfiles: %s' % Nfiles)
	except:
		pass
	
	
	badants = []
	# badants_append = [26, 84, 121]
	for p in range(2):
		for i, fbase in enumerate(files_bases[p]):
			antfname = fbase.split('.')
			antfname.pop(3)
			if 'OR' in '.'.join(antfname):
				antfname = os.path.join(DATA_PATH + Observing_Session, '.'.join(antfname)[:-2] + '.ant_metrics.json')
			else:
				antfname = os.path.join(DATA_PATH + Observing_Session, '.'.join(antfname) + '.ant_metrics.json')
			try:
				antmets = hqm.ant_metrics.load_antenna_metrics(antfname)
				badants.extend(map(lambda x: x[0], antmets['xants']))
				# badants = np.unique(badants)
				# print('Badants: %s'%str(badants))
			except:
				print('Failed to detect Badants.')
	badants = np.unique(badants)
	print('Badants before appending: %s' % str(badants))
	if len(badants_append) >= 1:
		badants = np.append(badants, badants_append)
	badants = np.unique(badants)
	print('Badants: %s' % str(badants))
	
	lst = {}
	jd = {}
	utc_range = {}
	utc_center = {}
	source_files = {}
	source_utc_range = {}
	source_utc_center = {}
	
	complist_gleam02_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/complist_gleam02.py'
	pbcorr_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/pbcorr.py'
	beamfits_path = DATA_PATH + '/HERA-47/Beam-Dipole/NF_HERA_Beams.beamfits'
	gleam02_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/gleam02'
	gleam2clfits_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/gleam02.cl.fits'
	gleam2clpbcorrfits_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/gleam02.cl.pbcorr.fits'
	gleam2clpbcorrimage_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/gleam02.cl.pbcorr.image'
	# miriad_to_uvfits_path =
	sky_image_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/sky_image.py'
	skynpz2calfits_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/skynpz2calfits.py'
	
	sandbox_nsk_script_path = DATA_PATH + '/../../hera_sandbox/nsk/scripts/'
	# AbsCal_files = False
	if AbsCal_files:
		try:
			for i in range(2):
				lst[i], jd[i], utc_range[i], utc_center[i], source_files[i], source_utc_range[i], source_utc_center[i] = {}, {}, {}, {}, {}, {}, {}
				# 	source2file(ra=30.05005, lon=21.4286, duration=4, start_jd=file_JDays[i], jd_files=data_fnames[i], get_filetimes=True, verbose=True)  # ['/Users/JianshuLi/anaconda3/envs/Cosmology-Python27/lib/python2.7/site-packages/HERA_MapMaking_VisibilitySimulation/data/ObservingSession-1192114862/2458042/zen.2458042.12552.xx.HH.uvOR']
				for id_f, f in enumerate(data_fnames[i]):
					if not os.path.exists(f + 'X'):
						lst[i][id_f], jd[i][id_f], utc_range[i][id_f], utc_center[i][id_f], source_files[i][id_f], source_utc_range[i][id_f], source_utc_center[i][id_f] = \
							source2file(ra=30.05005, lon=21.4286, duration=4, start_jd=file_JDays[i], jd_files=[data_fnames[i][id_f]], get_filetimes=True, verbose=True)
						source_utc_center_pol = source_utc_center[i][id_f]
						source_utc_range_pol = source_utc_range[i][id_f]
						os.chdir('%s' % sandbox_nsk_script_path)
						# os.system("casa --nologger --nocrashreport --nogui --agg -c complist_gleam02.py --image --freqs 100,200,1024 --cell 60arcsec --imsize 512")
						os.system("casa --nologger --nocrashreport --nogui --agg -c %s --image --freqs 100,200,1024 --cell 60arcsec --imsize 512" % complist_gleam02_path)
						# os.system("python pbcorr.py --beamfile ../../../HERA_MapMaking_VisibilitySimulation/data/HERA-47/Beam-Dipole/NF_HERA_Beams.beamfits --outdir ./ --pol -5 --time %s --ext pbcorr --lon 21.42830 --lat -30.72152 --overwrite --multiply gleam02.cl.fits"%source_utc_center[i])
						os.system("python %s %s --beamfile %s --outdir %s --pol -5 --time '%s' --ext pbcorr --lon 21.42830 --lat -30.72152 --overwrite --multiply " % (pbcorr_path, gleam2clfits_path, beamfits_path, sandbox_nsk_script_path, source_utc_center_pol))
						os.system("casa --nologger --nocrashreport --nogui --agg -c \"importfits('%s', '%s', overwrite=True)\" " % (gleam2clpbcorrfits_path, gleam2clpbcorrimage_path))
						os.system('miriad_to_uvfits.py %s' % f)
						# os.chdir('%s'%sandbox_nsk_script_path)
						# os.system("casa --nologger --nocrashreport --nogui --agg -c %s --msin %s --source %s --model_im %s --refant 53 --ex_ants '%s' --rflag --KGcal --Acal --BPcal --imsize 512 --pxsize 250 --image_mfs --image_model --plot_uvdist --niter 50 --timerange %s"  % (sky_image_path, f + '.uvfits', gleam02_path, gleam2clpbcorrimage_path, str(badants)[1:-1], source_utc_range_pol)) #must overlap with the data set
						os.system("casa --nologger --nocrashreport --nogui --agg -c %s --msin %s --source gleam02 --model_im gleam02.cl.pbcorr.image --refant 53 --ex_ants '%s' --rflag --KGcal --Acal --BPcal --imsize 512 --pxsize 250 --image_mfs --image_model --plot_uvdist --niter 50 --timerange %s" % (sky_image_path, f + '.uvfits', str(badants)[1:-1], source_utc_range_pol))
						os.system("python %s --fname %s --uv_file %s --dly_file %s --phs_file %s --amp_file %s --bp_file %s --plot_bp --plot_phs --plot_amp --plot_dlys --bp_medfilt --medfilt_kernel 13 --bp_gp_smooth --bp_gp_max_dly 200 --bp_gp_thin 4 --overwrite" % (skynpz2calfits_path, f + '.ms.abs.calfits', f, f + '.ms.K.cal.npz', f + '.ms.Gphs.cal.npz', f + '.ms.Gamp.cal.npz', f + '.ms.B.cal.npz'))
						
						os.system("omni_apply.py -p xx \
									--omnipath %s \
									--extension X \
									--overwrite \
									%s " % (f + '.ms.abs.calfits', f))
						os.system("miriad_to_uvfits.py %s" %(f + 'X'))
						print('%s: %s'%(id_f, f + 'X'))
						os.chdir('%s' % (DATA_PATH + '/../../'))
		except:
			print('Could not CASA absolute calibrate data.')
	
	
	if Use_CASA_Calibrated_Data:
		try:
			data_fnames[0] = xxfiles = sorted((glob.glob("{0}/zen.*.*.xx.HH.uvORX".format(DATA_PATH + Observing_Session))))[:Nfiles]
			data_fnames[1] = yyfiles = sorted((glob.glob("{0}/zen.*.*.yy.HH.uvORX".format(DATA_PATH + Observing_Session))))[:Nfiles]
			
			files_bases[0] = xxfile_bases = map(os.path.basename, xxfiles)
			files_bases[1] = yyfile_bases = map(os.path.basename, yyfiles)
			
			file_times[0] = xxfile_times = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:3]), xxfiles), np.float)
			file_times[1] = yyfile_times = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:3]), yyfiles), np.float)
			
			file_JDays[0] = np.array(map(lambda x: '.'.join(os.path.basename(x).split('.')[1:2]), xxfiles), np.int)[0]
			file_JDays[1] = np.array(map(lambda y: '.'.join(os.path.basename(y).split('.')[1:2]), yyfiles), np.int)[0]
			
			if file_JDays[0] != file_JDays[1]:
				raise ValueError('Two Pols from diffent days.')
			else:
				INSTRUMENT = INSTRUMENT + '-' + str(file_JDays[0])
			
			print ('Nfiles: %s' % Nfiles)
		except:
			print('Absolute-Calibrated .uvORX files not ready.')
	if not Parallel_DataPolsLoad:
		for i in range(2):
			if Small_ModelData:
				if Model_Calibration:
					
					# model = mflags = mantpos = mant = model_freqs = model_times = model_lsts = model_pols = {}
					# for i in range(2):
					model_fname[i] = os.path.join(DATA_PATH, "zen.2458042.12552.%s.HH.uvXA" % ['xx', 'yy'][i])  # /Users/JianshuLi/Documents/Miracle/Research/Cosmology/21cm Cosmology/Algorithm-Data/Data/HERA-47/Observation-1192115507/2458042/zen.2458042.13298.xx.HH.uv
					# model_fname[1] = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA") #/Users/JianshuLi/Documents/Miracle/Research/Cosmology/21cm Cosmology/Algorithm-Data/Data/HERA-47/Observation-1192114862/2458042/zen.2458042.12552.xx.HH.uv
					if i == 1:
						try:
							# data_fname[1] = os.path.join(DATA_PATH, "zen.2458043.12552.yy.HH.uvORA") #zen.2457698.40355.yy.HH.uvcA
							if not os.path.isfile(model_fname[i]):
								model_fname[1] = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
						except:
							pass
					(model[i], mflags[i], mantpos[i], mants[i], model_freqs[i], model_times[i], model_lsts[i], model_pols[i], model_autos[i], model_autos_flags[i], model_redundancy[i]) = UVData2AbsCalDict_Auto(model_fname[i], return_meta=True, Time_Average=Time_Average_preload,
					                                                                                                                                                                                              Frequency_Average=Frequency_Average_preload, Dred=Dred_preload, inplace=inplace_preload, tol=Tolerance, Select_freq=Select_freq, Select_time=Select_time, Badants=badants, Parallel_Files=Parallel_Files)
					print('model_Pol_%s is done.' % ['xx', 'yy'][i])
				# specify data file and load into UVData, load into dictionary
				# for i in range(2):
				data_fname[i] = os.path.join(DATA_PATH, "zen.2458043.12552.%s.HH.uvORA" % ['xx', 'yy'][i])  # zen.2457698.40355.xx.HH.uvcA
				if i == 1:
					try:
						# data_fname[1] = os.path.join(DATA_PATH, "zen.2458043.12552.yy.HH.uvORA") #zen.2457698.40355.yy.HH.uvcA
						if not os.path.isfile(data_fname[i]):
							data_fname[1] = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
					except:
						pass
				data_fname_full[i] = os.path.join(DATA_PATH, 'ObservingSession-1192201262/2458043/zen.2458043.12552.%s.HH.uvOR' % ['xx', 'yy'][i])
				# (data[i], dflags[i], antpos[i], ants[i], data_freqs[i], data_times[i], data_lsts[i], data_pols[i]) = hc.abscal.UVData2AbsCalDict(data_fname[i], return_meta=True)
				(data[i], dflags[i], antpos[i], ants[i], data_freqs[i], data_times[i], data_lsts[i], data_pols[i], data_autos[i], data_autos_flags[i], redundancy[i]) = UVData2AbsCalDict_Auto(data_fname[i], return_meta=True, Time_Average=Time_Average_preload,
				                                                                                                                                                                               Frequency_Average=Frequency_Average_preload, Dred=Dred_preload, inplace=inplace_preload, tol=Tolerance, Select_freq=Select_freq, Select_time=Select_time, Badants=badants, Parallel_Files=Parallel_Files)
				print('small_Pol_%s is done.' % ['xx', 'yy'][i])
				
				# for i in range(2):
				flist[i] = np.array(data_freqs[i]) / 10 ** 6
				try:
					# index_freq[i] = np.where(flist[i] == 150)[0][0]
					index_freq[i] = np.abs(Frequency_Select - flist[i]).argmin()
				#		index_freq = 512
				except:
					index_freq[i] = len(flist[i]) / 2
				
				if Replace_Data:
					if i == 0:
						findex_list = {}
						autocorr_data_mfreq_ff = {}
						data_full = {}
						dflags_full = {}
						antpos_full = {}
						ants_full = {}
						data_freqs_full = {}
						
						data_ff = {}
						dflags_ff = {}
						
					# for i in range(2):
					timer = time.time()
					(data_full[i], dflags_full[i], antpos_full[i], ants_full[i], data_freqs_full[i], data_times[i], data_lsts[i], data_pols[i], data_autos[i], data_autos_flags[i], redundancy[i]) = UVData2AbsCalDict_Auto(data_fnames[i], return_meta=True, Time_Average=Time_Average_preload,
					                                                                                                                                                                                                        Frequency_Average=Frequency_Average_preload, Dred=Dred_preload, inplace=inplace_preload, tol=Tolerance, Select_freq=Select_freq, Select_time=Select_time, Badants=badants, Parallel_Files=Parallel_Files)
					data_freqs_full[i] = data_freqs_full[i] / 1.e6
					# findex_list[i] = np.array([np.where(data_freqs_full[i] == flist[i][j])[0][0] for j in range(len(flist[i]))])
					findex_list[i] = np.unique(np.array([np.abs(data_freqs_full[i] - flist[i][j]).argmin() for j in range(len(flist[i]))]))
	
					autocorr_data_mfreq[i] = np.mean(np.array([np.abs(data_autos[i][data_autos[i].keys()[k]]) for k in range(len(data_autos[i].keys()))]), axis=0)
					print('raw_Pol_%s is done. %s seconds used.' % (['xx', 'yy'][i], time.time() - timer))
					# autocorr_data_mfreq[1] = np.mean(np.array([np.abs(uvd_yy.get_data((ants[k], ants[k]))) for k in range(Nants)]), axis=0)
					
					findex = np.abs(data_freqs_full[i] - Frequency_Select).argmin()
					# for i in range(2):
					timer = time.time()
					data_ff[i] = LastUpdatedOrderedDict()
					dflags_ff[i] = LastUpdatedOrderedDict()
					autocorr_data_mfreq_ff[i] = np.zeros(autocorr_data_mfreq[i][:, findex_list[i]].shape)
					
					for id_f in range(len(findex_list[i])):
						# autocorr_data_mfreq_ff[i] = autocorr_data_mfreq[i][:, findex_list[i]]
						if id_f <= (len(findex_list[i]) - 2):
							autocorr_data_mfreq_ff[i][:, id_f] = np.mean(autocorr_data_mfreq[i][:, findex_list[i][id_f]: findex_list[i][id_f + 1]], axis=-1)
						else:
							autocorr_data_mfreq_ff[i][:, id_f] = np.mean(autocorr_data_mfreq[i][:, findex_list[i][id_f]:], axis=-1)
					
					for id_key, key in enumerate(dflags[i].keys()):
						data_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'] = np.zeros(data_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i]].shape)
						dflags_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'] = np.zeros(dflags_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i]].shape)
						for id_f in range(len(findex_list[i])):
							if id_f <= (len(findex_list[i]) - 2):
								data_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, id_f] = np.mean(data_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i][id_f]: findex_list[i][id_f + 1]], axis=-1)
								dflags_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, id_f] = np.mean(dflags_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i][id_f]: findex_list[i][id_f + 1]], axis=-1) > 0
							else:
								data_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, id_f] = np.mean(data_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i][id_f]:], axis=-1)
								dflags_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, id_f] = np.mean(dflags_full[i][key[0], key[1], 'xx' if i == 0 else 'yy'][:, findex_list[i][id_f]:], axis=-1)
	
					print('Pol_%s is done. %s seconds used.' % (['xx', 'yy'][i], time.time() - timer))
					# del data_ff[dflags[i].keys()[id_key]]
					
					data[i] = copy.deepcopy(data_ff[i])
					dflags[i] = copy.deepcopy(dflags_ff[i])
					autocorr_data_mfreq[i] = copy.deepcopy(autocorr_data_mfreq_ff[i])
					
					del (data_ff[i])
					del (dflags_ff[i])
					del (autocorr_data_mfreq_ff[i])
					del (data_full[i])
					del (dflags_full[i])
			
			else:
				if Model_Calibration:
					
					# for i in range(2):
					model_fname[i] = "/Users/JianshuLi/Documents/Miracle/Research/Cosmology/21cm Cosmology/Algorithm-Data/Data/HERA-47/ObservingSession-1192115507/2458042/zen.2458042.12552.%s.HH.uv" % ['xx', 'yy'][i]  # /Users/JianshuLi/Documents/Miracle/Research/Cosmology/21cm Cosmology/Algorithm-Data/Data/HERA-47/Observation-1192115507/2458042/zen.2458042.13298.xx.HH.uv
					(model[i], mflags[i], mantpos[i], mants[i], model_freqs[i], model_times[i], model_lsts[i], model_pols[i], model_autos[i], model_autos_flags[i], model_redundancy[i]) = UVData2AbsCalDict_Auto(model_fname[i], return_meta=True, Time_Average=Time_Average_preload,
					                                                                                                                                                                                              Frequency_Average=Frequency_Average_preload, Dred=Dred_preload, inplace=inplace_preload, tol=Tolerance, Select_freq=Select_freq, Select_time=Select_time, Badants=badants, Parallel_Files=Parallel_Files)
					print('model_Pol_%s is done.' % ['xx', 'yy'][i])
				# specify data file and load into UVData, load into dictionary
				
				timer = time.time()
				(data[i], dflags[i], antpos[i], ants[i], data_freqs[i], data_times[i], data_lsts[i], data_pols[i], data_autos[i], data_autos_flags[i], redundancy[i]) = UVData2AbsCalDict_Auto(data_fnames[i], return_meta=True, Time_Average=Time_Average_preload,
				                                                                                                                                                                               Frequency_Average=Frequency_Average_preload, Dred=Dred_preload, inplace=inplace_preload, tol=Tolerance, Select_freq=Select_freq, Select_time=Select_time, Badants=badants, Parallel_Files=Parallel_Files)
				autocorr_data_mfreq[i] = np.mean(np.array([np.abs(data_autos[i][data_autos[i].keys()[k]]) for k in range(len(data_autos[i].keys()))]), axis=0)
				
				print('Pol_%s is done. %s seconds used.' % (['xx', 'yy'][i], time.time() - timer))
			
			if Lst_Hourangle:
				# for i in range(2):
				data_lsts[i] = set_lsts_from_time_array_hourangle(data_times[i])
				
			
			if Compress_Average:
				data[i], dflags[i], autocorr_data_mfreq[i], data_freqs[i], data_times[i], data_lsts[i] = \
					Compress_Data_by_Average(data=data[i], dflags=dflags[i], autocorr_data_mfreq=autocorr_data_mfreq[i], Contain_Autocorr=True,
				                            data_freqs=data_freqs[i], data_times=data_times[i], data_lsts=data_lsts[i], Time_Average=Time_Average_afterload, Frequency_Average=Frequency_Average_afterload, DicData=Small_ModelData, pol=['xx', 'yy'][i],  use_select_time=use_select_time, use_select_freq=use_select_freq)
	
	else:
		timer = time.time()
		pool = Pool()
		PolsData_process = [pool.apply_async(UVData2AbsCalDict_Auto, args=(data_fnames[p], None, True, True, 'miriad', True, True, Time_Average_preload, Frequency_Average_preload, Dred_preload, True, Tolerance, Select_freq, Select_time, badants, Parallel_Files)) for p in range(2)]
		PolsData = [poldata.get() for poldata in PolsData_process]
		print('Parallel_2Pols is done. %s seconds used.' % (time.time() - timer))
		
		for i in range(2):
			data[i] = PolsData[i][0]
			dflags[i] = PolsData[i][1]
			antpos[i] = PolsData[i][2]
			ants[i] = PolsData[i][3]
			data_freqs[i] = PolsData[i][4]
			data_times[i] = PolsData[i][5]
			data_lsts[i] = PolsData[i][6]
			data_pols[i] = PolsData[i][7]
			data_autos[i] = PolsData[i][8]
			data_autos_flags[i] = PolsData[i][9]
			redundancy[i] = PolsData[i][10]
			
			autocorr_data_mfreq[i] = np.mean(np.array([np.abs(data_autos[i][data_autos[i].keys()[k]]) for k in range(len(data_autos[i].keys()))]), axis=0)
			
			if Lst_Hourangle:
				# for i in range(2):
				data_lsts[i] = set_lsts_from_time_array_hourangle(data_times[i])
			
			if Compress_Average:
				data[i], dflags[i], autocorr_data_mfreq[i], data_freqs[i], data_times[i], data_lsts[i] = \
					Compress_Data_by_Average(data=data[i], dflags=dflags[i], autocorr_data_mfreq=autocorr_data_mfreq[i], Contain_Autocorr=True,
					                         data_freqs=data_freqs[i], data_times=data_times[i], data_lsts=data_lsts[i], Time_Average=Time_Average_afterload, Frequency_Average=Frequency_Average_afterload, DicData=Small_ModelData, pol=['xx', 'yy'][i], use_select_time=use_select_time, use_select_freq=use_select_freq)
		
		del(PolsData)
		del(PolsData_process)
	
	# RFI_Free_Thresh = 0.8
	# RFI_AlmostFree_Thresh = 0.6
	
	Freq_RFI_Sort = {}
	Freq_RFI_Free = {}
	Freq_RFI_Mid = {}
	Freq_RFI_AlmostFree = {}
	Freq_RFI_AlmostFree_bool = {}
	for i in range(2):
		Freq_RFI_Sort[i] = np.argsort(-np.mean(np.array(dflags[i].values()).reshape(np.array(dflags[i].values()).shape[0] * np.array(dflags[i].values()).shape[1], np.array(dflags[i].values()).shape[2]), axis=0))
		Freq_RFI_Free[i] = np.where(np.mean(np.array(dflags[i].values()).reshape(np.array(dflags[i].values()).shape[0] * np.array(dflags[i].values()).shape[1], np.array(dflags[i].values()).shape[2]), axis=0) >= RFI_Free_Thresh)[0]
		Freq_RFI_AlmostFree[i] = np.where(np.mean(np.array(dflags[i].values()).reshape(np.array(dflags[i].values()).shape[0] * np.array(dflags[i].values()).shape[1], np.array(dflags[i].values()).shape[2]), axis=0) >= RFI_AlmostFree_Thresh)[0]
		Freq_RFI_AlmostFree_bool[i] = np.mean(np.array(dflags[i].values()).reshape(np.array(dflags[i].values()).shape[0] * np.array(dflags[i].values()).shape[1], np.array(dflags[i].values()).shape[2]), axis=0) >= RFI_AlmostFree_Thresh
		if len(Freq_RFI_Free[i]) >= 1:
			Freq_RFI_Mid[i] = Freq_RFI_Free[i][len(Freq_RFI_Free[i]) / 2]
		else:
			Freq_RFI_Mid[i] = Freq_RFI_Sort[i][0]
	if Freq_RFI_Mid[0] != Freq_RFI_Mid[1]:
		Freq_RFI_Mid[1] = Freq_RFI_Mid[0]
		print('Compy y pol to x pol for Freq_RFI_Mid.')
	
	try:
		print('>>>>>>>>>>>>>Freq_RFI_Sort: %s'%str(Freq_RFI_Sort))
		print('>>>>>>>>>>>>>Freq_RFI_Free: %s'%str(Freq_RFI_Free))
		print('>>>>>>>>>>>>>Freq_RFI_AlmostFree: %s' % str(Freq_RFI_AlmostFree))
		print('>>>>>>>>>>>>>Freq_RFI_Mid: %s'%str(Freq_RFI_Mid))
	except:
		print('>>>>>>>>>>>>>No Freq_RFI_Sort printed.')
	
	# Freq_Low = [107, 107]
	# Freq_High = [190, 190]
	for i in range(2):
		# flist[i] = np.array(data_freqs[i]) / 10 ** 6
		try:
			flist[i] = np.array(data_freqs[i]) / 10 ** 6
			Freq_RFI_AlmostFree_bool[i] = Freq_RFI_AlmostFree_bool[i] * (flist[i] > Freq_Low[i]) * (flist[i] < Freq_High[i])
			for bq in Bad_Freqs[i]:
				Freq_RFI_AlmostFree_bool[i] *= (flist[i] != bq)
			flist[i] = np.array(data_freqs[i][Freq_RFI_AlmostFree_bool[i]]) / 10 ** 6
			flist[i] = flist[i][(flist[i] > Freq_Low[i]) * (flist[i] < Freq_High[i])]
		except:
			print('flist[%s] not calculated or already calculated.'%i)
		try:
			# index_freq[i] = np.where(flist[i] == 150)[0][0]
		#		index_freq = 512
			index_freq[i] = np.abs(Frequency_Select - flist[i]).argmin()
		except:
			index_freq[i] = len(flist[i]) / 2
		
		if Comply2RFI and index_freq[i] not in Freq_RFI_Free[i] and not np.mean(np.array(dflags[i].values()).reshape(np.array(dflags[i].values()).shape[0] * np.array(dflags[i].values()).shape[1], np.array(dflags[i].values()).shape[2]), axis=0)[index_freq[i]] >= RFI_AlmostFree_Thresh:
			index_freq[i] = Freq_RFI_Free[i][np.abs(index_freq[i] - Freq_RFI_Free[i]).argmin()] #Freq_RFI_Mid[i]
			print('>>>>>index_freq[%s] replaced by closest Freq_RFI_Free[%s]'%(i, i))
		elif Comply2RFI:
			print('Frequency_Select[%s] is also (almost) RFI_Free. No Replacement.'%i)
		
		try:
			data[i], dflags[i], autocorr_data_mfreq[i], data_freqs[i], data_times[i], data_lsts[i] = \
				Compress_Data_by_Average(data=data[i], dflags=dflags[i], autocorr_data_mfreq=autocorr_data_mfreq[i], Contain_Autocorr=True, use_RFI_AlmostFree_Freq=True, Freq_RFI_AlmostFree_bool=Freq_RFI_AlmostFree_bool[i],
				                         data_freqs=data_freqs[i], data_times=data_times[i], data_lsts=data_lsts[i], Time_Average=1, Frequency_Average=1, DicData=Small_ModelData, pol=['xx', 'yy'][i], use_select_time=False, use_select_freq=False)
		except:
			raise ValueError('Cannot Use_RFI_AlmostFree_Freq.')
		
	if Data_Deteriorate:
		autocorr_data_mfreq[0] = np.random.uniform(0, np.max(autocorr_data_mfreq[0]), autocorr_data_mfreq[0].shape)
		autocorr_data_mfreq[1] = np.random.uniform(0, np.max(autocorr_data_mfreq[1]), autocorr_data_mfreq[1].shape)
	else:
		pass
	
	for i in range(2):
		autocorr_data[i] = autocorr_data_mfreq[i][:, index_freq[i]]
	
	Synthesize_MultiFreq_start = [[], []]
	Synthesize_MultiFreq_end = [[], []]
	Flist_select = [[], []]
	Flist_select_index = [[], []]
	try:
		for i in range(2):
			# Synthesize_MultiFreq_Nfreq = 2 * np.min([Synthesize_MultiFreq_Nfreq / 2, index_freq[0], index_freq[1], len(flist[0]) - index_freq[0], len(flist[1] - index_freq[1]) - index_freq[1]])
			Synthesize_MultiFreq_start[i] = (index_freq[i] - Synthesize_MultiFreq_Nfreq / 2) if (index_freq[i] - Synthesize_MultiFreq_Nfreq / 2.) >= 0 else 0
			Synthesize_MultiFreq_end[i] = (index_freq[i] + Synthesize_MultiFreq_Nfreq / 2) if (index_freq[i] + Synthesize_MultiFreq_Nfreq / 2.) < len(flist[i]) else (len(flist[i]) - 1)
			Flist_select[i] = flist[i][Synthesize_MultiFreq_start[i]: Synthesize_MultiFreq_end[i] + 1: Synthesize_MultiFreq_Step]
			Flist_select_index[i] = np.arange(Synthesize_MultiFreq_start[i], Synthesize_MultiFreq_end[i] + 1, Synthesize_MultiFreq_Step)
		# noinspection PyUnboundLocalVariable
		Synthesize_MultiFreq_Nfreq = len(Flist_select[0])
	except:
		print('No Flist_select or Flist_select_index calculated.')
	
	print ('\n' + '>>>>>>>>>>>>>>>>%s minutes used for loading data' % ((time.time() - timer_loading) / 60.) + '\n')
	# tempt = data[0][37, 65, 'xx'].reshape(6, 20, 64)
	
	################# Select Frequency ####################
	timer_pre = time.time()
	# flist = {}
	# index_freq = {}
	antloc = {}
	dflags_sf = {}  # single frequency
	
	for i in range(2):
		dflags_sf[i] = LastUpdatedOrderedDict()
		for key in dflags[i].keys():
			dflags_sf[i][key] = dflags[i][key][:, index_freq[i]]
	
	# ant locations
	for i in range(2):
		antloc[i] = np.array(map(lambda k: antpos[i][k], ants[i]))
	#	antloc_yy = np.array(map(lambda k: antpos_yy[k], ants_yy))
	
	# plot sub-array HERA layout
	for j in range(2):
		plt.figure(100000 + 50 * j)
		plt.grid()
		plt.scatter(antloc[j][:, 0], antloc[j][:, 1], s=500)
		_ = [plt.text(antloc[j][i, 0] - 1, antloc[j][i, 1], str(ants[j][i]), fontsize=5, color='w') for i in range(len(antloc[j]))]
		plt.title('%s-polarization selected subarray' % ['xx', 'yy'][j])
		# plt.xlim(-30, 30)
		# plt.ylim(-30, 30)
		plt.savefig(script_dir + '/../Output/%s-Nant%s-Ant_Locations.pdf' % (INSTRUMENT, len(antloc[0])))
		plt.show(block=False)
	plt.clf()
	
	############################## Autocorrelation #################################
	More_Details = False
	if More_Details:
		xxfiles = data_fnames[0] if not Small_ModelData else data_fname_full[0]
		yyfiles = data_fnames[1] if not Small_ModelData else data_fname_full[1]
		
		# Load data for autocorrelation calculating
		uvd_xx = UVData_HR()
		uvd_xx.read_miriad(xxfiles)
		uvd_xx.ants = np.unique(np.concatenate([uvd_xx.ant_1_array, uvd_xx.ant_2_array]))
		uvd_yy = UVData_HR()
		uvd_yy.read_miriad(yyfiles)
		uvd_yy.ants = np.unique(np.concatenate([uvd_yy.ant_1_array, uvd_yy.ant_2_array]))
		
		# Get metadata
		freqs = uvd_xx.freq_array.squeeze() / 1e6
		times = uvd_xx.time_array.reshape(uvd_xx.Ntimes, uvd_xx.Nbls)[:, 0]
		jd_start = np.floor(times.min())
		Nfreqs = len(freqs)
		Ntimes = len(times)
		
		# get redundant info
		aa = hc.utils.get_aa_from_uv(uvd_xx)
		info = hc.omni.aa_to_info(aa)
		red_bls = np.array(info.get_reds())
		ants = sorted(np.unique(np.concatenate(red_bls)))
		Nants = len(ants)
		Nside = int(np.ceil(np.sqrt(Nants)))
		Yside = int(np.ceil(float(Nants) / Nside))
		
		try:
			plot_data_autocorr = False
			if plot_data_autocorr:  # at specific frequency
				### plot autos
				t_index = 0
				jd = times[t_index]
				utc = Time(jd, format='jd').datetime
				
				xlim = (-50, Nfreqs + 50)
				ylim = (-10, 30)
				
				fig, axes = plt.subplots(Yside, Nside, figsize=(14, 14), dpi=75)
				fig.subplots_adjust(wspace=0.2, hspace=0.2)
				fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=14)
				fig.tight_layout(rect=(0, 0, 1, 0.95))
				
				k = 0
				for i in range(Yside):
					for j in range(Nside):
						ax = axes[i, j]
						ax.set_xlim(xlim)
						ax.set_ylim(ylim)
						if k < Nants:
							px, = ax.plot(10 * np.log10(np.abs(uvd_xx.get_data((ants[k], ants[k]))[t_index])), color='steelblue', alpha=0.75, linewidth=3)
							py, = ax.plot(10 * np.log10(np.abs(uvd_yy.get_data((ants[k], ants[k]))[t_index])), color='darkorange', alpha=0.75, linewidth=3)
							ax.grid(True, which='both')
							ax.set_title(str(ants[k]), fontsize=14)
							if k == 0:
								ax.legend([px, py], ['East', 'North'], fontsize=12)
						else:
							ax.axis('off')
						if j != 0:
							ax.set_yticklabels([])
						else:
							[t.set_fontsize(12) for t in ax.get_yticklabels()]
							ax.set_ylabel(r'$10\cdot\log_{10}$ amplitude', fontsize=14)
						if i != Yside - 1:
							ax.set_xticklabels([])
						else:
							[t.set_fontsize(12) for t in ax.get_xticklabels()]
							ax.set_xlabel('freq channel', fontsize=14)
						
						k += 1
				plt.show(block=False)
		
		except:
			pass
		
		autocorr_data_mfreq = {}  # np.zeros((2, Ntimes, Nfreqs))
		autocorr_data_mfreq[0] = np.mean(np.array([np.abs(uvd_xx.get_data((ants[k], ants[k]))) for k in range(Nants)]), axis=0)
		autocorr_data_mfreq[1] = np.mean(np.array([np.abs(uvd_yy.get_data((ants[k], ants[k]))) for k in range(Nants)]), axis=0)
		if Data_Deteriorate:
			autocorr_data_mfreq[0] = np.random.uniform(0, np.max(autocorr_data_mfreq[0]), autocorr_data_mfreq[0].shape)
			autocorr_data_mfreq[1] = np.random.uniform(0, np.max(autocorr_data_mfreq[1]), autocorr_data_mfreq[1].shape)
		else:
			pass
		
		autocorr_data = {}
		for i in range(2):
			autocorr_data[i] = autocorr_data_mfreq[i][:, index_freq[i]]
		
		if Replace_Data and Small_ModelData:
			findex = np.abs(freqs - Frequency_Select).argmin()
			findex_list = {}
			autocorr_data_mfreq_ff = {}
			for i in range(2):
				findex_list[i] = np.array([np.where(freqs == flist[i][j])[0][0] for j in range(len(flist[i]))])
			
			data_ff = {}
			dflags_ff = {}
			for i in range(2):
				data_ff[i] = LastUpdatedOrderedDict()
				dflags_ff[i] = LastUpdatedOrderedDict()
				for id_key, key in enumerate(dflags[i].keys()):
					# key[2] = 'xx' if i == 0 else 'yy'
					data_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'] = uvd_xx.get_data((key[0], key[1]))[:, findex_list[i]] if i == 0 else uvd_yy.get_data((key[0], key[1]))[:, findex_list[i]]
					autocorr_data_mfreq_ff[i] = autocorr_data_mfreq[i][:, findex_list[i]]
					dflags_ff[i][key[0], key[1], 'xx' if i == 0 else 'yy'] = dflags[i][key]
			# del data_ff[dflags[i].keys()[id_key]]
			
			data = copy.deepcopy(data_ff)
			dflags = copy.deepcopy(dflags_ff)
			autocorr_data_mfreq = copy.deepcopy(autocorr_data_mfreq_ff)
			
			del (data_ff)
			del (dflags_ff)
			del (autocorr_data_mfreq_ff)
		# del(autocorr_data_mfreq)
		
		del (uvd_xx)
		del (uvd_yy)
		del (aa)
		del (info)
	
	
	################################################### Visibility ########################################################
	vis_data_mfreq = {}
	# vis_data_Omni_mfreq = np.array([data_omni[bslkeys] for bslkeys in data_omni.keys()], dtype='complex128').transpose((2,1,0)) if Absolute_Calibration_dred_mfreq else None
	for i in range(2):
		vis_data_mfreq[i] = np.array([data[i][bslkeys] for bslkeys in data[i].keys()], dtype='complex128').transpose((2, 1, 0))
		if Data_Deteriorate:
			# vis_data_mfreq[i] *= np.random.normal(1, 3, vis_data_mfreq[i].shape) * np.exp(np.random.normal(1, 3, vis_data_mfreq[i].shape) + np.random.normal(1, 3, vis_data_mfreq[i].shape) * 1.j)
			vis_data_mfreq[i] = np.random.uniform(-np.max(np.abs(vis_data_mfreq[i])), np.max(np.abs(vis_data_mfreq[i])), vis_data_mfreq[i].shape) * np.exp(np.random.uniform(-1, 1, vis_data_mfreq[i].shape) + np.random.uniform(-1, 1, vis_data_mfreq[i].shape) * 1.j)
		else:
			pass
	
	vis_freq_selected = freq = flist[0][index_freq[0]]  # MHz For Omni:  0:100, 16:125, 32:150, 48:175;;; For Model:  512:150MHz   Choose xx as reference
	# vis_data = np.zeros((2,vis_data_mfreq.shape[2], vis_data_xx_mfreq.shape[3]), dtype='complex128')
	vis_data = {}
	for i in range(2):
		vis_data[i] = vis_data_mfreq[i][index_freq[i], :, :]  # [pol][ freq, time, bl]
	
	# del(vis_data_mfreq)
	
	######################################### Calculate Baseline Coordinates #############################################
	# bls = odict([(x, antpos[x[0]] - antpos[x[1]]) for x in model.keys()])
	# baseline_safety_low = 3.
	# baseline_safety_factor = 2.
	# baseline_safety_xx = 10.
	# baseline_safety_yy = 30.
	# baseline_safety_xx_max = 70.
	# baseline_safety_yy_max = 70.
	bls = [[], []]
	# bls_test = [[], []]
	for i in range(2):
		bls[i] = odict()
		for x in data[i].keys():
			if (la.norm((antpos[i][x[0]] - antpos[i][x[1]])) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor) and (la.norm((antpos[i][x[0]] - antpos[i][x[1]])) / (C / freq) >= baseline_safety_low) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0] >= baseline_safety_xx) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] >= baseline_safety_yy)\
					and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0] <= baseline_safety_xx_max) and (np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] <= baseline_safety_yy_max):
				if Conjugate_CertainBSL:
					bls[i][x] = np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])
				elif Conjugate_CertainBSL2:
					bls[i][x] = np.sign(np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] - np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0]) * (antpos[i][x[0]] - antpos[i][x[1]])
				elif Conjugate_CertainBSL3:
					if np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) == 1:
						bls[i][x] = np.sign(np.abs(antpos[i][x[0]] - antpos[i][x[1]])[1] - np.abs(antpos[i][x[0]] - antpos[i][x[1]])[0]) * (antpos[i][x[0]] - antpos[i][x[1]])
					elif np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) == -1:
						bls[i][x] = np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])
				else:
					bls[i][x] = (antpos[i][x[0]] - antpos[i][x[1]])
	# bls[i] = odict([(x, np.prod(np.sign(antpos[i][x[0]] - antpos[i][x[1]])[:2]) * (antpos[i][x[0]] - antpos[i][x[1]])) for x in data[i].keys()])
	# bls[1] = odict([(y, antpos_yy[y[0]] - antpos_yy[y[1]]) for y in data_yy.keys()])
	bls = np.array(bls)
	
	bsl_coord = [[], []]
	bsl_coord_x = bsl_coord[0] = np.array([bls[0][index] for index in bls[0].keys()])
	bsl_coord_y = bsl_coord[1] = np.array([bls[1][index] for index in bls[1].keys()])
	# bsl_coord_x=bsl_coord_y=bsl_coord
	bsl_coord = np.array(bsl_coord)
	
	####################################################################################################################################################
	################################################ Unique Base Lines and Remove Redundancy ###########################################################
	
	SingleFreq = True
	MultiFreq = True
	if SingleFreq and MultiFreq:
		vis_data_dred, vis_data_dred_mfreq, redundancy_pro, dflags_dred, dflags_dred_mfreq, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                                                             data_freqs=data_freqs, Nfreqs=64, data_times=data_times, Ntimes=60,
		                                                                                                                             FreqScaleFactor=1.e6, Frequency_Select=Frequency_Select, vis_data_mfreq=vis_data_mfreq, tol=Tolerance, Badants=badants, freq=freq, nside_standard=nside_standard,
		                                                                                                                             baseline_safety_factor=baseline_safety_factor, baseline_safety_low=baseline_safety_low, baseline_safety_xx=baseline_safety_xx, baseline_safety_yy=baseline_safety_yy, baseline_safety_xx_max = baseline_safety_xx_max, baseline_safety_yy_max = baseline_safety_yy_max)
	elif MultiFreq:
		vis_data_dred_mfreq, redundancy_pro, dflags_dred_mfreq, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                                 data_freqs=None, Nfreqs=None, data_times=None, Ntimes=60, FreqScaleFactor=None,
		                                                                                                 Frequency_Select=Frequency_Select, vis_data_mfreq=vis_data_mfreq, tol=Tolerance, Badants=badants, freq=freq, nside_standard=nside_standard,
		                                                                                                 baseline_safety_factor=baseline_safety_factor, baseline_safety_low=baseline_safety_low, baseline_safety_xx=baseline_safety_xx, baseline_safety_yy=baseline_safety_yy, baseline_safety_xx_max = baseline_safety_xx_max, baseline_safety_yy_max = baseline_safety_yy_max)
	elif SingleFreq:
		vis_data_dred, redundancy_pro, dflags_dred, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                     data_freqs=data_freqs, data_times=data_times, Ntimes=60,
		                                                                                     FreqScaleFactor=1.e6, Frequency_Select=Frequency_Select, vis_data=vis_data, tol=Tolerance, Badants=badants, freq=freq, nside_standard=nside_standard,
		                                                                                     baseline_safety_factor=baseline_safety_factor, baseline_safety_low=baseline_safety_low, baseline_safety_xx=baseline_safety_xx, baseline_safety_yy=baseline_safety_yy, baseline_safety_xx_max = baseline_safety_xx_max, baseline_safety_yy_max = baseline_safety_yy_max)
	
	for i in range(2):
		if Dred_preload:
			try:
				redundancy[i] = redundancy[i] + (np.array(redundancy_pro[i]) - 1)
			except:
				raise ValueError('Redundancy from preload and afterload does not match each other')
		
		else:
			print('Redundancy_preload: %s' % len(redundancy[i]))
			redundancy[i] = redundancy_pro[i]
	
	
	Del = True
	if Del and not Small_ModelData:
		try:
			# del(red_bls)
			# del(autocorr_data_mfreq)
			# del(vis_data_mfreq)
			del (var_data_mfreq)
		
		except:
			pass
		
		try:
			if not Keep_Red:
				del (bsl_coord)
		except:
			pass
	
	sys.stdout.flush()
	
	############################### t and f ##########################
	# Using one of the two polarization, which should basically be same from choosing files
	tlist_JD = np.array(data_times[0])
	JD2SI_time = Time(data_times[0], format='jd').datetime
	tlist = np.zeros(len(data_times[0]))
	nt = len(tlist)
	nf = len(flist[0])
	for i in range(len(data_times[0])):
		tlist[i] = si_t = (JD2SI_time[i].hour * 3600. + JD2SI_time[i].minute * 60. + JD2SI_time[i].second) / 3600.
	#	tlist[i] = '%.2f' %si_t
	
	if tag == '-ampcal-':
		tag = '%s-%.2f' % (INSTRUMENT, freq) + '-%s-%s'%('Use_BeamWeight' if Use_BeamWeight else 'Use_gsmWeight', valid_pix_thresh) + tag
	else:
		tag = '%s-%.2f' % (INSTRUMENT, freq) + '-%s-%s'%('Use_BeamWeight' if Use_BeamWeight else 'Use_gsmWeight', valid_pix_thresh)
	tmasks = {}
	for id_p, p in enumerate(['x', 'y']):
		# tmasks[p] = np.ones_like(tlist).astype(bool)
		if not Synthesize_MultiFreq:
			try:
				tmasks[p] = np.mean(np.array([dflags_dred_mfreq[id_p][dflags_dred_mfreq[id_p].keys()[k]][:, index_freq[id_p]] for k in range(len(dflags_dred_mfreq[id_p].keys()))]), axis=0) == 1
			except:
				try:
					tmasks[p] = np.mean(np.array([dflags_dred[id_p][dflags_dred[id_p].keys()[k]][:] for k in range(len(dflags_dred[id_p].keys()))]), axis=0) == 1
				except:
					try:
						tmasks[p] = np.mean(np.array([dflags[id_p][dflags[id_p].keys()[k]][:, index_freq[id_p]] for k in range(len(dflags[id_p].keys()))]), axis=0) == 1
					except:
						raise ValueError('No tmasks[%s] can be calculated.'%p)
		else:
			try:
				tmasks[p] = np.prod(np.array([np.mean(np.array([dflags_dred_mfreq[id_p][dflags_dred_mfreq[id_p].keys()[k]][:, Flist_select_index[id_p][id_freqsyn]] for k in range(len(dflags_dred_mfreq[id_p].keys()))]), axis=0) == 1 for id_freqsyn in range(len(Flist_select_index[id_p]))]), axis=0) == 1
			except:
				try:
					tmasks[p] = np.prod(np.array([np.mean(np.array([dflags[id_p][dflags[id_p].keys()[k]][:, Flist_select_index[id_p][id_freqsyn]] for k in range(len(dflags[id_p].keys()))]), axis=0) == 1 for id_freqsyn in range(len(Flist_select_index[id_p]))]), axis=0) == 1
				except:
					raise ValueError('No tmasks[%s] can be calculated.' % p)
					
	# tmasks[p] = np.zeros_like(tlist).astype(bool)
	# tmasks[p][[nt/2, nt/2 + 1]] = True # Only use the median 2 times.
	
	LST_Renorm = (0.179446946 / 60.) / ((data_lsts[0][1] - data_lsts[0][0]) / Time_Average) if not Lst_Hourangle else 1.
	# for i in range(2):
	# 	data_lsts[i] = data_lsts[i] * LST_Renorm
	
	tmask = tmasks['x'] & tmasks['y']
	if Tmask_temp:
		tmask[ :np.max([0, Tmask_temp_start])] = False
		tmask[np.min([Tmask_temp_end, (len(tmask) - 1)]) :len(tmask)] = False
	try:
		print('Tmask: %s'%str(tmask))
	except:
		print('No Tmask printed.')
	if Time_Expansion_Factor == 1.:
		tlist_full = copy.deepcopy(tlist)
		lsts_full = data_lsts[0] * LST_Renorm
		tlist = tlist[tmask]
		lsts = data_lsts[0][tmask] * LST_Renorm
	else:
		tlist_full = np.arange(tlist[0], tlist[0] + (tlist[-1] - tlist[0]) * (Time_Expansion_Factor + 1), (tlist[-1] - tlist[0]) * Time_Expansion_Factor / (len(tlist) - 1))
		lsts_full = data_lsts[0] * LST_Renorm
		tlist = np.arange(tlist[0], tlist[0] + (tlist[-1] - tlist[0]) * (Time_Expansion_Factor + 1), (tlist[-1] - tlist[0]) * Time_Expansion_Factor / (len(tlist) - 1))[tmask]
		lsts = data_lsts[0][tmask] * LST_Renorm
		for j in range(len(lsts)):
			lsts_full[j] = lsts_full[0] + (lsts_full[j] - lsts_full[0]) * Time_Expansion_Factor
			lsts[j] = lsts[0] + (lsts[j] - lsts[0]) * Time_Expansion_Factor
	# lsts = np.arange(lsts[0], lsts[0] + (lsts[-1] - lsts[0]) * (Time_Expansion_Factor + 1), (lsts[-1] - lsts[0]) * Time_Expansion_Factor/(len(lsts)-1))[tmask]
	nt_used = len(tlist)
	nf_used = len(flist[0])
	jansky2kelvin = 1.e-26 * (C / freq) ** 2 / 2 / kB / (4 * PI / (12 * nside_standard ** 2))
	
	############################## Common UBL ###########################
	ubls = {}
	bls_red = {}
	# freq = 150
	#	nside_standard = 32
	# baseline_safety_factor = 0.3
	
	nBL_red = len(bsl_coord_x)
	for p in ['x', 'y']:
		# ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
		bls_red[p] = globals()['bsl_coord_' + p]
	common_bls_red = np.array([u for u in bls_red['x'] if (u in bls_red['y'] or -u in bls_red['y'])])
	
	used_common_bls_red = common_bls_red[la.norm(common_bls_red, axis=-1) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor]  # [np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
	nBL_red_used = len(used_common_bls_red)
	
	if Keep_Red:
		nUBL = len(bsl_coord_x)
		for p in ['x', 'y']:
			# ubl_filename = datadir + tag + '_%s%s_%i_%i.ubl' % (p, p, nUBL, 3)
			ubls[p] = globals()['bsl_coord_' + p]
		common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
	
	else:
		nUBL = len(bsl_coord_dred[0])
		nUBL_yy = len(bsl_coord_dred[1])
		for i in range(2):
			p = ['x', 'y'][i]
			ubls[p] = bsl_coord_dred[i]
		common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
	
	# common_ubls = np.array([u for u in ubls['x'] if (u in ubls['y'] or -u in ubls['y'])])
	# manually filter UBLs
	used_common_ubls = common_ubls[la.norm(common_ubls, axis=-1) / (C / freq) <= 1.4 * nside_standard / baseline_safety_factor]  # [np.argsort(la.norm(common_ubls, axis=-1))[10:]]     #remove shorted 10
	nUBL_used = len(used_common_ubls)
	ubl_index = {}  # stored index in each pol's ubl for the common ubls
	used_redundancy = {}
	for p in ['x', 'y']:
		ubl_index[p] = np.zeros(nUBL_used, dtype='int')
		for i, u in enumerate(used_common_ubls):
			if u in ubls[p]:
				ubl_index[p][i] = np.argmin(la.norm(ubls[p] - u, axis=-1)) + 1
			elif -u in ubls[p]:
				ubl_index[p][i] = - np.argmin(la.norm(ubls[p] + u, axis=-1)) - 1
			else:
				raise Exception('Logical Error')
	
	used_redundancy[0] = np.array(redundancy[0])[ubl_index['x'] - 1]
	used_redundancy[1] = np.array(redundancy[0])[ubl_index['y'] - 1]
	
	print('Selected Data Frequencies: ')
	print(flist[0])
	print(flist[1])
	
	print('Selected Single Frequency: %s MHz'%freq)
	
	print '\n>>>>>>>>>>Used nUBL = %s, nt = %s, nf = %s.\n' % (nUBL_used, nt_used, nf_used)
	sys.stdout.flush()
	
	######################### Beam Pattern #############################
	
	filename = script_dir + '/../data/HERA-47/Beam-Dipole/healpix_beam.fits'
	beam_E = fits.getdata(filename, extname='BEAM_E').T  # E is east corresponding to X polarization
	beam_nside = hp.npix2nside(beam_E.shape[1])
	beam_freqs = fits.getdata(filename, extname='FREQS')
	
	print('Beam Data Frequencies: ')
	print(beam_freqs)
	
	# take East pol and rotate to get North pol
	Nfreqs = len(beam_freqs)
	beam_theta, beam_phi = hp.pix2ang(64, np.arange(64 ** 2 * 12))
	# R = hp.Rotator(rot=[0,0,-np.pi/2], deg=False)
	R = hp.Rotator(rot=[-np.pi / 2, 0, 0], deg=False)
	beam_theta2, beam_phi2 = R(beam_theta, beam_phi)
	beam_N = np.array(map(lambda x: hp.get_interp_val(x, beam_theta2, beam_phi2), beam_E))
	beam_EN = np.array([beam_E, beam_N])
	beam_EN.resize(2, Nfreqs, 49152)
	
	#	# normalize each frequency to max of 1
	#	for i in range(beam_EN.shape[-2]):
	#		beam_EN[:, i, :] /= beam_EN[:, i, :].max()
	
	local_beam_unpol = si.interp1d(beam_freqs, beam_EN.transpose(1, 0, 2), axis=0)
	
	del(beam_N)
	del(beam_E)
	del(beam_EN)
	
	Plot_Beam = True
	if Plot_Beam:
		plt.figure(0)
		# ind = np.where(beam_freqs == freq)[0][0]
		hp.mollview(10.0 * np.log10(local_beam_unpol(freq)[0, :]), title='HERA-Dipole Beam-East (%sMHz, bnside=%s)' % (freq, bnside),
					unit='dBi')
		#     hp.mollview(10.0 * np.log10(beam_E[ind,:]), title='HERA-Dipole Beam-East (%sMHz, bnside=%s)'%(beam_freqs[ind], bnside),
		#             unit='dBi')
		plt.savefig(script_dir + '/../Output/%s-dipole-Beam-east-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
		hp.mollview(10.0 * np.log10(local_beam_unpol(freq)[1, :]), title='HERA-Dipole Beam-North (%sMHz, bnside=%s)' % (freq, bnside),
					unit='dBi')
		#     hp.mollview(10.0 * np.log10(beam_N[ind,:]), title='HERA-Dipole Beam-North (%sMHz, bnside=%s)'%(beam_freqs[ind], bnside),
		#             unit='dBi')
		plt.savefig(script_dir + '/../Output/%s-dipole-Beam-north-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
		plt.show(block=False)
		# plt.gcf().clear()
		# plt.clf()
		# plt.close()

print ('\n>>>>>>>>>>>>>>%s minutes used for preparing data.\n'%((time.time()-timer_pre)/60.))
sys.stdout.flush()

#################
####set up vs and beam
################
vs = sv.Visibility_Simulator()
vs.initial_zenith = np.array([0, lat_degree * PI / 180])  # self.zenithequ
beam_heal_hor_x = local_beam_unpol(freq)[0]
beam_heal_hor_y = local_beam_unpol(freq)[1]
beam_heal_equ_x = sv.rotate_healpixmap(beam_heal_hor_x, 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0])
beam_heal_equ_y = sv.rotate_healpixmap(beam_heal_hor_y, 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0])

beam_heal_equ_x_mfreq = np.zeros(0)
beam_heal_equ_y_mfreq = np.zeros(0)
if Absolute_Calibration_dred_mfreq or PointSource_AbsCal or Synthesize_MultiFreq:
	beam_heal_hor_x_mfreq = np.array([local_beam_unpol(flist[0][i])[0] for i in range(nf_used)])
	beam_heal_hor_y_mfreq = np.array([local_beam_unpol(flist[1][i])[1] for i in range(nf_used)])
	beam_heal_equ_x_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_x_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(nf_used)])
	beam_heal_equ_y_mfreq = np.array([sv.rotate_healpixmap(beam_heal_hor_y_mfreq[i], 0, PI / 2 - vs.initial_zenith[1], vs.initial_zenith[0]) for i in range(nf_used)])

Plot_Beam = True
if Plot_Beam:
	plt.figure(10)
	# ind = np.where(beam_freqs == freq)[0][0]
	hp.mollview(10.0 * np.log10(beam_heal_equ_x), title='HERA-Dipole Beam-East (%sMHz, bnside=%s)' % (freq, bnside),
				unit='dBi')
	plt.savefig(script_dir + '/../Output/%s-equ-dipole-Beam-east-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
	hp.mollview(10.0 * np.log10(beam_heal_equ_y), title='HERA-Dipole Beam-North (%sMHz, bnside=%s)' % (freq, bnside),
				unit='dBi')
	plt.savefig(script_dir + '/../Output/%s-equ-dipole-Beam-north-%.2f-bnside-%s.pdf' % (INSTRUMENT, freq, bnside))
	plt.show(block=False)

sys.stdout.flush()

########################### Delete some Input Data ##############################
Del = True
if Del:
	# del(data)
	# del(data_yy)
	try:
		# del(model)
		# del(model_yy)
		del (data_omni)
	except:
		pass

######################
####initial A to compute beam weight
#####################
sys.stdout.flush()

sys.stdout.flush()

####################################################
###beam weights using an equal pixel A matrix######
###################################################
sys.stdout.flush()

try:
	A, beam_weight = get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=False, Compute_beamweight=True, A_path='', A_got=None, A_version=1.0, AllSky=True, MaskedSky=False, Synthesize_MultiFreq=False, flist=flist, Flist_select=None, Parallel_A=Parallel_A,
                                 Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=None,
                                 ubls=ubls, used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=None, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
	
except:
	raise ValueError('No A or beam_weight calculated.')

try:
	del(A)
	print('A has been successfully deleted.')
except:
	print('A not to be deleted.')

########################################################
###################### GSM ############################
#######################################################
pca1 = hp.fitsfunc.read_map(script_dir + '/../data/gsm1.fits' + str(nside_standard))
pca2 = hp.fitsfunc.read_map(script_dir + '/../data/gsm2.fits' + str(nside_standard))
pca3 = hp.fitsfunc.read_map(script_dir + '/../data/gsm3.fits' + str(nside_standard))
components = np.loadtxt(script_dir + '/../data/components.dat')
scale_loglog = si.interp1d(np.log(components[:, 0]), np.log(components[:, 1]))
w1 = si.interp1d(components[:, 0], components[:, 2])
w2 = si.interp1d(components[:, 0], components[:, 3])
w3 = si.interp1d(components[:, 0], components[:, 4])
gsm_standard = np.exp(scale_loglog(np.log(freq))) * (w1(freq) * pca1 + w2(freq) * pca2 + w3(freq) * pca3)
if Absolute_Calibration_dred_mfreq or PointSource_AbsCal or Synthesize_MultiFreq:
	gsm_standard_mfreq = np.array([np.exp(scale_loglog(np.log(flist[0][i]))) * (w1(flist[0][i]) * pca1 + w2(flist[0][i]) * pca2 + w3(flist[0][i]) * pca3) for i in range(nf_used)])

# rotate sky map and converts to nest
equatorial_GSM_standard = np.zeros(12 * nside_standard ** 2, 'float')
print "Rotating GSM_standard and converts to nest...",

if INSTRUMENT == 'miteor':
	DecimalYear = 2013.58  # 2013, 7, 31, 16, 47, 59, 999998)
	JulianEpoch = 2013.58
elif 'hera47' in INSTRUMENT:
	DecimalYear = Time(data_times[0][0], format='jd').decimalyear + (np.mean(Time(data_times[0], format='jd').decimalyear) - Time(data_times[0][0], format='jd').decimalyear) * Time_Expansion_Factor
	JulianEpoch = Time(data_times[0][0], format='jd').jyear + (np.mean(Time(data_times[0], format='jd').jyear) - Time(data_times[0][0], format='jd').jyear) * Time_Expansion_Factor  # np.mean(Time(data_times[0], format='jd').jyear)

sys.stdout.flush()
equ_to_gal_matrix = hp.rotator.Rotator(coord='cg').mat.dot(sv.epoch_transmatrix(2000, stdtime=JulianEpoch))
ang0, ang1 = hp.rotator.rotateDirection(equ_to_gal_matrix,
										hpf.pix2ang(nside_standard, range(12 * nside_standard ** 2), nest=True))
equatorial_GSM_standard = hpf.get_interp_val(gsm_standard, ang0, ang1)
equatorial_GSM_standard_mfreq = np.zeros(0)
if Absolute_Calibration_dred_mfreq or PointSource_AbsCal or Synthesize_MultiFreq:
	equatorial_GSM_standard_mfreq = np.array([hpf.get_interp_val(gsm_standard_mfreq[i], ang0, ang1) for i in range(nf_used)])
print "done."

Del = True
if Del:
	del (pca1)
	del (pca2)
	del (pca3)
	del (w1)
	del (w2)
	del (w3)
	del (components)
	del (scale_loglog)
	del (gsm_standard)
	try:
		del (gsm_standard_mfreq)
	except:
		pass
sys.stdout.flush()


####################### Test get_A_multifreq() and Simulate_Visibility_mfreq() and De_Redundancy() ########################
# Synthesize_MultiFreq = False

Test_A_mfreq = False
AllSky = True
MaskedSky = False
Compute_A = True

if Test_A_mfreq:
	A_path_test = datadir + tag + 'test0'
	if not Synthesize_MultiFreq:
		if not MaskedSky:
			A_test, beam_weight_test = get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, Compute_beamweight=True, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq, flist=flist, Flist_select=None, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=None,
			                         ubls=ubls, used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=None, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
		elif Compute_A:
			A_test, gsm_beamweighted_test, nside_distribution_test, final_index_test, thetas_test, phis_test, sizes_test, abs_thresh_test, npix_test, valid_pix_mask_test, valid_npix_test, fake_solution_map_test, fake_solution_test = \
				get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq,
				                flist=flist, Flist_select=None, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
				                used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
		else:
			gsm_beamweighted_test, nside_distribution_test, final_index_test, thetas_test, phis_test, sizes_test, abs_thresh_test, npix_test, valid_pix_mask_test, valid_npix_test, fake_solution_map_test, fake_solution_test  = \
				get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq,
				                flist=flist, Flist_select=None, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
				                used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
	else:
		if not MaskedSky:
			A_test, beam_weight_test = get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, Compute_beamweight=True, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq, flist=flist, Flist_select=Flist_select, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
			                         ubls=ubls, used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=None, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
		elif Compute_A:
			A_test, gsm_beamweighted_test, nside_distribution_test, final_index_test, thetas_test, phis_test, sizes_test, abs_thresh_test, npix_test, valid_pix_mask_test, valid_npix_test, fake_solution_map_test, fake_solution_test  = \
				get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq,
				                flist=flist, Flist_select=Flist_select, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
				                used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
		else:
			gsm_beamweighted_test, nside_distribution_test, final_index_test, thetas_test, phis_test, sizes_test, abs_thresh_test, npix_test, valid_pix_mask_test, valid_npix_test, fake_solution_map_test, fake_solution_test  = \
				get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=Compute_A, A_path=A_path_test, A_got=None, A_version=1.0, AllSky=AllSky, MaskedSky=MaskedSky, Synthesize_MultiFreq=Synthesize_MultiFreq,
				                flist=flist, Flist_select=Flist_select, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
				                used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)





Test_SV_mfreq = False
flist_test = [flist[0][31:34], flist[1][31:34]]
if Test_SV_mfreq:
	full_sim_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	sim_vis_xx_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_xx.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	sim_vis_yy_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_yy.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	full_redabs_sim_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_redabs.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	redabs_sim_vis_xx_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_redabs_sim_xx.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	redabs_sim_vis_yy_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_redabs_sim_yy.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	full_sim_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	sim_vis_xx_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_xx_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	sim_vis_yy_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_yy_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[1]))
	
	fullsim_vis_test_sf, autocorr_vis_test_sf, autocorr_vis_normalized_test_sf = Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_sim_filename, sim_vis_xx_filename_mfreq=sim_vis_xx_filename, sim_vis_yy_filename_mfreq=sim_vis_yy_filename, Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_ubls,
	                                                                      flist=None, freq_index=None, freq=[Frequency_Select, Frequency_Select], equatorial_GSM_standard_xx=equatorial_GSM_standard, equatorial_GSM_standard_yy=equatorial_GSM_standard, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None, beam_weight=beam_weight,
	                                                                      C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
	                                                                      nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)

	fullsim_vis_red_test, autocorr_vis_red_test, autocorr_vis_normalized_red_test = Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_redabs_sim_filename, sim_vis_xx_filename_mfreq=redabs_sim_vis_xx_filename, sim_vis_yy_filename_mfreq=redabs_sim_vis_yy_filename, Force_Compute_Vis=True,  Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_bls_red,
	                                                                        flist=None, freq_index=None, freq=[Frequency_Select, Frequency_Select], equatorial_GSM_standard_xx=equatorial_GSM_standard, equatorial_GSM_standard_yy=equatorial_GSM_standard, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None,
	                                                                        beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
	                                                                        nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)

	fullsim_vis_mfreq_test, autocorr_vis_mfreq_test, autocorr_vis_mfreq_normalized_test, fullsim_vis_mfreq_sf_test, autocorr_vis_mfreq_sf_test, autocorr_vis_mfreq_normalized_sf_test = \
		Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_sim_filename_mfreq, sim_vis_xx_filename_mfreq=sim_vis_xx_filename_mfreq, sim_vis_yy_filename_mfreq=sim_vis_yy_filename_mfreq,
		                          Force_Compute_Vis=True,  Multi_freq=True, Multi_Sin_freq=True, used_common_ubls=used_common_ubls, flist=flist, freq_index=None, freq=[Frequency_Select, Frequency_Select],
		                          equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq, beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
	                                nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
	
	fullsim_vis_mfreq_test_force, autocorr_vis_mfreq_test_force, _, fullsim_vis_mfreq_sf_test_force, autocorr_vis_mfreq_sf_test_force, _ = \
		Simulate_Visibility_mfreq(script_dir=script_dir, Force_Compute_beam_GSM=True, full_sim_filename_mfreq=full_sim_filename_mfreq, sim_vis_xx_filename_mfreq=sim_vis_xx_filename_mfreq, sim_vis_yy_filename_mfreq=sim_vis_yy_filename_mfreq,
		                          Force_Compute_Vis=True, Multi_freq=True, Multi_Sin_freq=True, used_common_ubls=used_common_ubls, flist=flist_test, freq_index=None, freq=[Frequency_Select, Frequency_Select],
		                          equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq, beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
		                          nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts, tlist=tlist_JD)
	
	fullsim_vis_mfreq_test_fake, autocorr_vis_mfreq_test_fake, _, fullsim_vis_mfreq_sf_test_fake, autocorr_vis_mfreq_sf_test_fake, _ = \
		Simulate_Visibility_mfreq(script_dir=script_dir, Force_Compute_beam_GSM=True, full_sim_filename_mfreq=full_sim_filename_mfreq, sim_vis_xx_filename_mfreq=sim_vis_xx_filename_mfreq, sim_vis_yy_filename_mfreq=sim_vis_yy_filename_mfreq, Fake_Multi_freq=True,
		                          Force_Compute_Vis=True, Multi_freq=True, Multi_Sin_freq=True, used_common_ubls=used_common_ubls, flist=flist_test, freq_index=None, freq=[Frequency_Select, Frequency_Select],
		                          equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq, beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
		                          nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts, tlist=tlist_JD)

if Check_Dred_AFreq_ATime:
	SingleFreq = True
	MultiFreq = False
	if SingleFreq and MultiFreq:
		vis_data_dred, vis_data_dred_mfreq, redundancy_pro, dflags_dred, dflags_dred_mfreq, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                                                             data_freqs=data_freqs, Nfreqs=64, data_times=data_times, Ntimes=60, FreqScaleFactor=1.e6, Frequency_Select=Frequency_Select, vis_data_mfreq=fullsim_vis_mfreq.transpose(1, 3, 2, 0), tol=Tolerance)
	elif MultiFreq:
		vis_data_dred_mfreq, redundancy_pro, dflags_dred_mfreq, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                                 data_freqs=None, Nfreqs=None, data_times=None, Ntimes=60, FreqScaleFactor=None, Frequency_Select=Frequency_Select, vis_data_mfreq=fullsim_vis_mfreq.transpose(1, 3, 2, 0), tol=Tolerance)
	elif SingleFreq:
		vis_data_dred, redundancy_pro, dflags_dred, bsl_coord_dred, Ubl_list = De_Redundancy(dflags=dflags, antpos=antpos, ants=ants, SingleFreq=SingleFreq, MultiFreq=MultiFreq, Conjugate_CertainBSL=Conjugate_CertainBSL, Conjugate_CertainBSL2=Conjugate_CertainBSL2, Conjugate_CertainBSL3=Conjugate_CertainBSL3,
		                                                                                     data_freqs=data_freqs, data_times=data_times, Ntimes=nt_used, FreqScaleFactor=1.e6, Frequency_Select=Frequency_Select, vis_data=fullsim_vis_red.transpose(1, 2, 0), tol=Tolerance)
	
	try:
		print('>>>>>>>>>>> Discrepancy between fullsim_vis and vis_data_dred from fullsim_vis_dred xx: %s' % (la.norm(fullsim_vis.transpose(1, 2, 0)[0] - vis_data_dred[0])))
		print('>>>>>>>>>>> Discrepancy between fullsim_vis and vis_data_dred from fullsim_vis_dred yy: %s' % (la.norm(fullsim_vis.transpose(1, 2, 0)[1] - vis_data_dred[1])))
	except:
		raise ValueError('Cannot check De-Redundancy')
	
	
	

###############################################################################################################################################################
############################################################### Fullsky Sinfreq Visibility Simulaiton ########################################################
#############################################################################################################################################################
full_sim_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
sim_vis_xx_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_xx.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
sim_vis_yy_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_yy.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)

# Parallel_Mulfreq_Visibility_deep = True
fullsim_vis=None
autocorr_vis=None
autocorr_vis_normalized=None
fullsim_vis, autocorr_vis, autocorr_vis_normalized = Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_sim_filename, sim_vis_xx_filename_mfreq=sim_vis_xx_filename, sim_vis_yy_filename_mfreq=sim_vis_yy_filename, Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_ubls,
                                                                      flist=None, freq_index=None, freq=[freq, freq], equatorial_GSM_standard_xx=equatorial_GSM_standard, equatorial_GSM_standard_yy=equatorial_GSM_standard, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None, beam_weight=beam_weight,
                                                                      C=299.792458, nUBL_used=None, nUBL_used_mfreq=None, Parallel_Mulfreq_Visibility_deep=Parallel_Mulfreq_Visibility_deep,
                                                                      nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts_full)

fullsim_vis = fullsim_vis[:, :, tmask]
autocorr_vis = autocorr_vis[:, tmask]
autocorr_vis_normalized = autocorr_vis_normalized[:, tmask]

if plot_data_error:
	# plt.clf()
	plt.figure(30)
	plt.plot(autocorr_vis_normalized.transpose())
	plt.title('Autocorr_vis_normalized(%s-dipole-data_error-fullvis-north-%.2f_u%i_t%i_tave%s_fave%s-bnside-%s-nside_standard-%s-texp%s'%(INSTRUMENT, freq, nUBL_used+1, nt_used, Time_Average, Frequency_Average, bnside, nside_standard, Time_Expansion_Factor))
	plt.ylim([0, 2])
	plt.savefig(script_dir + '/../Output/%s-dipole-data_error-fullvis-north-%.2f_u%i_t%i_tave%s_fave%s-bnside-%s-nside_standard-%s-texp%s.pdf'%(INSTRUMENT, freq, nUBL_used+1, nt_used, Time_Average, Frequency_Average, bnside, nside_standard, Time_Expansion_Factor))
	plt.show(block=False)
# plt.gcf().clear()
# plt.clf()
# plt.close()
sys.stdout.flush()

########################################################### Sinfreq Redundant Visibility Simulation #######################################################
if Absolute_Calibration_red or Check_Dred_AFreq_ATime:
	full_redabs_sim_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_redabs.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	redabs_sim_vis_xx_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_redabs_sim_xx.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	redabs_sim_vis_yy_filename = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_redabs_sim_yy.simvis' % (INSTRUMENT, freq, nBL_red_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
	
	
	fullsim_vis_red, autocorr_vis_red = Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_redabs_sim_filename, sim_vis_xx_filename_mfreq=redabs_sim_vis_xx_filename, sim_vis_yy_filename_mfreq=redabs_sim_vis_yy_filename, Force_Compute_Vis=True,  Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_bls_red,
	                                                                        flist=None, freq_index=None, freq=[freq, freq], equatorial_GSM_standard_xx=equatorial_GSM_standard, equatorial_GSM_standard_yy=equatorial_GSM_standard, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None,
	                                                                        beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None, Parallel_Mulfreq_Visibility_deep=Parallel_Mulfreq_Visibility_deep,
	                                                                        nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts_full)
	
	
	fullsim_vis_red = fullsim_vis_red[:, :, tmask]
	autocorr_vis_red = autocorr_vis_red[:, tmask]


	if plot_data_error:
		# plt.clf()
		plt.figure(3000000)
		plt.plot(autocorr_vis_red_normalized.transpose())
		plt.title('Autocorr_vis_normalized(%s-dipole-data_error-fullvis_red-north-%.2f-bnside-%s-nside_standard-%s-texp-%s)' % (INSTRUMENT, freq, bnside, nside_standard, Time_Expansion_Factor))
		plt.ylim([0, 2])
		plt.savefig(script_dir + '/../Output/%s-dipole-data_error-fullvis_red-north-%.2f-bnside-%s-nside_standard-%s-texp-%s.pdf' % (INSTRUMENT, freq, bnside, nside_standard, Time_Expansion_Factor))
		plt.show(block=False)
	# plt.gcf().clear()
	# plt.clf()
	# plt.close()
	sys.stdout.flush()

########################################################################################################################################################
##################################################### Multifreq Visibility Simulation #################################################################

# Parallel_Mulfreq_Visibility = True

if Absolute_Calibration_dred_mfreq or Absolute_Calibration_dred or Synthesize_MultiFreq:  # Used 9.4 min. 64*9*60*12280
	if Parallel_Mulfreq_Visibility and not Parallel_Mulfreq_Visibility_deep:
		# pass
		full_sim_filename_mfreq = {}
		sim_vis_xx_filename_mfreq = {}
		sim_vis_yy_filename_mfreq = {}
		for id_f in range(len(flist[0])):
			full_sim_filename_mfreq[id_f] = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s.simvis' % (INSTRUMENT, flist[0][id_f], nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
			sim_vis_xx_filename_mfreq[id_f] = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_xx.simvis' % (INSTRUMENT, flist[0][id_f], nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
			sim_vis_yy_filename_mfreq[id_f] = script_dir + '/../Output/%s_%s_p2_u%i_t%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_yy.simvis' % (INSTRUMENT, flist[1][id_f], nUBL_used + 1, nt_used, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor)
		
		pool = Pool()
		multifreq_list_process = [pool.apply_async(Simulate_Visibility_mfreq, args=(script_dir, INSTRUMENT, full_sim_filename_mfreq[id_f], sim_vis_xx_filename_mfreq[id_f], sim_vis_yy_filename_mfreq[id_f], True, False, False, False, False, False, '',
		                                                                    None, None, [flist[0][id_f], flist[0][id_f]], equatorial_GSM_standard, equatorial_GSM_standard, None, None,
		                                                                    beam_weight, 299.792458, used_common_ubls, None, None, None, nside_standard, None, None,
		                                                                    beam_heal_equ_x, beam_heal_equ_y, None, None, lsts_full, tlist, True, 1., False)) for id_f in range(len(flist[0]))]
		multifreq_list = np.array([np.array(p.get()) for p in multifreq_list_process])
		fullsim_vis_mfreq = np.array([multifreq_list[id_f, 0] for id_f in range(len(flist[0]))]).transpose((1, 2, 3, 0))
		autocorr_vis_mfreq = np.array([multifreq_list[id_f, 1] for id_f in range(len(flist[0]))]).transpose((1, 2, 0))
		autocorr_vis_mfreq_normalized = np.array([multifreq_list[id_f, 2] for id_f in range(len(flist[0]))]).transpose((1, 2, 0))
		
		del(multifreq_list_process)
		del(multifreq_list)
		
		fullsim_vis_mfreq_sf = np.concatenate((fullsim_vis_mfreq[:, 0:1, tmask, index_freq[0]], fullsim_vis_mfreq[:, 1:2, tmask, index_freq[1]]), axis=1)
		autocorr_vis_mfreq_sf = np.concatenate((autocorr_vis_mfreq[0:1, tmask, index_freq[0]], autocorr_vis_mfreq[1:2, tmask, index_freq[1]]), axis=0)
		autocorr_vis_mfreq_sf_normalized = np.concatenate((autocorr_vis_mfreq_normalized[0:1, tmask, index_freq[0]], autocorr_vis_mfreq_normalized[1:2, tmask, index_freq[1]]), axis=0)
		try:
			if not (np.prod(fullsim_vis_mfreq_sf == fullsim_vis) and np.prod(autocorr_vis_mfreq_sf == autocorr_vis) and np.prod(autocorr_vis_mfreq_sf_normalized == autocorr_vis_normalized)):
				print ('>>>>>>>>Single Freq Visibility Simulation Not perfectly match Multi Freq Visibility Simulation')
			else:
				print ('>>>>>>>>Single-Freq Visibility Simulation Perfectly match Multi-Freq Visibility Simulation')
		except:
			print('SinFreq-MultiFreq Visibility Simulation NOT Comparared.')
	# fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized
	
	# Simulate_Visibility_mfreq(script_dir, INSTRUMENT, full_sim_filename_mfreq[id_f], sim_vis_xx_filename_mfreq[id_f], sim_vis_yy_filename_mfreq[id_f], True, False, False, False, False, False, '',
	#                      None, None, [flist[0][id_f], flist[1][id_f]], equatorial_GSM_standard, equatorial_GSM_standard, None, None,
	#                      beam_weight, 299.792458, used_common_ubls, None, None, None, nside_standard, None, None,
	#                      beam_heal_equ_x, beam_heal_equ_y, None, None, lsts_full, tlist, True, 1., False)
	else:
		full_sim_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t-full%i_tave%s_fave%s_nside%i_bnside%i_texp%s_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
		sim_vis_xx_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t-full%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_xx_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
		sim_vis_yy_filename_mfreq = script_dir + '/../Output/%s_p2_u%i_t-full%i_tave%s_fave%s_nside%i_bnside%i_texp%s_vis_sim_yy_mfreq%s-%s-%s.simvis' % (INSTRUMENT, nUBL_used + 1, nt, Time_Average, Frequency_Average, nside_standard, bnside, Time_Expansion_Factor, np.min(flist[0]), np.max(flist[0]), len(flist[1]))
		
		
		fullsim_vis_mfreq, autocorr_vis_mfreq, autocorr_vis_mfreq_normalized, fullsim_vis_mfreq_sf, autocorr_vis_mfreq_sf, autocorr_vis_mfreq_sf_normalized = \
			Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_sim_filename_mfreq, sim_vis_xx_filename_mfreq=sim_vis_xx_filename_mfreq, sim_vis_yy_filename_mfreq=sim_vis_yy_filename_mfreq, Parallel_Mulfreq_Visibility_deep=Parallel_Mulfreq_Visibility_deep,
			                          Force_Compute_Vis=False, Multi_freq=True, Multi_Sin_freq=True, used_common_ubls=used_common_ubls, flist=flist, freq_index=None, freq=[freq, freq],
			                          equatorial_GSM_standard_xx=None, equatorial_GSM_standard_yy=None, equatorial_GSM_standard_mfreq_xx=equatorial_GSM_standard_mfreq, equatorial_GSM_standard_mfreq_yy=equatorial_GSM_standard_mfreq, beam_weight=beam_weight, C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
			                          nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts_full)
		
		
		fullsim_vis_mfreq_sf = fullsim_vis_mfreq_sf[:, :, tmask]
		autocorr_vis_mfreq_sf = autocorr_vis_mfreq_sf[:, tmask]
		autocorr_vis_mfreq_sf_normalized = autocorr_vis_mfreq_sf_normalized[:, tmask]
		
		try:
			if not (np.prod(fullsim_vis_mfreq_sf == fullsim_vis) and np.prod(autocorr_vis_mfreq_sf == autocorr_vis) and np.prod(autocorr_vis_mfreq_sf_normalized == autocorr_vis_normalized)):
				print ('>>>>>>>>Single Freq Visibility Simulation Not perfectly match Multi Freq Visibility Simulation')
			else:
				print ('>>>>>>>>Single-Freq Visibility Simulation Perfectly match Multi-Freq Visibility Simulation')
		except:
			print('SinFreq-MultiFreq Visibility Simulation NOT Comparared.')
	
	

sys.stdout.flush()

##################################################################################################################################################
############################################################ Model Calibration  #################################################################
################################################################################################################################################
# Bandpass_Constrain = True
# re_cal_times = 1
# Absolute_Calibration_dred_mfreq_byEachBsl = True
# AmpCal_Pro = True
if Absolute_Calibration_dred_mfreq_byEachBsl:
	pool = Pool()
	add_Autobsl = False
	# AbsByUbl_bin = 2
	dflags_dred_mfreq_ublbin = [[], []]
	for i in range(2):
		AbsByUbl_bin_number = len(dflags_dred_mfreq[i].keys()) / AbsByUbl_bin + np.sign(np.mod(len(dflags_dred_mfreq[i].keys()), AbsByUbl_bin))
		print('AbsByUbl_bin_number-%s: %s' %(['xx', 'yy'][i], AbsByUbl_bin_number))
		dflags_dred_mfreq_ublbin[i] = [{} for id_bin in range(AbsByUbl_bin_number)]
		for id_bin in range(AbsByUbl_bin_number):
			for id_key, key in enumerate(dflags_dred_mfreq[i].keys()[id_bin * AbsByUbl_bin: np.min([(id_bin + 1) * AbsByUbl_bin, len(dflags_dred_mfreq[i].keys())])]):
				dflags_dred_mfreq_ublbin[i][id_bin][key] = dflags_dred_mfreq[i][key]
	
	AbsByUbl_list_process = [pool.apply_async(Model_Calibration_mfreq, args=(Absolute_Calibration_dred_mfreq, Absolute_Calibration_dred, Fake_wgts_dred_mfreq, re_cal_times, antpos, Bandpass_Constrain,
	                                                                         Mocal_time_bin_temp, nt, lsts_full, Mocal_freq_bin_temp, flist, fullsim_vis_mfreq[id_bin * AbsByUbl_bin: np.min([(id_bin + 1) * AbsByUbl_bin, nUBL_used])],
	                                                                         [vis_data_dred_mfreq[0][:, :, id_bin * AbsByUbl_bin: np.min([(id_bin + 1) * AbsByUbl_bin, nUBL_used])], vis_data_dred_mfreq[1][:, :, id_bin * AbsByUbl_bin: np.min([(id_bin + 1) * AbsByUbl_bin, nUBL_used])]],
	                                                                         [dflags_dred_mfreq_ublbin[0][id_bin], dflags_dred_mfreq_ublbin[1][id_bin]], add_Autobsl, autocorr_vis_mfreq, autocorr_data_mfreq, 0,
	                                                                         INSTRUMENT, used_common_ubls[None, id_bin * AbsByUbl_bin], freq, nUBL_used, bnside, nside_standard, AmpCal_Pro, id_bin)) for id_bin in range(AbsByUbl_bin_number)]
	AbsByUbl_list = np.array([p.get() for p in AbsByUbl_list_process])
	
	vis_data_dred_mfreq_abscal = np.array([np.concatenate(np.array([AbsByUbl_list[id_bin, 0][0][:, :, :].transpose((2, 0, 1)) for id_bin in range(AbsByUbl_bin_number)]), axis=0).transpose((1, 2, 0)),
	                                       np.concatenate(np.array([AbsByUbl_list[id_bin, 0][1][:, :, :].transpose((2, 0, 1)) for id_bin in range(AbsByUbl_bin_number)]), axis=0).transpose((1, 2, 0))])
	autocorr_data_dred_mfreq_abscal_multiubl = np.array([np.array([AbsByUbl_list[id_bin, 1][0] for id_bin in range(AbsByUbl_bin_number)]).transpose((1, 2, 0)),
	                                                     np.array([AbsByUbl_list[id_bin, 1][1] for id_bin in range(AbsByUbl_bin_number)]).transpose((1, 2, 0))])
	autocorr_data_dred_mfreq_abscal = np.mean(autocorr_data_dred_mfreq_abscal_multiubl, axis=-1)
	vis_data_dred_abscal = np.array([np.concatenate(np.array([AbsByUbl_list[id_bin, 2][0][:, :].transpose((1, 0)) for id_bin in range(AbsByUbl_bin_number)]), axis=0).transpose((1, 0)),
	                                 np.concatenate(np.array([AbsByUbl_list[id_bin, 2][1][:, :].transpose((1, 0)) for id_bin in range(AbsByUbl_bin_number)]), axis=0).transpose((1, 0))])
	autocorr_data_dred_abscal_multiubl = np.array([np.array([AbsByUbl_list[id_bin, 3][0] for id_bin in range(AbsByUbl_bin_number)]).transpose((1, 0)),
	                                               np.array([AbsByUbl_list[id_bin, 3][1] for id_bin in range(AbsByUbl_bin_number)]).transpose((1, 0))])
	autocorr_data_dred_abscal = np.mean(autocorr_data_dred_abscal_multiubl, axis=-1)
	mocal_time_bin_multiubl = np.array([AbsByUbl_list[id_bin, 4] for id_bin in range(AbsByUbl_bin_number)])
	mocal_time_bin = int(np.mean(mocal_time_bin_multiubl))
	mocal_freq_bin_multiubl = np.array([AbsByUbl_list[id_bin, 5] for id_bin in range(AbsByUbl_bin_number)])
	mocal_freq_bin = int(np.mean(mocal_freq_bin_multiubl))
	
	# AbsByUbl_list_process = [pool.apply_async(Model_Calibration_mfreq, args=(Absolute_Calibration_dred_mfreq, Absolute_Calibration_dred, Fake_wgts_dred_mfreq, re_cal_times, antpos, Bandpass_Constrain,
	#                                                                     Mocal_time_bin_temp, nt, lsts_full, Mocal_freq_bin_temp, flist, fullsim_vis_mfreq[None, id_ubl], [vis_data_dred_mfreq[0][:, :, None, id_ubl], vis_data_dred_mfreq[1][:, :, None, id_ubl]],
	#                                                                     [{dflags_dred_mfreq[0].keys()[id_ubl]: dflags_dred_mfreq[0][dflags_dred_mfreq[0].keys()[id_ubl]]}, {dflags_dred_mfreq[1].keys()[id_ubl]: dflags_dred_mfreq[1][dflags_dred_mfreq[1].keys()[id_ubl]]}], add_Autobsl, autocorr_vis_mfreq, autocorr_data_mfreq, 0,
	#                                                                     INSTRUMENT, used_common_ubls[None, id_ubl], freq, nUBL_used, bnside, nside_standard)) for id_ubl in range(nUBL_used)]
	# AbsByUbl_list = np.array([p.get() for p in AbsByUbl_list_process])
	#
	# vis_data_dred_mfreq_abscal = np.array([np.array([AbsByUbl_list[id_ubl, 0][0][:, :, 0] for id_ubl in range(nUBL_used)]).transpose((1, 2, 0)), np.array([AbsByUbl_list[id_ubl, 0][1][:, :, 0] for id_ubl in range(nUBL_used)]).transpose((1, 2, 0))])
	# autocorr_data_dred_mfreq_abscal_multiubl = np.array([np.array([AbsByUbl_list[id_ubl, 1][0] for id_ubl in range(nUBL_used)]).transpose((1, 2, 0)), np.array([AbsByUbl_list[id_ubl, 1][1] for id_ubl in range(nUBL_used)]).transpose((1, 2, 0))])
	# autocorr_data_dred_mfreq_abscal = np.mean(autocorr_data_dred_mfreq_abscal_multiubl, axis=-1)
	# vis_data_dred_abscal = np.array([np.array([AbsByUbl_list[id_ubl, 2][0][:, 0] for id_ubl in range(nUBL_used)]).transpose((1, 0)), np.array([AbsByUbl_list[id_ubl, 2][1][:, 0] for id_ubl in range(nUBL_used)]).transpose((1, 0))])
	# autocorr_data_dred_abscal_multiubl = np.array([np.array([AbsByUbl_list[id_ubl, 3][0] for id_ubl in range(nUBL_used)]).transpose((1, 0)), np.array([AbsByUbl_list[id_ubl, 3][1] for id_ubl in range(nUBL_used)]).transpose((1, 0))])
	# autocorr_data_dred_abscal = np.mean(autocorr_data_dred_abscal_multiubl, axis=-1)
	# mocal_time_bin_multiubl = np.array([AbsByUbl_list[id_ubl, 4] for id_ubl in range(nUBL_used)])
	# mocal_time_bin = int(np.mean(mocal_time_bin_multiubl))
	# mocal_freq_bin_multiubl = np.array([AbsByUbl_list[id_ubl, 5] for id_ubl in range(nUBL_used)])
	# mocal_freq_bin = int(np.mean(mocal_freq_bin_multiubl))
	
	try:
		del(AbsByUbl_list_process)
		del(AbsByUbl_list)
	except:
		print('AbsByUbl multiprocess not deleted.')
	
	# Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq, Absolute_Calibration_dred, Fake_wgts_dred_mfreq, re_cal_times, antpos, Bandpass_Constrain,
	#                         Mocal_time_bin_temp, nt, lsts_full, Mocal_freq_bin_temp, flist, fullsim_vis_mfreq[None, id_ubl], [vis_data_dred_mfreq[0][:, :, None, id_ubl], vis_data_dred_mfreq[1][:, :, None, id_ubl]],
	#                         [{dflags_dred_mfreq[0].keys()[id_ubl]: dflags_dred_mfreq[0][dflags_dred_mfreq[0].keys()[id_ubl]]}, {dflags_dred_mfreq[1].keys()[id_ubl]: dflags_dred_mfreq[1][dflags_dred_mfreq[1].keys()[id_ubl]]}], add_Autobsl, autocorr_vis_mfreq, autocorr_data_mfreq, 0, INSTRUMENT, used_common_ubls[None, id_ubl], freq, nUBL_used, bnside, nside_standard)
	#
	# Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq=False, Absolute_Calibration_dred=False, Fake_wgts_dred_mfreq=False, re_cal_times=1, antpos=None, Bandpass_Constrain=True,
	#                         Mocal_time_bin_temp=None, nt_used=None, lsts=None, Mocal_freq_bin_temp=None, flist=None, fullsim_vis_mfreq=None, vis_data_dred_mfreq=None, dflags_dred_mfreq=None, add_Autobsl=False, autocorr_vis_mfreq=None, autocorr_data_mfreq=None, bl_dred_mfreq_select=8,
	#                         INSTRUMENT=None, used_common_ubls=None, freq=None, nUBL_used=None, bnside=None, nside_standard=None)

else:
	try:
		bl_dred_mfreq_select = Ubl_list[0][0].index([bls[0].keys().index((24, 53, 'xx'))])
	except:
		try:
			bl_dred_mfreq_select = Ubl_list[0][0].index([bls[0].keys().index((38, 52, 'xx'))])
		except:
			try:
				bl_dred_mfreq_select = Ubl_list[0][0].index([bls[0].keys().index((24, 25, 'xx'))])
			except:
				print('Use 0 as selected bsl.')
				bl_dred_mfreq_select = 0
	if Absolute_Calibration_dred_mfreq or Absolute_Calibration_dred:
		vis_data_dred_mfreq_abscal, autocorr_data_dred_mfreq_abscal, vis_data_dred_abscal, autocorr_data_dred_abscal, mocal_time_bin, mocal_freq_bin = \
			Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, Absolute_Calibration_dred=Absolute_Calibration_dred, Fake_wgts_dred_mfreq=Fake_wgts_dred_mfreq, Bandpass_Constrain=Bandpass_Constrain, re_cal_times=re_cal_times,
			                        antpos=antpos, Mocal_time_bin_temp=Mocal_time_bin_temp, nt_used=nt, lsts=lsts_full, Mocal_freq_bin_temp=Mocal_freq_bin_temp, flist=flist,
			                        fullsim_vis_mfreq=fullsim_vis_mfreq, vis_data_dred_mfreq=vis_data_dred_mfreq, dflags_dred_mfreq=dflags_dred_mfreq, add_Autobsl=False, autocorr_vis_mfreq=autocorr_vis_mfreq, autocorr_data_mfreq=autocorr_data_mfreq, bl_dred_mfreq_select=bl_dred_mfreq_select, AmpCal_Pro=AmpCal_Pro)
		


sys.stdout.flush()

#####################################################################################################
################################# Noise and Vis Data Loading #######################################
###################################################################################################
Recal_IntegrationTime = False
Recal_FrequencyBin = False
if len(tlist) >= 2 and 'hera47' in INSTRUMENT and Recal_IntegrationTime:
	Time_seperation_real = np.array([3600. * np.abs(tlist[i + 1] - tlist[i]) for i in range(len(tlist) - 1)])  # in second
elif 'hera47' in INSTRUMENT:
	Time_seperation_real = 11  # second
elif 'miteor' in INSTRUMENT:
	Time_seperation_real = 2.7  # second

if len(flist) >= 2 and 'hera47' in INSTRUMENT and Recal_FrequencyBin:
	Frequency_gap_real = np.array([1.e6 * np.abs(flist[0][i + 1] - flist[0][i]) for i in range(len(flist[0]) - 1)])  # Hz
elif 'hera47' in INSTRUMENT:
	Frequency_gap_real = 101562.5  # Hz
elif 'miteor' in INSTRUMENT:
	Frequency_gap_real = 0.5 * 1.e6  # Hz

# Integration_Time = np.mean(Time_seperation_real) / ((Time_Average_preload if Select_time else 1) * (Time_Average_afterload if use_select_time else 1))
# Frequency_Bin = np.mean(Frequency_gap_real) / ((Frequency_Average_preload if Select_freq else 1) * (Frequency_Average_afterload if use_select_freq else 1))
try:
	if len(Time_seperation_real) > 1:
		Integration_Time = Counter(Time_seperation_real).most_common(1)[0][0] / ((Time_Average_preload if Select_time else 1) * (Time_Average_afterload if use_select_time else 1))
except:
	Integration_Time = Time_seperation_real * ((Time_Average_preload if not Select_time else 1) * (Time_Average_afterload if not use_select_time else 1))

try:
	if len(Frequency_gap_real) > 1:
		Frequency_Bin = Counter(Frequency_gap_real).most_common(1)[0][0] / ((Frequency_Average_preload if Select_freq else 1) * (Frequency_Average_afterload if use_select_freq else 1))
except:
	Frequency_Bin = Frequency_gap_real * ((Frequency_Average_preload if not Select_freq else 1) * (Frequency_Average_afterload if not use_select_freq else 1))

try:
	print('>>>>>>>>>>>>>>>Integration_Time: %s\n>>>>>>>>>>>>>>>Frequency_Bin: %s' % (Integration_Time, Frequency_Bin))
except:
	print('>>>>>>>>>>>>>>>Integration_Time and Frequency_Bin are not printed.')

Calculate_SimulationData_Noise = True
Calculate_Data_Noise = True

# scale_noise = True
# Use_AbsCal = False

noise = {}
noise_data = {}
if Calculate_SimulationData_Noise:
	if Keep_Red:
		noise['x'] = np.array([np.random.normal(0, autocorr_vis_red[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
		noise['y'] = np.array([np.random.normal(0, autocorr_vis_red[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
	elif not scale_noise:
		noise['x'] = np.array([(np.random.normal(0, autocorr_vis[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
		noise['y'] = np.array([(np.random.normal(0, autocorr_vis[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
	else:
		noise['x'] = np.array([(np.random.normal(0, autocorr_vis[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
		noise['y'] = np.array([(np.random.normal(0, autocorr_vis[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
	
	#	N_acu = {}
	#	N_acu['x'] = np.outer(noise['x'], noise['x'].T)
	#	N_acu['y'] = np.outer(noise['y'], noise['y'].T)
	N = {}
	N['x'] = noise['x'] * noise['x']
	N['y'] = noise['y'] * noise['y']
	
	sim_var_xx_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_sim_xx.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
	sim_var_yy_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_sim_yy.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
	if not os.path.isfile(sim_var_xx_filename):
		N['x'].astype('float64').tofile(sim_var_xx_filename)
	if not os.path.isfile(sim_var_yy_filename):
		N['y'].astype('float64').tofile(sim_var_yy_filename)
	
	Del = True
	if Del:
		del (noise)
		del (N)

if Calculate_Data_Noise:
	if INSTRUMENT == 'miteor':
		noise_data['x'] = (var_data[0].flatten()) ** 0.5
		noise_data['y'] = (var_data[1].flatten()) ** 0.5
	
	elif 'hera47' in INSTRUMENT:
		if Keep_Red:
			noise_data['x'] = np.array([np.random.normal(0, autocorr_data[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Not Absolute Calibrated
			noise_data['y'] = np.array([np.random.normal(0, autocorr_data[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
		else:
			if not scale_noise:
				if Use_AbsCal:
					noise_data['x'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[0]))], dtype='float64').flatten()  # Absolute Calibrated
					noise_data['y'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[1]))], dtype='float64').flatten()
				else:
					noise_data['x'] = np.array([(np.random.normal(0, autocorr_data[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Absolute Calibrated
					noise_data['y'] = np.array([(np.random.normal(0, autocorr_data[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
			else:
				if Use_AbsCal:
					noise_data['x'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[0]))], dtype='float64').flatten()  # Absolute Calibrated
					noise_data['y'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[1]))], dtype='float64').flatten()
				else:
					noise_data['x'] = np.array([(np.random.normal(0, autocorr_data[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Absolute Calibrated
					noise_data['y'] = np.array([(np.random.normal(0, autocorr_data[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
	
	#	N_data_acu = {}
	#	N_data_acu['x'] = np.outer(noise_data['x'], noise_data['x'].T)
	#	N_data_acu['y'] = np.outer(noise_data['y'], noise_data['y'].T)
	N_data = {}
	N_data['x'] = noise_data['x'] * noise_data['x']
	N_data['y'] = noise_data['y'] * noise_data['y']
	
	Store_Data_Noise = True
	Re_Save = False
	
	if Store_Data_Noise:
		data_var_xx_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_xx.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
		data_var_yy_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_yy.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
		if not os.path.isfile(data_var_xx_filename):
			N_data['x'].astype('float64').tofile(data_var_xx_filename)
			print('N_data[x] saved to disc.')
		elif Re_Save:
			N_data['x'].astype('float64').tofile(data_var_xx_filename)
			print('N_data[x] saved to disc again.')
		else:
			print('N_data[x] not saved to disc again.')
		
		if not os.path.isfile(data_var_yy_filename):
			N_data['y'].astype('float64').tofile(data_var_yy_filename)
			print('N_data[y] saved to disc.')
		elif Re_Save:
			N_data['y'].astype('float64').tofile(data_var_yy_filename)
			print('N_data[y] saved to disc again.')
		else:
			print('N_data[x] not saved to disc again.')
	
	Del = True
	if Del:
		del (noise_data)

sys.stdout.flush()

########################################################################################################################
########################################## Point Source Calibration  ##################################################
######################################################################################################################
southern_points = {'hyd': {'ra': '09:18:05.7', 'dec': '-12:05:44'},
                   'cen': {'ra': '13:25:27.6', 'dec': '-43:01:09'},
                   'cyg': {'ra': '19:59:28.3', 'dec': '40:44:02'},
                   'pic': {'ra': '05:19:49.7', 'dec': '-45:46:44'},
                   'vir': {'ra': '12:30:49.4', 'dec': '12:23:28'},
                   'for': {'ra': '03:22:41.7', 'dec': '-37:12:30'},
                   'sag': {'ra': '17:45:40.045', 'dec': '-29:0:27.9'},
                   'cas': {'ra': '23:23:26', 'dec': '58:48:00'},
                   'crab': {'ra': '5:34:31.97', 'dec': '22:00:52.1'}}

data_var_xx_filename_pscal = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_xx_pscal.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL, nt, nside_standard, bnside)
data_var_yy_filename_pscal = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_yy_pscal.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL, nt, nside_standard, bnside)

if PointSource_AbsCal:
	vis_data_dred_mfreq_pscal, autocorr_data_dred_mfreq_pscal, vis_data_dred_pscal, pt_vis, pt_sources = \
		PointSource_Calibration(data_var_xx_filename_pscal=data_var_xx_filename_pscal, data_var_yy_filename_pscal=data_var_yy_filename_pscal, PointSource_AbsCal=PointSource_AbsCal, PointSource_AbsCal_SingleFreq=PointSource_AbsCal_SingleFreq, Pt_vis=True, From_AbsCal=False, comply_ps2mod_autocorr=comply_ps2mod_autocorr,
		                        southern_points=southern_points, phase_degen_niter_max=30,
		                        index_freq=index_freq, freq=freq, flist=flist, lsts=lsts_full, tlist=tlist_full, tmask=tmask, vis_data_dred_mfreq=vis_data_dred_mfreq, vis_data_dred_mfreq_abscal=None, autocorr_data_mfreq=autocorr_data_mfreq, autocorr_data_dred_mfreq_abscal=None, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq,
		                        beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, Integration_Time=Integration_Time, Frequency_Bin=Frequency_Bin, used_redundancy=used_redundancy, nt=nt, nUBL=nUBL, ubls=ubls, bl_dred_mfreq_pscal_select=8, dflags_dred_mfreq=dflags_dred_mfreq,
		                        INSTRUMENT=INSTRUMENT, used_common_ubls=used_common_ubls, nUBL_used=nUBL_used, nt_used=nt_used, bnside=bnside, nside_standard=nside_standard, scale_noise=scale_noise, scale_noise_ratio=scale_noise_ratio)

sys.stdout.flush()
########################################### Redo GSM Model Abs Calibration on Point Source Calibrated Data ###########################################
# Absolute_Calibration_dred_mfreq_pscal = True

if Absolute_Calibration_dred_mfreq_pscal:
	vis_data_dred_mfreq_pscal_abscal, autocorr_data_dred_mfreq_pscal_abscal, vis_data_dred_pscal_abscal, autocorr_data_dred_pscal_abscal, _, _ = \
		Model_Calibration_mfreq(Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, Absolute_Calibration_dred=Absolute_Calibration_dred, Fake_wgts_dred_mfreq=Fake_wgts_dred_mfreq, Bandpass_Constrain=Bandpass_Constrain, re_cal_times=re_cal_times,
		                        antpos=antpos, Mocal_time_bin_temp=Mocal_time_bin_temp, nt_used=nt_used, lsts=lsts, Mocal_freq_bin_temp=Mocal_freq_bin_temp, flist=flist,
		                        fullsim_vis_mfreq=fullsim_vis_mfreq, vis_data_dred_mfreq=vis_data_dred_mfreq_pscal, dflags_dred_mfreq=dflags_dred_mfreq, add_Autobsl=False, autocorr_vis_mfreq=autocorr_vis_mfreq, autocorr_data_mfreq=autocorr_data_dred_mfreq_pscal, bl_dred_mfreq_select=8)

sys.stdout.flush()

####################################### Synthesize_Multifreq Data and Simulation #############################################
# fullsim_vis_mfreq     # (uBL, Pol, Times, Freqs)
# vis_data_dred_mfreq     # [pol][freq,time,ubl_index]
if Synthesize_MultiFreq:
	vis_data_dred_abscal = {}
	vis_data_dred_pscal = {}
	vis_data_dred_mfreq_pscal_abscal = {}
	
	ubl_index_sinfreq = {}
	used_redundancy_sinfreq = {}
	
	try:
		fullsim_vis = np.concatenate((fullsim_vis_mfreq[:, 0:1, :, Flist_select_index[0]][:, :, tmask, :], fullsim_vis_mfreq[:, 1:2, :, Flist_select_index[1]][:, :, tmask, :]), axis=1).transpose(3, 0, 1, 2).reshape(len(Flist_select_index[0]) * fullsim_vis_mfreq.shape[0], fullsim_vis_mfreq.shape[1], np.sum(tmask))
	except:
		print('No fullsim_vis Synthesize_Multifreq.')
	for i in range(2):
		p = ['x', 'y'][i]
		try:
			vis_data_dred[i] = vis_data_dred_mfreq[i][Flist_select_index[i], :, :].transpose(0, 2, 1).reshape(len(Flist_select_index[i]) * vis_data_dred_mfreq[i].shape[2], vis_data_dred_mfreq[i].shape[1]).transpose()
		except:
			print('No vis_data_dred[%s] Synthesize_Multifreq.' % i)
		try:
			# if i == 0:
			# 	vis_data_dred_abscal = {}
			vis_data_dred_abscal[i] = vis_data_dred_mfreq_abscal[i][Flist_select_index[i], :, :].transpose(0, 2, 1).reshape(len(Flist_select_index[i]) * vis_data_dred_mfreq_abscal[i].shape[2], vis_data_dred_mfreq_abscal[i].shape[1]).transpose()
		except:
			print('No vis_data_dred_abscal[%s] Synthesize_Multifreq.' % i)
		try:
			vis_data_dred_pscal[i] = vis_data_dred_mfreq_pscal[i][Flist_select_index[i], :, :].transpose(0, 2, 1).reshape(len(Flist_select_index[i]) * vis_data_dred_mfreq_pscal[i].shape[2], vis_data_dred_mfreq_pscal[i].shape[1]).transpose()
		except:
			print('No vis_data_dred_pscal[%s] Synthesize_Multifreq.' % i)
		try:
			vis_data_dred_mfreq_pscal_abscal[i] = vis_data_dred_mfreq_pscal_abscal[i][Flist_select_index[i], :, :].transpose(0, 2, 1).reshape(len(Flist_select_index[i]) * vis_data_dred_mfreq_pscal_abscal[i].shape[2], vis_data_dred_mfreq_pscal_abscal[i].shape[1]).transpose()
		except:
			print('No vis_data_dred_pscal_abscal[%s] Synthesize_Multifreq.' % i)
		try:
			used_redundancy_sinfreq[i] = used_redundancy[i]
		except:
			print('No used_redundancy_sinfreq[%s] Synthesize_Multifreq.' % i)
		try:
			used_redundancy[i] = np.squeeze([used_redundancy[i]] * Synthesize_MultiFreq_Nfreq).flatten()
		except:
			print('No used_redundancy[%s] Synthesize_Multifreq.' % i)
		try:
			ubl_index_sinfreq[p] = ubl_index[p]
		except:
			print('No ubl_index_sinfreq[%s] Synthesize_Multifreq.' % p)
		try:
			# ubl_index[p] = np.squeeze([ubl_index[p]] * Synthesize_MultiFreq_Nfreq).flatten()
			ubl_index[p] = np.array([(ubl_index[p] + len(ubl_index_sinfreq[p]) * id_f) for id_f in range(Synthesize_MultiFreq_Nfreq)]).flatten()
		except:
			print('No ubl_index[%s] Synthesize_Multifreq.' % p)
	
	try:
		used_common_ubls_sinfreq = used_common_ubls
	except:
		print('No used_common_ubls_sinfreq Synthesize_Multifreq.')
	try:
		used_common_ubls = np.squeeze([used_common_ubls] * Synthesize_MultiFreq_Nfreq).reshape(len(used_common_ubls_sinfreq) * Synthesize_MultiFreq_Nfreq, 3)
	except:
		print('No used_common_ubls Synthesize_Multifreq.')
	try:
		nUBL_used_sinfreq = len(used_common_ubls_sinfreq)
	except:
		nUBL_used_sinfreq = nUBL_used
	try:
		nUBL_sinfreq = len(bsl_coord_dred[0])
	except:
		nUBL_sinfreq = nUBL
	nUBL_used = len(Flist_select_index[0]) * nUBL_used_sinfreq
	nUBL = len(Flist_select_index[0]) * nUBL_sinfreq

else:
	try:
		used_redundancy_sinfreq = used_redundancy
		bl_index_sinfreq = ubl_index
		nUBL_used_sinfreq = nUBL_used
		nUBL_sinfreq = nUBL
		used_common_ubls_sinfreq = used_common_ubls
	except:
		pass

#############################################################################################################################
################################ Noise and Vis Data Loading after Synthesize_Multifreq #####################################
###########################################################################################################################
if Synthesize_MultiFreq:
	Recal_IntegrationTime = False
	Recal_FrequencyBin = False
	if len(tlist) >= 2 and 'hera47' in INSTRUMENT and Recal_IntegrationTime:
		Time_seperation_real = np.array([3600. * np.abs(tlist[i + 1] - tlist[i]) for i in range(len(tlist) - 1)])  # in second
	elif 'hera47' in INSTRUMENT:
		Time_seperation_real = 11  # second
	elif 'miteor' in INSTRUMENT:
		Time_seperation_real = 2.7  # second
	
	if len(flist) >= 2 and 'hera47' in INSTRUMENT and Recal_FrequencyBin:
		Frequency_gap_real = np.array([1.e6 * np.abs(flist[0][i + 1] - flist[0][i]) for i in range(len(flist[0]) - 1)])  # Hz
	elif 'hera47' in INSTRUMENT:
		Frequency_gap_real = 101562.5  # Hz
	elif 'miteor' in INSTRUMENT:
		Frequency_gap_real = 0.5 * 1.e6  # Hz
	
	# Integration_Time = np.mean(Time_seperation_real) / ((Time_Average_preload if Select_time else 1) * (Time_Average_afterload if use_select_time else 1))
	# Frequency_Bin = np.mean(Frequency_gap_real) / ((Frequency_Average_preload if Select_freq else 1) * (Frequency_Average_afterload if use_select_freq else 1))
	try:
		if len(Time_seperation_real) > 1:
			Integration_Time = Counter(Time_seperation_real).most_common(1)[0][0] / ((Time_Average_preload if Select_time else 1) * (Time_Average_afterload if use_select_time else 1))
	except:
		Integration_Time = Time_seperation_real * ((Time_Average_preload if not Select_time else 1) * (Time_Average_afterload if not use_select_time else 1))
	
	try:
		if len(Frequency_gap_real) > 1:
			Frequency_Bin = Counter(Frequency_gap_real).most_common(1)[0][0] / ((Frequency_Average_preload if Select_freq else 1) * (Frequency_Average_afterload if use_select_freq else 1))
	except:
		Frequency_Bin = Frequency_gap_real * ((Frequency_Average_preload if not Select_freq else 1) * (Frequency_Average_afterload if not use_select_freq else 1))
	
	try:
		print('>>>>>>>>>>>>>>>Integration_Time: %s\n>>>>>>>>>>>>>>>Frequency_Bin: %s' % (Integration_Time, Frequency_Bin))
	except:
		print('>>>>>>>>>>>>>>>Integration_Time and Frequency_Bin are not printed.')
	
	Calculate_SimulationData_Noise = True
	Calculate_Data_Noise = True
	
	# scale_noise = True
	# Use_AbsCal = False
	
	noise = {}
	noise_data = {}
	if Calculate_SimulationData_Noise:
		if Keep_Red:
			noise['x'] = np.array([np.random.normal(0, autocorr_vis_red[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
			noise['y'] = np.array([np.random.normal(0, autocorr_vis_red[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
		elif not scale_noise:
			noise['x'] = np.array([(np.random.normal(0, autocorr_vis[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
			noise['y'] = np.array([(np.random.normal(0, autocorr_vis[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
		else:
			noise['x'] = np.array([(np.random.normal(0, autocorr_vis[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_vis[0]))], dtype='float64').flatten()
			noise['y'] = np.array([(np.random.normal(0, autocorr_vis[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_vis[1]))], dtype='float64').flatten()
		
		#	N_acu = {}
		#	N_acu['x'] = np.outer(noise['x'], noise['x'].T)
		#	N_acu['y'] = np.outer(noise['y'], noise['y'].T)
		N = {}
		N['x'] = noise['x'] * noise['x']
		N['y'] = noise['y'] * noise['y']
		
		sim_var_xx_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_sim_xx.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
		sim_var_yy_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_sim_yy.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
		if not os.path.isfile(sim_var_xx_filename):
			N['x'].astype('float64').tofile(sim_var_xx_filename)
		if not os.path.isfile(sim_var_yy_filename):
			N['y'].astype('float64').tofile(sim_var_yy_filename)
		
		Del = True
		if Del:
			del (noise)
			del (N)
	
	if Calculate_Data_Noise:
		if INSTRUMENT == 'miteor':
			noise_data['x'] = (var_data[0].flatten()) ** 0.5
			noise_data['y'] = (var_data[1].flatten()) ** 0.5
		
		elif 'hera47' in INSTRUMENT:
			if Keep_Red:
				noise_data['x'] = np.array([np.random.normal(0, autocorr_data[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Not Absolute Calibrated
				noise_data['y'] = np.array([np.random.normal(0, autocorr_data[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
			else:
				if not scale_noise:
					if Use_AbsCal:
						noise_data['x'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[0]))], dtype='float64').flatten()  # Absolute Calibrated
						noise_data['y'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[1]))], dtype='float64').flatten()
					else:
						noise_data['x'] = np.array([(np.random.normal(0, autocorr_data[0][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Absolute Calibrated
						noise_data['y'] = np.array([(np.random.normal(0, autocorr_data[1][t_index] / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[1]) ** 0.5) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
				else:
					if Use_AbsCal:
						noise_data['x'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[0]))], dtype='float64').flatten()  # Absolute Calibrated
						noise_data['y'] = np.array([(np.random.normal(0, autocorr_data_dred_abscal[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data_dred_abscal[1]))], dtype='float64').flatten()
					else:
						noise_data['x'] = np.array([(np.random.normal(0, autocorr_data[0][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[0]))], dtype='float64').flatten()  # Absolute Calibrated
						noise_data['y'] = np.array([(np.random.normal(0, autocorr_data[1][t_index] * scale_noise_ratio / (Integration_Time * Frequency_Bin) ** 0.5, nUBL_used) / np.array(used_redundancy[0]) ** 0.5) for t_index in range(len(autocorr_data[1]))], dtype='float64').flatten()
		
		#	N_data_acu = {}
		#	N_data_acu['x'] = np.outer(noise_data['x'], noise_data['x'].T)
		#	N_data_acu['y'] = np.outer(noise_data['y'], noise_data['y'].T)
		N_data = {}
		N_data['x'] = noise_data['x'] * noise_data['x']
		N_data['y'] = noise_data['y'] * noise_data['y']
		
		Store_Data_Noise = True
		Re_Save = False
		
		if Store_Data_Noise:
			data_var_xx_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_xx.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
			data_var_yy_filename = script_dir + '/../Output/%s_%s_p2_scale%s_u%i_t%i_nside%i_bnside%i_var_data_yy.simvis' % (INSTRUMENT, freq, scale_noise_ratio if scale_noise else '-none', nUBL_used + 1, nt_used, nside_standard, bnside)
			if not os.path.isfile(data_var_xx_filename):
				N_data['x'].astype('float64').tofile(data_var_xx_filename)
				print('N_data[x] saved to disc.')
			elif Re_Save:
				N_data['x'].astype('float64').tofile(data_var_xx_filename)
				print('N_data[x] saved to disc again.')
			else:
				print('N_data[x] not saved to disc again.')
			
			if not os.path.isfile(data_var_yy_filename):
				N_data['y'].astype('float64').tofile(data_var_yy_filename)
				print('N_data[y] saved to disc.')
			elif Re_Save:
				N_data['y'].astype('float64').tofile(data_var_yy_filename)
				print('N_data[y] saved to disc again.')
			else:
				print('N_data[x] not saved to disc again.')
		
		Del = True
		if Del:
			del (noise_data)
	
	sys.stdout.flush()

##################################################################################################################################################################################
########################################################### Store Visibility Data and prepare to Delete Variable ################################################################
try:
	data_vis_xx_filename = script_dir + '/../Output/%s_%sMHz_p2_u%i_t%i_nside%i_bnside%i_vis_data_xx.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, nside_standard, bnside)
	data_vis_yy_filename = script_dir + '/../Output/%s_%sMHz_p2_u%i_t%i_nside%i_bnside%i_vis_data_yy.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, nside_standard, bnside)
	vis_data[0].astype('complex128').tofile(data_vis_xx_filename)
	vis_data[1].astype('complex128').tofile(data_vis_yy_filename)
except:
	pass

try:
	data_vis_dred_xx_filename = script_dir + '/../Output/%s_%sMHz_p2_u%i_t%i_nside%i_bnside%i_vis_data_dred_xx.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, nside_standard, bnside)
	data_vis_dred_yy_filename = script_dir + '/../Output/%s_%sMHz_p2_u%i_t%i_nside%i_bnside%i_vis_data_dred_yy.simvis' % (INSTRUMENT, freq, nUBL_used + 1, nt_used, nside_standard, bnside)
	vis_data_dred[0].astype('complex128').tofile(data_vis_dred_xx_filename)
	vis_data_dred[1].astype('complex128').tofile(data_vis_dred_yy_filename)
except:
	pass

try:
	data_vis_dred_mfreq_xx_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_xx_mfreq%s-%s-%sMHz_dred_mfreq.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, nside_standard, bnside, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	data_vis_dred_mfreq_yy_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_yy_mfreq%s-%s-%sMHz_dred_mfreq.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, nside_standard, bnside, np.min(flist[1]), np.max(flist[1]), len(flist[1]))
	vis_data_dred_mfreq[0].astype('complex128').tofile(data_vis_dred_mfreq_xx_filename)
	vis_data_dred_mfreq[1].astype('complex128').tofile(data_vis_dred_mfreq_yy_filename)
except:
	pass

try:
	abscal_data_vis_dred_mfreq_xx_filename = script_dir + '/../Output/%s_p2_u%i_t%i-mtbin%s-mfbin%s_nside%i_bnside%i_vis_data_xx_mfreq%s-%s-%sMHz_dred_mfreq_abscal.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', nside_standard, bnside, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	abscal_data_vis_dred_mfreq_yy_filename = script_dir + '/../Output/%s_p2_u%i_t%i-mtbin%s-mfbin%s_nside%i_bnside%i_vis_data_yy_mfreq%s-%s-%sMHz_dred_mfreq_abscal.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', nside_standard, bnside, np.min(flist[1]), np.max(flist[1]), len(flist[1]))
	vis_data_dred_mfreq_abscal[0].astype('complex128').tofile(abscal_data_vis_dred_mfreq_xx_filename)
	vis_data_dred_mfreq_abscal[1].astype('complex128').tofile(abscal_data_vis_dred_mfreq_yy_filename)
except:
	pass

try:
	abscal_data_vis_dred_xx_filename = script_dir + '/../Output/%s_p2_u%i_t%i-mtbin%s-mfbin%s_nside%i_bnside%i_vis_data_xx_%sMHz_dred_abscal.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', nside_standard, bnside, freq)
	abscal_data_vis_dred_yy_filename = script_dir + '/../Output/%s_p2_u%i_t%i-mtbin%s-mfbin%s_nside%i_bnside%i_vis_data_yy_%sMhz_dred_abscal.simvis' % (INSTRUMENT, nUBL_used + 1, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', nside_standard, bnside, freq)
	vis_data_dred_abscal[0].astype('complex128').tofile(abscal_data_vis_dred_xx_filename)
	vis_data_dred_abscal[1].astype('complex128').tofile(abscal_data_vis_dred_yy_filename)
except:
	pass

try:
	pscal_data_vis_dred_mfreq_xx_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_xx_mfreq%s-%s-%sMHz_dred_mfreq_pscal.simvis' % (INSTRUMENT, nUBL, nt, nside_standard, bnside, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	pscal_data_vis_dred_mfreq_yy_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_yy_mfreq%s-%s-%sMHz_dred_mfreq_pscal.simvis' % (INSTRUMENT, nUBL, nt, nside_standard, bnside, np.min(flist[0]), np.max(flist[0]), len(flist[0]))
	vis_data_dred_mfreq_pscal[0].astype('complex128').tofile(pscal_data_vis_dred_mfreq_xx_filename)
	vis_data_dred_mfreq_pscal[1].astype('complex128').tofile(pscal_data_vis_dred_mfreq_yy_filename)
	
	pscal_data_vis_dred_xx_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_xx_%sMHz_pscal.simvis' % (INSTRUMENT, nUBL, nt, nside_standard, bnside, freq)
	pscal_data_vis_dred_yy_filename = script_dir + '/../Output/%s_p2_u%i_t%i_nside%i_bnside%i_vis_data_yy_%sMHz_pscal.simvis' % (INSTRUMENT, nUBL, nt, nside_standard, bnside, freq)
	vis_data_dred_pscal[0].astype('complex128').tofile(pscal_data_vis_dred_xx_filename)
	vis_data_dred_pscal[1].astype('complex128').tofile(pscal_data_vis_dred_yy_filename)
except:
	pass

sys.stdout.flush()

###################################################################################################################################################################################
############################################################################# read data and N ####################################################################################
##################################################################################################################################################################################
################
####read data and N
################
data = {}
Ni = {}
data_shape = {}
ubl_sort = {}
data_filename = full_sim_filename

# Use_Simulation_noise = True
# From_File_Data = True

for p in ['x', 'y']:
	pol = p + p
	print "%i UBLs to include, longest baseline is %i wavelengths" % (
		nUBL_used, np.max(np.linalg.norm(used_common_ubls, axis=1)) / (C / freq))
	if p == 'x':
		pol_index = 0
		sim_var_filename = sim_var_xx_filename
		sim_vis_filename = sim_vis_xx_filename
		try:
			data_var_filename = data_var_xx_filename
			data_var_filename_pscal = data_var_xx_filename_pscal
		except:
			pass
	elif p == 'y':
		pol_index = 1
		sim_var_filename = sim_var_yy_filename
		sim_vis_filename = sim_vis_yy_filename
		try:
			data_var_filename = data_var_yy_filename
			data_var_filename_pscal = data_var_yy_filename_pscal
		except:
			pass
	
	if Use_SimulatedData == 1:
		#		Ni[pol] = 1. / (np.fromfile(data_var_filename, dtype='float64').reshape((nt_used, nUBL_used))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten() * jansky2kelvin ** 2)
		Ni[pol] = 1. / (np.fromfile(sim_var_filename, dtype='float64').reshape((nt_used, nUBL_used)).transpose().flatten())
		data[pol] = (np.fromfile(sim_vis_filename, dtype='complex128').reshape((nUBL_used, nt_used)))  # .conjugate()
	else:
		if INSTRUMENT == 'miteor':
			if Use_Simulation_noise:
				Ni[pol] = 1. / (np.fromfile(data_var_filename, dtype='float64').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten())
			else:
				Ni[pol] = 1. / var_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1].flatten()
			data[pol] = vis_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1]  # = (time_vis_data[:,1:,1::3] + time_vis_data[:,1:,2::3] * 1j).astype('complex64')
			data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
			#			data[pol] = (data[pol].flatten() * jansky2kelvin).conjugate()  # there's a conjugate convention difference
			data[pol] = (data[pol].flatten()).conjugate()  # there's a conjugate convention difference
		elif 'hera47' in INSTRUMENT:
			if From_File_Data:
				if Use_PsAbsCal:
					Ni[pol] = 1. / (np.fromfile(data_var_filename_pscal, dtype='float64').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten())  # var_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1].flatten()
				elif Use_Fullsim_Noise:
					Ni[pol] = 1. / (np.fromfile(sim_var_filename, dtype='float64').reshape((nt_used, nUBL_used)).transpose().flatten())
				else:
					Ni[pol] = 1. / (np.fromfile(data_var_filename, dtype='float64').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten())  # var_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1].flatten()
			else:
				if Use_PsAbsCal:
					Ni[pol] = 1. / N_data_pscal[p].reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten()  # var_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1].flatten()
				elif Use_Fullsim_Noise:
					Ni[pol] = 1. / (np.fromfile(sim_var_filename, dtype='float64').reshape((nt_used, nUBL_used)).transpose().flatten())
				else:
					Ni[pol] = 1. / N_data[p].reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1].flatten()  # var_data[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1].flatten()
			if From_File_Data:
				if Use_PsAbsCal:
					data[pol] = np.fromfile(globals()['pscal_data_vis_dred_' + pol + '_filename'], dtype='complex128').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1]
				elif Use_AbsCal:
					data[pol] = np.fromfile(globals()['abscal_data_vis_dred_' + pol + '_filename'], dtype='complex128').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1]
				else:
					data[pol] = jansky2kelvin * np.fromfile(globals()['data_vis_dred_' + pol + '_filename'], dtype='complex128').reshape((nt, nUBL))[tmask].transpose()[abs(ubl_index[p]) - 1]  # .conjugate()
			else:
				if Use_PsAbsCal:
					data[pol] = vis_data_dred_pscal[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1]  # = (time_vis_data[:,1:,1::3] + time_vis_data[:,1:,2::3] * 1j).astype('complex64')
					# data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0]#.conjugate()
					data[pol] = data[pol].flatten()  # .conjugate()  # there's a conjugate convention difference
				elif Use_AbsCal:
					data[pol] = vis_data_dred_abscal[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1]  # = (time_vis_data[:,1:,1::3] + time_vis_data[:,1:,2::3] * 1j).astype('complex64')
					# data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0]#.conjugate()
					data[pol] = data[pol].flatten()  # .conjugate()  # there's a conjugate convention difference
				else:
					data[pol] = jansky2kelvin * vis_data_dred[pol_index][tmask].transpose()[abs(ubl_index[p]) - 1]  # .conjugate()  # = (time_vis_data[:,1:,1::3] + time_vis_data[:,1:,2::3] * 1j).astype('complex64')
	# data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0].conjugate()
	# data[pol] = (data[pol].flatten() * jansky2kelvin).conjugate()  # there's a conjugate convention difference
	# data[pol][ubl_index[p] < 0] = data[pol][ubl_index[p] < 0]#.conjugate()
	# data[pol] = (data[pol].flatten() * jansky2kelvin)#.conjugate()  # there's a conjugate convention difference
	data_shape[pol] = (nUBL_used, nt_used)
	# ubl_sort[p] = np.argsort(la.norm(used_common_ubls, axis=1))
	ubl_sort[p] = np.arange(len(used_common_ubls))
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sys.stdout.flush()

# Only_AbsData = False

# Merge data
data = np.array([data['xx'], data['yy']]).reshape([2] + list(data_shape['xx'])).transpose(
	(1, 0, 2)).flatten()
Ni = np.concatenate((Ni['xx'], Ni['yy'])).reshape([2] + list(data_shape['xx'])).transpose(
	(1, 0, 2)).flatten()
if Only_AbsData:
	# data = np.abs(data)
	Ni = Ni + Ni * 1.j
else:
	data = np.concatenate((np.real(data), np.imag(data))).astype('float64')
	Ni = np.concatenate((Ni * 2, Ni * 2))

sys.stdout.flush()

##################### Delete or Erase Data #######################
# Erase = False
if Erase:
	
	try:
		del (dflags_sf)
	except:
		pass
	
	try:
		del (dflags_dred)
	except:
		pass
	
	try:
		del (dflags)
	except:
		pass
	
	try:
		del (dflags_dred_mfreq)
	except:
		pass
	
	try:
		del (vis_data)
		del (vis_data_dred)
	except:
		pass
	try:
		del (vis_data_mfreq)
		del (vis_data_dred_mfreq)
	except:
		pass
	
	try:
		del (N_data)
	except:
		pass
	try:
		del (noise_data)
	except:
		pass
	try:
		del (var_data)
	except:
		pass
	try:
		del (vis_data_dred_mfreq_pscal)
		del (vis_data_dred_pscal)
	except:
		pass
	try:
		del (vis_data_dred_mfreq_abscal)
		del (vis_data_dred_abscal)
	except:
		pass
	try:
		del (vis_data_dred_mfreq_pscal_abscal)
		del (vis_data_dred_pscal_abscal)
	except:
		pass
	try:
		del (autocorr_data_dred_mfreq_abscal)
		del (autocorr_data_dred_abscal)
	except:
		pass
	try:
		del (autocorr_data_dred_mfreq_pscal)
		del (autocorr_data_mfreq)
	except:
		pass
	try:
		del (autocorr_data_dred_mfreq_pscal_abscal)
		del (autocorr_data_dred_pscal_abscal)
	except:
		pass
	try:
		del (autocorr_vis_mfreq_sf_normalized)
		del (autocorr_vis_mfreq_sf)
	except:
		pass
	try:
		del (fullsim_vis_mfreq_sf)
	except:
		pass
	try:
		del (fullsim_vis_red)
	except:
		pass
	try:
		del (A)
	except:
		print('No A to be deleted.')

sys.stdout.flush()

################################
################################
####pre_calibrate################
################################
################################
try:
	if pre_calibrate:
		if Absolute_Calibration_dred_mfreq and PointSource_AbsCal:
			data, Ni, additive_A, additive_term, additive_sol, precal_time_bin = Pre_Calibration(pre_calibrate=pre_calibrate, pre_ampcal=pre_ampcal, pre_phscal=pre_phscal, pre_addcal=pre_addcal, comply_ps2mod_autocorr=comply_ps2mod_autocorr, Use_PsAbsCal=Use_PsAbsCal, Use_AbsCal=Use_AbsCal, Use_Fullsim_Noise=Use_Fullsim_Noise, Only_AbsData=Only_AbsData,
			                                                                                     Precal_time_bin_temp=Precal_time_bin_temp, nt_used=nt_used, nUBL_used=nUBL_used,
			                                                                                     data_shape=data_shape, cal_times=1, niter_max=50, antpairs=None, ubl_index=ubl_index, autocorr_vis_normalized=autocorr_vis_normalized, fullsim_vis=fullsim_vis, data=data, Ni=Ni, pt_vis=pt_vis, pt_sources=pt_sources, used_common_ubls=used_common_ubls, freq=freq, lsts=lsts,
			                                                                                     lst_offset=lst_offset, INSTRUMENT=INSTRUMENT, Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, mocal_time_bin=mocal_time_bin, mocal_freq_bin=mocal_freq_bin, bnside=bnside, nside_standard=nside_standard, ubl_sort=ubl_sort)
		elif Absolute_Calibration_dred_mfreq:
			data, Ni, additive_A, additive_term, additive_sol, precal_time_bin = Pre_Calibration(pre_calibrate=pre_calibrate, pre_ampcal=pre_ampcal, pre_phscal=pre_phscal, pre_addcal=pre_addcal, comply_ps2mod_autocorr=comply_ps2mod_autocorr, Use_PsAbsCal=Use_PsAbsCal, Use_AbsCal=Use_AbsCal, Use_Fullsim_Noise=Use_Fullsim_Noise, Only_AbsData=Only_AbsData,
			                                                                                     Precal_time_bin_temp=Precal_time_bin_temp, nt_used=nt_used, nUBL_used=nUBL_used,
			                                                                                     data_shape=data_shape, cal_times=1, niter_max=50, antpairs=None, ubl_index=ubl_index, autocorr_vis_normalized=autocorr_vis_normalized, fullsim_vis=fullsim_vis, data=data, Ni=Ni, pt_vis=[], pt_sources=[], used_common_ubls=used_common_ubls, freq=freq, lsts=lsts,
			                                                                                     lst_offset=lst_offset, INSTRUMENT=INSTRUMENT, Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, mocal_time_bin=mocal_time_bin, mocal_freq_bin=mocal_freq_bin, bnside=bnside, nside_standard=nside_standard, ubl_sort=ubl_sort)
		elif PointSource_AbsCal:
			data, Ni, additive_A, additive_term, additive_sol, precal_time_bin = Pre_Calibration(pre_calibrate=pre_calibrate, pre_ampcal=pre_ampcal, pre_phscal=pre_phscal, pre_addcal=pre_addcal, comply_ps2mod_autocorr=comply_ps2mod_autocorr, Use_PsAbsCal=Use_PsAbsCal, Use_AbsCal=Use_AbsCal, Use_Fullsim_Noise=Use_Fullsim_Noise, Only_AbsData=Only_AbsData,
			                                                                                     Precal_time_bin_temp=Precal_time_bin_temp, nt_used=nt_used, nUBL_used=nUBL_used,
			                                                                                     data_shape=data_shape, cal_times=1, niter_max=50, antpairs=None, ubl_index=ubl_index, autocorr_vis_normalized=autocorr_vis_normalized, fullsim_vis=fullsim_vis, data=data, Ni=Ni, pt_vis=pt_vis, pt_sources=pt_sources, used_common_ubls=used_common_ubls, freq=freq, lsts=lsts,
			                                                                                     lst_offset=lst_offset, INSTRUMENT=INSTRUMENT, Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, mocal_time_bin=None, mocal_freq_bin=None, bnside=bnside, nside_standard=nside_standard, ubl_sort=ubl_sort)
		else:
			data, Ni, additive_A, additive_term, additive_sol, precal_time_bin = Pre_Calibration(pre_calibrate=pre_calibrate, pre_ampcal=pre_ampcal, pre_phscal=pre_phscal, pre_addcal=pre_addcal, comply_ps2mod_autocorr=comply_ps2mod_autocorr, Use_PsAbsCal=Use_PsAbsCal, Use_AbsCal=Use_AbsCal, Use_Fullsim_Noise=Use_Fullsim_Noise, Only_AbsData=Only_AbsData,
			                                                                                     Precal_time_bin_temp=Precal_time_bin_temp, nt_used=nt_used, nUBL_used=nUBL_used,
			                                                                                     data_shape=data_shape, cal_times=1, niter_max=50, antpairs=None, ubl_index=ubl_index, autocorr_vis_normalized=autocorr_vis_normalized, fullsim_vis=fullsim_vis, data=data, Ni=Ni, pt_vis=[], pt_sources=[], used_common_ubls=used_common_ubls, freq=freq, lsts=lsts,
			                                                                                     lst_offset=lst_offset, INSTRUMENT=INSTRUMENT, Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, mocal_time_bin=None, mocal_freq_bin=None, bnside=bnside, nside_standard=nside_standard, ubl_sort=ubl_sort)
		
	else:
		additive_A, additive_term, precal_time_bin = Pre_Calibration(pre_calibrate=pre_calibrate, pre_ampcal=pre_ampcal, pre_phscal=pre_phscal, pre_addcal=pre_addcal, comply_ps2mod_autocorr=comply_ps2mod_autocorr, Use_PsAbsCal=Use_PsAbsCal, Use_AbsCal=Use_AbsCal, Use_Fullsim_Noise=Use_Fullsim_Noise, Only_AbsData=Only_AbsData,
		                                                             Precal_time_bin_temp=Precal_time_bin_temp, nt_used=nt_used, nUBL_used=nUBL_used,
		                                                             data_shape=data_shape, cal_times=1, niter_max=50, antpairs=None, ubl_index=ubl_index, autocorr_vis_normalized=autocorr_vis_normalized, fullsim_vis=fullsim_vis, data=data, Ni=Ni, pt_vis=[], pt_sources=[], used_common_ubls=used_common_ubls, freq=freq, lsts=lsts,
		                                                             lst_offset=lst_offset, INSTRUMENT=INSTRUMENT, Absolute_Calibration_dred_mfreq=Absolute_Calibration_dred_mfreq, mocal_time_bin=mocal_time_bin, mocal_freq_bin=mocal_freq_bin, bnside=bnside, nside_standard=nside_standard, ubl_sort=ubl_sort)
	
		if not fit_for_additive:
			additive_A = np.ones(0)
			additive_term = np.ones(0)
			print('additive_A and additive_term have been deleted.')
			
except:
	additive_A = np.ones(0)
	additive_term = np.ones(0)
	fit_for_additive = False
	print('No pre_calibration plotted or calculated and not fit_for_additive.')

Del = True
if Del:
	try:
		# if not fit_for_additive:
		# 	additive_A = None
		# 	additive_term = None
		del (crdata)
		del (cNi)
		del (cdata)
		del (cadd)
	except:
		pass

sys.stdout.flush()

################
####Use N and the par file generated by pixel_parameter_search to determine dynamic pixel parameters
################
if seek_optimal_threshs:
	par_result_filename = full_sim_filename.replace('.simvis', '_par_search.npz')
	par_file = np.load(par_result_filename)
	qualified_par_mask = (par_file['err_norm'] / np.sum(1. / Ni) ** .5) < dynamic_precision
	index_min_pix_in_mask = np.argmin(par_file['n_pix'][qualified_par_mask])
	thresh, valid_pix_thresh = par_file['parameters'][qualified_par_mask][index_min_pix_in_mask]
print "<<<<<<<<<<<<picked std thresh %.3f, pix thresh %.1e" % (thresh, valid_pix_thresh)

sys.stdout.flush()

##########################################################################################################################################
######################################### Dynamica Pixelization and A Matrix Calculation ################################################
#########################################################################################################################################

beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution = \
	get_A_multifreq(fit_for_additive=False, additive_A=None, force_recompute=False, Compute_A=False, A_path='', A_got=None, A_version=1.0, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=Synthesize_MultiFreq, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight, Only_AbsData=Only_AbsData,
	                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=beam_weight,
	                used_common_ubls=used_common_ubls, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)

try:
	if plot_pixelization:
		##################################################################
		####################################sanity check########################
		###############################################################
		plotcoord = 'C'
		stds = np.std((equatorial_GSM_standard * beam_weight).reshape(12 * nside_standard ** 2 / 4, 4), axis=1)
		
		##################################################################
		####################################plotting########################
		###############################################################
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=RuntimeWarning)
			# plt.clf()
			plt.figure(50)
			hpv.mollview(beam_weight, min=0, max=4, coord=plotcoord, title='%s-dipole-beam_weight-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s' % (tag, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'), nest=True)
			plt.savefig(script_dir + '/../Output/%s-%sMHz-dipole-beam_weight-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.pdf' % (tag, freq, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
			hpv.mollview(np.log10(equatorial_GSM_standard), min=0, max=4, coord=plotcoord, title='GSM', nest=True)
			plt.savefig(script_dir + '/../Output/GSM-3C-for-%s-%sMHz-bnside-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.pdf' % (tag, freq, bnside, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
			hpv.mollview(np.log10(sol2map(fake_solution, valid_npix=valid_npix, npix=npix, valid_pix_mask=valid_pix_mask, final_index=final_index)[:len(equatorial_GSM_standard)]), min=0, max=4, coord=plotcoord,
			             title='GSM gridded', nest=True)
			plt.savefig(script_dir + '/../Output/maskedfsol_GSM-3C-%s-%sMHz-dipole-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.pdf' % (tag, freq, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
			hpv.mollview(np.log2(nside_distribution), min=np.log2(nside_start), max=np.log2(nside_standard),
			             coord=plotcoord,
			             title='nside_distribution(count %i %.3f)' % (len(thetas), float(len(thetas)) / (12 * nside_standard ** 2)), nest=True)
			plt.savefig(script_dir + '/../Output/nside_distribution-%s-%sMHz-dipole-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.pdf' % (tag, freq, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
			hpv.mollview(np.log10(stds / abs_thresh), min=np.log10(thresh) - 3, max=3, coord=plotcoord, title='std-%s-%s-dipole-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s' % (tag, freq, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'),
			             nest=True)
			plt.savefig(script_dir + '/../Output/stds-beam_weight_GSM-%s-%s-dipole-bnside-%s-nside_standard-%s-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.pdf' % (tag, freq, bnside, nside_standard, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
			plt.show(block=False)
#           plt.gcf().clear()
except:
	print('Error when Plotting GSM Maps.')
sys.stdout.flush()

############################################################ Compute A Matrix ##########################################################
A_version = 1.0
A_tag = 'A_dI'
if not Synthesize_MultiFreq:
	A_filename = A_tag + '_freq%sM_u%i_t%i_mtb%s-mfb%s-tb%s_p%i_n%i_%i_b%i_%.3f_v%.1f' % (freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_N', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_N', precal_time_bin if pre_calibrate else '_N', valid_npix, nside_start, nside_standard, bnside, thresh, A_version)
else:
	A_filename = A_tag + '_freq%sM-multifreq%s-%s-%s_u%i_t%i_mtb%s-mfb%s-tb%s_p%i_n%i_%i_b%i_%.3f_v%.1f' % (freq, Synthesize_MultiFreq_Nfreq, Flist_select[0][0], Flist_select[0][-1], nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_N', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_N', precal_time_bin if pre_calibrate else '_N', valid_npix, nside_start, nside_standard, bnside, thresh, A_version)
A_path = datadir + tag + A_filename
AtNiA_tag = 'AtNiA_N%s' % vartag
if not fit_for_additive:
	AtNiA_tag += "_noadd"
elif crosstalk_type == 'autocorr':
	AtNiA_tag += "_autocorr"
if pre_ampcal:
	AtNiA_tag += "_ampcal"
AtNiA_filename = AtNiA_tag + A_filename
AtNiA_path = datadir + tag + AtNiA_filename
if os.path.isfile(AtNiA_path) and AtNiA_only and not force_recompute:
	sys.exit(0)

try:
	del (A)
except:
	print('A has already been successfully deleted.')
try:
	A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution = \
		get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=False, Compute_A=True, A_path=A_path, A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=Synthesize_MultiFreq, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight, Only_AbsData=Only_AbsData, Del_A=Del_A,
		                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=beam_weight,
		                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts, Parallel_A=Parallel_A)
except:
	try:
		if Parallel_A:
			Parallel_A = False
			print('Use Unparallel Computing for A instead.')
			A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution = \
				get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=False, Compute_A=True, A_path=A_path, A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=Synthesize_MultiFreq, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight, Only_AbsData=Only_AbsData, Del_A=Del_A,
				                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=beam_weight,
				                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts, Parallel_A=Parallel_A)
			
	except:
		if not Del_A:
			Del_A = True
			A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution = \
				get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=False, Compute_A=True, A_path=A_path, A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=Synthesize_MultiFreq, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight, Only_AbsData=Only_AbsData, Del_A=Del_A,
				                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=beam_weight,
				                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts, Parallel_A=Parallel_A)
		else:
			pass
	
# A_test, beam_weight_test, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map_test, fake_solution_test = \
# 	get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=True, Compute_A=True, A_path=A_path, A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=False, thresh=2., valid_pix_thresh = 1.e-4,
# 	                flist=flist, Flist_select=None, Flist_select_index=None, Reference_Freq_Index=None, Reference_Freq=[148.4375, 148.4375], equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=None,
# 	                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)
#
# A_test2, beam_weight_test2, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map_test2, fake_solution_test2, fake_solution_map_mfreq_test2 = \
# 	get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=True, Compute_A=True, A_path=A_path, A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=Synthesize_MultiFreq, thresh=2., valid_pix_thresh = 1.e-4,
# 	                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[148.4375, 148.4375], equatorial_GSM_standard=None, equatorial_GSM_standard_mfreq=equatorial_GSM_standard_mfreq, beam_weight=None,
# 	                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=None, beam_heal_equ_y=None, beam_heal_equ_x_mfreq=beam_heal_equ_x_mfreq, beam_heal_equ_y_mfreq=beam_heal_equ_y_mfreq, lsts=lsts)

Ashape0, Ashape1 = A.shape

print "Memory usage: %fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

sys.stdout.flush()
##############
# simulate visibilities according to the pixelized A matrix
##############
clean_sim_data = A.dot(fake_solution.astype(A.dtype))

try:
	if plot_data_error:
		if not Only_AbsData:
			cdata = get_complex_data(data, nubl=nUBL_used, nt=nt_used)
			cdynamicmodel = get_complex_data(clean_sim_data, nubl=nUBL_used, nt=nt_used)
			cNi = get_complex_data(Ni, nubl=nUBL_used, nt=nt_used)
		else:
			cdata = data.reshape((nUBL_used, 2, nt_used))
			cdynamicmodel = clean_sim_data.reshape((nUBL_used, 2, nt_used))
			cNi = Ni.reshape((nUBL_used, 2, nt_used))
			
		if pre_calibrate:
			cadd = get_complex_data(additive_term, nubl=nUBL_used, nt=nt_used)
		
		fun = np.imag
		srt = sorted((lsts - lst_offset) % 24. + lst_offset)
		asrt = np.argsort((lsts - lst_offset) % 24. + lst_offset)
		# pncol = min(int(60. / (srt[-1] - srt[0])), 12) if nt_used > 1 else (len(ubl_sort['x']) / 2)
		# us = ubl_sort['x'][::len(ubl_sort['x']) / pncol] if len(ubl_sort['x']) / pncol >= 1 else ubl_sort['x']
		us = ubl_sort['x'][::len(ubl_sort['x']) / 24]
		
		figure_D = {}
		for p in range(2):
			for nu, u in enumerate(us):
				plt.figure(6000 + 100 * p + nu)
				plt.clf()
				# plt.subplot(2, len(us), len(us) * p + nu + 1)
				figure_D[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
				figure_D[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
				figure_D[3], = plt.plot(srt, fun(cdynamicmodel[u, p][asrt]), '+', label='dynsim_vis')
				figure_D[4], = plt.plot(srt, fun(cNi[u, p][asrt]) ** -.5, label='Ni')
				if pre_calibrate:
					figure_D[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
					data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(np.abs(fun(cdynamicmodel[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
				else:
					data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(np.abs(fun(cdynamicmodel[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
				plt.title("Dynamic-Sim %s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
				plt.yscale('symlog')
				# plt.ylim([-1.05 * data_range, 1.05 * data_range])
				if pre_calibrate:
					plt.legend(handles=[figure_D[1], figure_D[2], figure_D[3], figure_D[4], figure_D[5]], labels=['calibrated_data', 'fullsim_vis', 'dynsim_vis', 'noise', 'additive'], loc=0)
				else:
					plt.legend(handles=[figure_D[1], figure_D[2], figure_D[3], figure_D[4]], labels=['calibrated_data', 'fullsim_vis', 'dynsim_vis', 'noise'], loc=0)
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Dynamic_Vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (tag, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside,
				                                                                                                                                                                                  nside_standard))  # -nubl%s-nt%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (tag, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, precal_time_bin, bnside, nside_standard))
				plt.show(block=False)
		
		fun = np.real
		figure_DR = {}
		for p in range(2):
			for nu, u in enumerate(us):
				plt.figure(60000 + 100 * p + nu)
				plt.clf()
				# plt.subplot(2, len(us), len(us) * p + nu + 1)
				figure_DR[1], = plt.plot(srt, fun(cdata[u, p][asrt]), label='calibrated_data')
				figure_DR[2], = plt.plot(srt, fun(fullsim_vis[u, p][asrt]), label='fullsim_vis')
				figure_DR[3], = plt.plot(srt, fun(cdynamicmodel[u, p][asrt]), '+', label='dynsim_vis')
				figure_DR[4], = plt.plot(srt, fun(cNi[u, p][asrt]) ** -.5, label='Ni')
				if pre_calibrate:
					figure_DR[5], = plt.plot(srt, fun(cadd[u, p][asrt]), label='additive')
					data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(np.abs(fun(cdynamicmodel[u, p]))), np.max(fun(cadd[u, p]))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
				else:
					data_range = np.max([np.max(np.abs(fun(cdata[u, p]))), np.max(np.abs(fun(fullsim_vis[u, p]))), np.max(np.abs(fun(cdynamicmodel[u, p])))])  # 5 * np.max(np.abs(fun(cNi[u, p])))
				plt.title("Dynamic-Sim-Real %s Baseline-%.1f_%.1f results on srtime" % (['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
				plt.yscale('symlog')
				# plt.ylim([-1.05 * data_range, 1.05 * data_range])
				if pre_calibrate:
					plt.legend(handles=[figure_DR[1], figure_DR[2], figure_DR[3], figure_DR[4], figure_DR[5]], labels=['calibrated_data', 'fullsim_vis', 'dynsim_vis', 'noise', 'additive'], loc=0)
				else:
					plt.legend(handles=[figure_DR[1], figure_DR[2], figure_DR[3], figure_DR[4]], labels=['calibrated_data', 'fullsim_vis', 'dynsim_vis', 'noise'], loc=0)
				plt.savefig(script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-Dynamic_Vis_Real-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (tag, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside,
				                                                                                                                                                                                  nside_standard))  # -nubl%s-nt%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (tag, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, precal_time_bin, bnside, nside_standard))
				plt.show(block=False)
		# plt.clf()
		# plt.gcf().clear()
		print "total deviation between dynamic and full sim compared to sim: ", la.norm(fullsim_vis - cdynamicmodel) / la.norm(fullsim_vis)
		print "total deviation between dynamic and full sim compared to data noise: ", la.norm(fullsim_vis - cdynamicmodel) / np.sum(Ni ** -1) ** .5
		
		plt.figure(70)
		try:
			fullsim_vis2 = 4 * np.fromfile(datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis' % (nUBL_used + 1, nt_used, nside_standard / 2, bnside), dtype='complex128').reshape((2, nUBL_used + 1, nt_used))[:, :-1].transpose((1, 0, 2))
			plt.plot(la.norm(used_common_ubls, axis=-1) * freq / C, la.norm(fullsim_vis - fullsim_vis2, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'g+', label='nside error(%s-dipole-beam_weight-bnside-%s-nside_standard-%s)' % (tag, bnside, nside_standard))
			plt.savefig(script_dir + '/../Output/nside_error-%s-dipole-beam_weight-bnside-%s-nside_standard-%s.pdf' % (tag, bnside, nside_standard))
		except:
			try:
				fullsim_vis2 = .25 * np.fromfile(datadir + tag + '_p2_u%i_t%i_nside%i_bnside%i.simvis' % (nUBL_used + 1, nt_used, nside_standard * 2, bnside), dtype='complex128').reshape((2, nUBL_used + 1, nt_used))[:, :-1].transpose((1, 0, 2))
				plt.plot(la.norm(used_common_ubls, axis=-1) * freq / C, la.norm(fullsim_vis - fullsim_vis2, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'g+', label='nside error(%s-dipole-beam_weight-bnside-%s-nside_standard-%s)' % (tag, bnside, nside_standard))
				plt.savefig(script_dir + '/../Output/nside_error-%s-dipole-beam_weight-bnside-%s-nside_standard-%s.pdf' % (tag, bnside, nside_standard))
			except:
				pass
		# plt.clf()
		# plt.plot(la.norm(used_common_ubls, axis=-1) * freq / C, np.sum(2. / np.real(get_complex_data(Ni, nubl=nUBL_used, nt=nt_used)), axis=-1)[:, 0] ** .5 / la.norm(fullsim_vis, axis=-1)[:, 0], 'b+', label='noise error(%s-%s-dipole-beam_weight-bnside-%s-nside_standard-%s)' % (tag, freq, bnside, nside_standard))
		plt.plot(la.norm(used_common_ubls, axis=-1) * freq / C, np.sum(2. / np.real(cNi), axis=-1)[:, 0] ** .5 / la.norm(fullsim_vis, axis=-1)[:, 0], 'b+', label='noise error(%s-%s-dipole-beam_weight-bnside-%s-nside_standard-%s)' % (tag, freq, bnside, nside_standard))
		plt.plot(la.norm(used_common_ubls, axis=-1) * freq / C, la.norm(fullsim_vis - cdynamicmodel, axis=-1)[:, 0] / la.norm(fullsim_vis, axis=-1)[:, 0], 'r+', label='dynamic pixel error(%s-%s-dipole-beam_weight-bnside-%s-nside_standard-%s)' % (tag, freq, bnside, nside_standard))
		plt.legend(loc=0)
		plt.xlabel('Baseline Length (wavelength)')
		plt.ylabel('Relative RMS Error')
		plt.savefig(script_dir + '/../Output/noise_and_dynamic_pixel_error-%s-dipole-beam_weight-bnside-%s-nside_standard-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s.png' % (tag, bnside, nside_standard, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none'))
		plt.show(block=False)
# plt.gcf().clear()
except:
	print ('Error when Plotting Comparison of Full and Dyna Simulation Results.')

vis_normalization = get_vis_normalization(data, stitch_complex_data(fullsim_vis), data_shape=data_shape) if not Only_AbsData else get_vis_normalization(stitch_complex_data(data), stitch_complex_data(fullsim_vis), data_shape=data_shape)
print "Normalization from visibilities", vis_normalization

Del = True
if Del:
	try:
		del (cdata)
		del (cNi)
		del (cdynamicmodel)
	except:
		pass

sys.stdout.flush()

##renormalize the model
Recompute_fullsim_vis = False
if Recompute_fullsim_vis:
	fullsim_vis, autocorr_vis, autocorr_vis_normalized = Simulate_Visibility_mfreq(full_sim_filename_mfreq=full_sim_filename, sim_vis_xx_filename_mfreq=sim_vis_xx_filename, sim_vis_yy_filename_mfreq=sim_vis_yy_filename, Multi_freq=False, Multi_Sin_freq=False, used_common_ubls=used_common_ubls,
	                                                                               flist=None, freq_index=None, freq=[freq, freq], equatorial_GSM_standard_xx=equatorial_GSM_standard, equatorial_GSM_standard_yy=equatorial_GSM_standard, equatorial_GSM_standard_mfreq_xx=None, equatorial_GSM_standard_mfreq_yy=None, beam_weight=beam_weight,
	                                                                               C=299.792458, nUBL_used=None, nUBL_used_mfreq=None,
	                                                                               nt_used=None, nside_standard=nside_standard, nside_start=None, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)

fake_solution *= vis_normalization  # GSM Masked and being Normalized (abs calibration), Clean
clean_sim_data *= vis_normalization  # Dynamic Simulated, Clean, being Normalized (abs calibration)
fullsim_vis *= vis_normalization  # Full Simulated, Clean, being Normalized (abs calibration)
sim_data = (stitch_complex_data(fullsim_vis) + np.random.randn(len(data)) / Ni ** .5) if not Only_AbsData else (fullsim_vis.flatten() + np.random.randn(len(data)) / Ni ** .5)  # Full Simulated, being Normalized (abs calibration), Noise
# sim_data = (clean_sim_data + np.random.randn(len(data)) / Ni ** .5) if not Only_AbsData else (fullsim_vis.flatten() + np.random.randn(len(data)) / Ni ** .5)  # Full Simulated, being Normalized (abs calibration), Noise
# "data" is Calibrated Full Simulated Visibilities

# add additive term
if fit_for_additive:
	sim_data.shape = (2, nUBL_used, 2, nt_used) if not Only_AbsData else (nUBL_used, 2, nt_used)
	sim_additive = (np.random.randn(2, nUBL_used, 2) * np.median(np.abs(data)) / 2.) if not Only_AbsData else (np.random.randn(nUBL_used, 2) * np.median(data) / 2.)
	sim_data = sim_data + np.array([np.outer(sim_additive[..., p], autocorr_vis_normalized[p]).reshape((2, nUBL_used, nt_used)) for p in range(2)]).transpose((1, 2, 0, 3)) if not Only_AbsData \
		else sim_data + np.array([np.outer(sim_additive[..., p], autocorr_vis_normalized[p]).reshape((nUBL_used, nt_used)) for p in range(2)]).transpose((1, 0, 2))  # sim_additive[..., None]
	sim_data = sim_data.flatten()

# Real_Visibility = True # Only use Real parts of visibilies to calculate.
if Real_Visibility:
	if len(data) == 2 * 2 * nUBL_used * nt_used:
		data = data[:len(data) / 2]
	if len(sim_data) == 2 * 2 * nUBL_used * nt_used:
		sim_data = sim_data[:len(sim_data) / 2]
	if len(clean_sim_data) == 2 * 2 * nUBL_used * nt_used:
		clean_sim_data = clean_sim_data[:len(clean_sim_data) / 2]
	if len(Ni) == 2 * 2 * nUBL_used * nt_used:
		Ni = Ni[:len(Ni) / 2]
	if len(A) == 2 * 2 * nUBL_used * nt_used:
		A = A[:len(A) / 2]
	if len(additive_term) == 2 * 2 * nUBL_used * nt_used:
		additive_term = additive_term[:len(additive_term) / 2]

# compute AtNi.y
AtNi_data = np.transpose(A).dot((data * Ni).astype(A.dtype))
AtNi_sim_data = np.transpose(A).dot((sim_data * Ni).astype(A.dtype))
AtNi_clean_sim_data = np.transpose(A).dot((clean_sim_data * Ni).astype(A.dtype))

# compute S
print "computing S...",
sys.stdout.flush()
timer = time.time()

# diagonal of S consists of S_diag_I and S-diag_add
sys.stdout.flush()

S_type_list = {}
for i in range(100):
	S_type_list['min' + str(i) + 'I'] = 2.5 * 10 ** (i + 1)
	S_type_list['max' + str(i) + 'I'] = 2.5 * 10 ** (-i)
S_type_list['lowI'] = 25.
S_type_list['minI'] = 250
S_type_list['maxI'] = 0.25
# S_type_list['I'] = 2.5


if S_type == 'none':
	S_diag = np.ones(Ashape1) * np.max(equatorial_GSM_standard) ** 2 * 1.e12
else:
	I_supress = 1.
	for id_key, key in enumerate(S_type_list.keys()):
		if key in S_type:
			I_supress = S_type_list[key]
			break
	print('I_supress: %s--%s' % (key, I_supress))
	if 'Iuniform' in S_type:
		S_diag_I = (np.median(equatorial_GSM_standard) * sizes) ** 2 / I_supress
	else:
		S_diag_I = fake_solution_map ** 2 / I_supress  # np.array([[1+pol_frac,0,0,1-pol_frac],[0,pol_frac,pol_frac,0],[0,pol_frac,pol_frac,0],[1-pol_frac,0,0,1+pol_frac]]) / 4 * (2*sim_x_clean[i])**2
	
	if not Real_Visibility:
		data_max = np.transpose(np.percentile(np.abs(data.reshape((2, nUBL_used, 2, nt_used))), 95, axis=-1), (1, 2, 0)).flatten() if not Only_AbsData else np.transpose(np.percentile(np.abs(data.reshape((nUBL_used, 2, nt_used))), 95, axis=-1), (0, 1)).flatten()
	else:
		data_max = np.transpose(np.percentile(np.abs(data.reshape((1, nUBL_used, 2, nt_used))), 95, axis=-1), (1, 2, 0)).flatten()
	if 'min2add' in S_type:
		add_supress = 1000000.
	elif 'minadd' in S_type:
		add_supress = 10000.
	elif 'lowadd' in S_type:
		add_supress = 100.
	else:
		add_supress = 1
	
	if 'adduniform' in S_type:
		S_diag_add = np.ones(nUBL_used * 4) * np.median(data_max) ** 2 / add_supress
	else:
		S_diag_add = data_max ** 2 / add_supress
	
	if not fit_for_additive:
		S_diag = S_diag_I.astype('complex128')
	else:
		S_diag = np.concatenate((S_diag_I, S_diag_add)).astype('complex128')
	print "Done."
	print "%f minutes used" % (float(time.time() - timer) / 60.)
	sys.stdout.flush()

# compute (AtNiA+Si)i
precision = 'complex128'
AtNiAi_tag = 'AtNiASii' + ('RV' if Real_Visibility else '') + "_u%i_t%i_mtbin%s-mfbin%s-tbin%s_p%i_n%i_%i_b%i" % (nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', valid_npix, nside_start, nside_standard, bnside)
if not fit_for_additive:
	AtNiAi_version = 0.3
elif crosstalk_type == 'autocorr':
	AtNiAi_version = 0.2
else:
	AtNiAi_version = 0.1
if pre_ampcal:
	AtNiAi_version += 1.
# rcond_list = 10.**np.arange(-9., -2., 1.)
# Parallel_AtNiA = True

# nchunk_AtNiA = 96
# nchunk_AtNiA_maxcut = 4
# UseDot = True
AtNiAi_candidate_files = glob.glob(datadir + tag + AtNiAi_tag + '_S%s_RE*_N%s_v%.1f' % (S_type, vartag, AtNiAi_version) + A_filename)
if len(AtNiAi_candidate_files) > 0 and not force_recompute_AtNiAi and not force_recompute and not force_recompute_S and not AtNiA_only:
	rcond = 10 ** min([float(fn.split('_RE')[1].split('_N')[0]) for fn in AtNiAi_candidate_files])
	
	AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f' % (S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
	AtNiAi_path = datadir + tag + AtNiAi_filename
	
	print "Reading Regularized AtNiAi...",
	sys.stdout.flush()
	AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision)
else:
	if os.path.isfile(AtNiA_path) and not force_recompute:
		print "Reading AtNiA...",
		sys.stdout.flush()
		AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
	else:
		print "Allocating AtNiA..."
		sys.stdout.flush()
		timer = time.time()
		AtNiA = np.zeros((A.shape[1], A.shape[1]), dtype=precision)
		print "Computing AtNiA...", datetime.datetime.now()
		sys.stdout.flush()
		if Parallel_AtNiA:
			for i in np.arange(0, nchunk_AtNiA_maxcut, nchunk_AtNiA_step):
				try:
					AtNiA = ATNIA_small_parallel(A, Ni, AtNiA, nchunk=int(nchunk_AtNiA * (i + 1)), dot=UseDot)
					break
				except:
					if (i + nchunk_AtNiA_step) >= nchunk_AtNiA_maxcut:
						ATNIA(A, Ni, AtNiA, nchunk=nchunk, dot=UseDot)
					continue
		else:
			ATNIA(A, Ni, AtNiA, nchunk=nchunk, dot=UseDot)

		print ">>>>>>>>> %f minutes used <<<<<<<<<<" % (float(time.time() - timer) / 60.)
		sys.stdout.flush()
		AtNiA.tofile(AtNiA_path)
	if AtNiA_only:
		sys.exit(0)
	
	# Del_A = False
	if Del_A:
		del (A)
	AtNiA_diag = np.diagonal(AtNiA)
	print "Computing Regularized AtNiAi, %s, expected time %f min" % (datetime.datetime.now(), 88. * (len(S_diag) / 4.6e4) ** 3.),
	sys.stdout.flush()
	timer = time.time()
	# if la.norm(S) != la.norm(np.diagonal(S)):
	#     raise Exception("Non-diagonal S not supported yet")
	
	for rcond in rcond_list:
		# add Si on top of AtNiA without renaming AtNiA to save memory
		maxAtNiA = np.max(AtNiA)
		AtNiA.shape = (len(AtNiA) ** 2)
		if Add_S_diag:
			if not Only_AbsData:
				AtNiA[::len(S_diag) + 1] += 1. / S_diag
			else:
				AtNiA[::len(S_diag) + 1] += (1. / S_diag + 1.j / S_diag)
		
		print 'trying', rcond,
		sys.stdout.flush()
		try:
			AtNiAi_filename = AtNiAi_tag + '_S%s_RE%.1f_N%s_v%.1f' % (S_type, np.log10(rcond), vartag, AtNiAi_version) + A_filename
			AtNiAi_path = datadir + tag + AtNiAi_filename
			if Add_Rcond:
				if not Only_AbsData:
					AtNiA[::len(S_diag) + 1] += maxAtNiA * rcond
				else:
					AtNiA[::len(S_diag) + 1] += (maxAtNiA * rcond + 1.j * maxAtNiA * rcond)
					
			AtNiA.shape = (Ashape1, Ashape1)
			AtNiAi = sv.InverseCholeskyMatrix(AtNiA).astype(precision)
			# del (AtNiA)
			try:
				AtNiAi.tofile(AtNiAi_path, overwrite=True)
			except:
				print('AtNiAi cannot be saved to file via AiNiAi_path.')
			del (AtNiA)
			print "%f minutes used" % (float(time.time() - timer) / 60.)
			print "regularization stength", (maxAtNiA * rcond) ** -.5, "median GSM ranges between", np.median(equatorial_GSM_standard) * min(sizes), np.median(equatorial_GSM_standard) * max(sizes)
			break
		except:
			AtNiA[::len(S_diag) + 1] -= maxAtNiA * rcond
			continue

sys.stdout.flush()

#####apply wiener filter##############
print ("Applying Regularized AtNiAi...")

sys.stdout.flush()
if not Only_AbsData:
	w_solution = AtNiAi.dotv(AtNi_data)
	w_GSM = AtNiAi.dotv(AtNi_clean_sim_data)
	w_sim_sol = AtNiAi.dotv(AtNi_sim_data)
else:
	w_solution = np.diagonal(np.outer(AtNiAi.dotv(AtNi_data), (AtNiAi.dotv(AtNi_data)).conjugate()))**0.5
	w_GSM = np.diagonal(np.outer(AtNiAi.dotv(AtNi_clean_sim_data), (AtNiAi.dotv(AtNi_clean_sim_data)).conjugate()))**0.5
	w_sim_sol = np.diagonal(np.outer(AtNiAi.dotv(AtNi_sim_data), (AtNiAi.dotv(AtNi_sim_data)).conjugate()))**0.5
	
print "Memory usage: %.3fMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000)
sys.stdout.flush()

del (AtNiAi)

try:
	A.shape
except:
	try:
		try:
			del (beam_heal_equ_x_mfreq)
			del (beam_heal_equ_y_mfreq)
			del (equatorial_GSM_standard_mfreq)
		except:
			pass
		if os.path.isfile(A_path + 'tmpre') or os.path.isfile(A_path):
			A = get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=False, Compute_A=True, Only_AbsData=Only_AbsData, A_path=A_path, A_RE_path=(A_path + 'tmpre'), A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=False, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight,
	                           flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=None, beam_weight=beam_weight, valid_npix=valid_npix,
	                            used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
		else:
			A, beam_weight, gsm_beamweighted, nside_distribution, final_index, thetas, phis, sizes, abs_thresh, npix, valid_pix_mask, valid_npix, fake_solution_map, fake_solution_original = \
				get_A_multifreq(fit_for_additive=fit_for_additive, additive_A=additive_A, force_recompute=False, Compute_A=True, Only_AbsData=Only_AbsData, A_path='', A_RE_path='', A_got=None, A_version=A_version, AllSky=False, MaskedSky=True, Synthesize_MultiFreq=False, thresh=thresh, valid_pix_thresh=valid_pix_thresh, Use_BeamWeight=Use_BeamWeight,
				                flist=flist, Flist_select=Flist_select, Flist_select_index=Flist_select_index, Reference_Freq_Index=None, Reference_Freq=[freq, freq], equatorial_GSM_standard=equatorial_GSM_standard, equatorial_GSM_standard_mfreq=None, beam_weight=beam_weight,
				                used_common_ubls=used_common_ubls_sinfreq, nt_used=nt_used, nside_standard=nside_standard, nside_start=nside_start, nside_beamweight=nside_beamweight, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, beam_heal_equ_x_mfreq=None, beam_heal_equ_y_mfreq=None, lsts=lsts)
		print('Use get_A_multifreq() to calculate A.')
	except:
		try:
			del (beam_heal_equ_x_mfreq)
			del (beam_heal_equ_y_mfreq)
			del (equatorial_GSM_standard_mfreq)
		except:
			pass
		
		if not fit_for_additive:
			A = get_A(A_path=A_path, force_recompute=force_recompute, Only_AbsData=Only_AbsData, nUBL_used=nUBL_used, nt_used=nt_used, valid_npix=valid_npix, thetas=thetas, phis=phis, used_common_ubls=used_common_ubls, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, lsts=lsts, freq=freq)
		else:
			A = get_A(additive_A=additive_A, A_path=A_path, force_recompute=force_recompute, Only_AbsData=Only_AbsData, nUBL_used=nUBL_used, nt_used=nt_used, valid_npix=valid_npix, thetas=thetas, phis=phis, used_common_ubls=used_common_ubls, beam_heal_equ_x=beam_heal_equ_x, beam_heal_equ_y=beam_heal_equ_y, lsts=lsts, freq=freq)
		print('Use get_A() instead get_A_multifreq(), so that global A is generated directly to save memory.')

if Real_Visibility:
	if len(A) == 2 * 2 * nUBL_used * nt_used:
		A = A[:len(A) / 2]


best_fit = A.dot(w_solution.astype(A.dtype))  # Reversely-Calculated-masked-GSM Dynamically simulated Visibilities.
best_fit_no_additive = A[..., :valid_npix].dot(w_solution[:valid_npix].astype(A.dtype))

sim_best_fit = A.dot(w_sim_sol.astype(A.dtype))
sim_best_fit_no_additive = A[..., :valid_npix].dot(w_sim_sol[:valid_npix].astype(A.dtype))

if Only_AbsData:
	best_fit = np.concatenate((np.real(best_fit), np.imag(best_fit))).astype('complex128')
	best_fit_no_additive = np.concatenate((np.real(best_fit_no_additive), np.imag(best_fit_no_additive))).astype('complex128')
	sim_best_fit = np.concatenate((np.real(sim_best_fit), np.imag(sim_best_fit))).astype('complex128')
	sim_best_fit_no_additive = np.concatenate((np.real(sim_best_fit_no_additive), np.imag(sim_best_fit_no_additive))).astype('complex128')
	

try:
	if plot_data_error:
		if Only_AbsData:
			qaz_model = (np.concatenate((np.real(clean_sim_data), np.imag(clean_sim_data)))).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])  # Dynamic Simulated, Clean, being Normalized    # * vis_normalization
			qaz_data = np.concatenate((np.real(data), np.imag(data))).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])  # Full Simulated, Calibrated, reference for normalization
			if pre_calibrate:
				# qaz_add = np.concatenate((np.real(additive_term), np.imag(additive_term))).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
				qaz_add = np.copy(additive_term).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
		elif Real_Visibility:
			qaz_model = (clean_sim_data).reshape(1, data_shape['xx'][0], 2, data_shape['xx'][1])  # Dynamic Simulated, Clean, being Normalized    # * vis_normalization
			qaz_data = np.copy(data).reshape(1, data_shape['xx'][0], 2, data_shape['xx'][1])  # Full Simulated, Calibrated, reference for normalization
			if pre_calibrate:
				qaz_add = np.copy(additive_term).reshape(1, data_shape['xx'][0], 2, data_shape['xx'][1])
		else:
			qaz_model = (clean_sim_data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])  # Dynamic Simulated, Clean, being Normalized    # * vis_normalization
			qaz_data = np.copy(data).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])  # Full Simulated, Calibrated, reference for normalization
			if pre_calibrate:
				qaz_add = np.copy(additive_term).reshape(2, data_shape['xx'][0], 2, data_shape['xx'][1])
		# pncol = min(int(60. / (srt[-1] - srt[0])), 12) if nt_used > 1 else (len(ubl_sort['x']) / 2)
		# us = ubl_sort['x'][::len(ubl_sort['x']) / pncol] if len(ubl_sort['x']) / pncol >= 1 else ubl_sort['x']  # [::max(1, len(ubl_sort['x'])/70)]
		us = ubl_sort['x'][::len(ubl_sort['x']) / 24]
		if not Real_Visibility:
			best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
			best_fit_no_additive.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
			sim_best_fit.shape = (2, data_shape['xx'][0], 2, data_shape['xx'][1])
			ri = 1
		else:
			best_fit.shape = (1, data_shape['xx'][0], 2, data_shape['xx'][1])
			best_fit_no_additive.shape = (1, data_shape['xx'][0], 2, data_shape['xx'][1])
			sim_best_fit.shape = (1, data_shape['xx'][0], 2, data_shape['xx'][1])
			ri = 0
		# ri = 1 if not Real_Visibility else 0
		
		figure_W = {}
		for p in range(2):
			
			for nu, u in enumerate(us):
				plt.figure(8000 + 10 * p + nu)
				# plt.subplot(6, (len(us) + 5) / 6, nu + 1)
				# plt.errorbar(range(nt_used), qaz_data[ri, u, p], yerr=Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p]**-.5)
				figure_W[1], = plt.plot(qaz_data[ri, u, p], '+')
				figure_W[2], = plt.plot(qaz_model[ri, u, p], '-')
				figure_W[3], = plt.plot(best_fit[ri, u, p], '.')
				figure_W[4], = plt.plot(sim_best_fit[ri, u, p])
				if pre_calibrate:
					figure_W[5], = plt.plot(qaz_add[ri, u, p], 'x')
				if fit_for_additive:
					figure_W[6], = plt.plot(autocorr_vis_normalized[p] * sol2additive(w_solution, valid_npix=valid_npix, nUBL_used=nUBL_used)[p, u, ri])
				figure_W[7], = plt.plot(best_fit[ri, u, p] - qaz_data[ri, u, p])
				if not Real_Visibility:
					figure_W[8], = plt.plot(Ni.reshape((2, nUBL_used, 2, nt_used))[ri, u, p] ** -.5) if not Only_AbsData else plt.plot(np.concatenate((np.real(Ni), np.imag(Ni))).reshape((2, nUBL_used, 2, nt_used))[ri, u, p] ** -.5)
				else:
					figure_W[8], = plt.plot(Ni.reshape((1, nUBL_used, 2, nt_used))[ri, u, p] ** -.5)
				if pre_calibrate:
					data_range = np.max([np.max(np.abs(qaz_data[ri, u, p])), np.max(np.abs(qaz_model[ri, u, p])),
					                     np.max(np.abs(best_fit[ri, u, p])), np.max(np.abs(sim_best_fit[ri, u, p])), np.max(
							np.abs((best_fit[ri, u, p] - qaz_data[
								ri, u, p])))])  # np.max(np.abs(qaz_add[ri, u, p])), #, 5 * np.max(np.abs(fun(cNi[u, p])))
					plt.legend(
						handles=[figure_W[1], figure_W[2], figure_W[3], figure_W[4], figure_W[5], figure_W[7], figure_W[8]],
						labels=['qaz_data', 'qaz_model', 'best_fit', 'sim_best_fit', 'additive', 'best_fit - qaz_data',
						        'noise'], loc=0)
				else:
					data_range = np.max([np.max(np.abs(qaz_data[ri, u, p])), np.max(np.abs(qaz_model[ri, u, p])),
					                     np.max(np.abs(best_fit[ri, u, p])), np.max(np.abs(sim_best_fit[ri, u, p])), np.max(
							np.abs((best_fit[ri, u, p] - qaz_data[ri, u, p])))])  # , 5 * np.max(np.abs(fun(cNi[u, p])))
					plt.legend(handles=[figure_W[1], figure_W[2], figure_W[3], figure_W[4], figure_W[7], figure_W[8]],
					           labels=['qaz_data', 'qaz_model', 'best_fit', 'sim_best_fit', 'best_fit - qaz_data', 'noise'],
					           loc=0)
				plt.yscale('symlog')
				# plt.ylim([-1.05 * data_range, 1.05 * data_range])
				# plt.title("%.1f,%.1f,%.1e"%(used_common_ubls[u, 0], used_common_ubls[u, 1], la.norm(best_fit[ri, u, p] - qaz_data[ri, u, p])))
				plt.title("Wiener Filter %s Baseline-%.1f_%.1f results on srtime" % (
					['xx', 'yy'][p], used_common_ubls[u, 0], used_common_ubls[u, 1]))
				plt.savefig(
					script_dir + '/../Output/%s-Baseline-%.1f_%.1f-dipole-precal_data_error-WienerFilter_Vis-%s-%.2fMHz-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s.pdf' % (tag, used_common_ubls[u, 0], used_common_ubls[u, 1], ['xx', 'yy'][p], freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard))
				plt.show(block=False)
except:
	print('Error when Plotting Various Methods.')

sys.stdout.flush()


def plot_IQU(solution, title, col, shape=(2, 3), coord='C'):
	# Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
	# I = Es[0] + Es[3]
	# Q = Es[0] - Es[3]
	# U = Es[1] + Es[2]
	I = sol2map(solution, valid_npix=valid_npix, npix=npix, valid_pix_mask=valid_pix_mask, final_index=final_index, sizes=sizes)
	plotcoordtmp = coord
	hpv.mollview(np.log10(I), min=0, max=4, coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))


# if col == shape[0] * shape[1]:
# plt.show(block=False)

def plot_IQU_unlimit(solution, title, col, shape=(2, 3), coord='C'):
	# Es=solution[np.array(final_index).tolist()].reshape((4, len(final_index)/4))
	# I = Es[0] + Es[3]
	# Q = Es[0] - Es[3]
	# U = Es[1] + Es[2]
	I = sol2map(solution, valid_npix=valid_npix, npix=npix, valid_pix_mask=valid_pix_mask, final_index=final_index, sizes=sizes)
	plotcoordtmp = coord
	hpv.mollview(np.log10(I), coord=plotcoordtmp, title=title, nest=True, sub=(shape[0], shape[1], col))


# if col == shape[0] * shape[1]:
# plt.show(block=False)

sys.stdout.flush()

rescale_factor = np.max(np.abs(fake_solution)) / np.max(np.abs(w_solution))

crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(900 + crd)
	crd += 10
	plot_IQU_unlimit(np.abs(w_GSM), 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU_unlimit(np.abs(w_sim_sol), 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU_unlimit(np.abs(w_solution), 'wienered solution(data)', 3, coord=coord)
	plot_IQU_unlimit(np.abs(fake_solution), 'True GSM masked', 4, coord=coord)
	plot_IQU_unlimit(np.abs(w_sim_sol - w_GSM + fake_solution), 'combined sim solution', 5, coord=coord)
	plot_IQU_unlimit(np.abs(w_solution - w_GSM + fake_solution), 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-none-unlimit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()


crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(9000000 + crd)
	crd += 10
	plot_IQU_unlimit(np.abs(w_GSM) * rescale_factor, 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU_unlimit(np.abs(w_sim_sol) * rescale_factor, 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU_unlimit(np.abs(w_solution) * rescale_factor, 'wienered solution(data)', 3, coord=coord)
	plot_IQU_unlimit(np.abs(fake_solution), 'True GSM masked', 4, coord=coord)
	plot_IQU_unlimit(np.abs((w_sim_sol - w_GSM) * rescale_factor + fake_solution), 'combined sim solution', 5, coord=coord)
	plot_IQU_unlimit(np.abs((w_solution - w_GSM) * rescale_factor + fake_solution), 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-%s-unlimit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, rescale_factor, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()

crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(9500000 + crd)
	crd += 10
	plot_IQU(np.abs(w_GSM) * rescale_factor, 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU(np.abs(w_sim_sol) * rescale_factor, 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU(np.abs(w_solution) * rescale_factor, 'wienered solution(data)', 3, coord=coord)
	plot_IQU(np.abs(fake_solution), 'True GSM masked', 4, coord=coord)
	plot_IQU(np.abs((w_sim_sol - w_GSM) * rescale_factor + fake_solution), 'combined sim solution', 5, coord=coord)
	plot_IQU(np.abs((w_solution - w_GSM) * rescale_factor + fake_solution), 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-%s-limit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, rescale_factor, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()


crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(950 + crd)
	crd += 10
	plot_IQU_unlimit((w_GSM + np.abs(w_GSM)) * 0.5 * rescale_factor + 1.e-6, 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU_unlimit((w_sim_sol + np.abs(w_sim_sol)) * 0.5 * rescale_factor + 1.e-6, 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU_unlimit((w_solution + np.abs(w_solution)) * 0.5 * rescale_factor + 1.e-6, 'wienered solution(data)', 3, coord=coord)
	plot_IQU_unlimit((fake_solution + np.abs(fake_solution)) * 0.5, 'True GSM masked', 4, coord=coord)
	plot_IQU_unlimit((((w_sim_sol - w_GSM) * rescale_factor + fake_solution) + np.abs((w_sim_sol - w_GSM) * rescale_factor + fake_solution)) * 0.5 + 1.e-6, 'combined sim solution', 5, coord=coord)
	plot_IQU_unlimit((((w_solution - w_GSM) * rescale_factor + fake_solution) + np.abs((w_solution - w_GSM) * rescale_factor + fake_solution)) * 0.5 + 1.e-6, 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-%s-Deg-unlimit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, rescale_factor, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()

crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(90000 + crd)
	crd += 10
	plot_IQU(np.abs(w_GSM), 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU(np.abs(w_sim_sol), 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU(np.abs(w_solution), 'wienered solution(data)', 3, coord=coord)
	plot_IQU(np.abs(fake_solution), 'True GSM masked', 4, coord=coord)
	plot_IQU(np.abs(w_sim_sol - w_GSM + fake_solution), 'combined sim solution', 5, coord=coord)
	plot_IQU(np.abs(w_solution - w_GSM + fake_solution), 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-none-limit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()


crd = 0
for coord in ['C', 'CG']:
	# plt.clf()
	plt.figure(9000 + crd)
	crd += 10
	plot_IQU(np.abs(w_GSM / vis_normalization), 'wienered GSM', 1, coord=coord)  # (clean dynamic_data)
	plot_IQU(np.abs(w_sim_sol / vis_normalization), 'wienered simulated solution', 2, coord=coord)  # (~noise+data)
	plot_IQU(np.abs(w_solution), 'wienered solution(data)', 3, coord=coord)
	plot_IQU(np.abs(fake_solution / vis_normalization), 'True GSM masked', 4, coord=coord)
	plot_IQU(np.abs((w_sim_sol - w_GSM + fake_solution) / vis_normalization), 'combined sim solution', 5, coord=coord)
	plot_IQU(np.abs((w_solution - w_GSM + fake_solution) / vis_normalization), 'combined solution', 6, coord=coord)
	plt.savefig(script_dir + '/../Output/results_wiener_renormalized-%s-%s-%sMHz-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-rescale-none-limit-S-%s-recond-%s.png' % (coord, tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, S_type, rcond if Add_Rcond else 'none'))
	plt.show(block=False)
# plt.gcf().clear()

# def sol4map(sol):
#	solx = sol[:valid_npix]
#	full_sol = np.zeros(npix)
#	full_sol[valid_pix_mask] = solx #/ sizes
#	return full_sol[final_index]
#
# I = sol4map(np.real(w_GSM))
# hpv.mollview(np.log10(I), coord='CG', nest=True)
# plt.show()

if not Only_AbsData:
	if not Real_Visibility:
		error = data.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1])) - best_fit
		chi = error * (Ni.reshape((2, data_shape['xx'][0], 2, data_shape['xx'][1]))) ** .5
	else:
		error = data.reshape((1, data_shape['xx'][0], 2, data_shape['xx'][1])) - best_fit
		chi = error * (Ni.reshape((1, data_shape['xx'][0], 2, data_shape['xx'][1]))) ** .5
else:
	error = data.reshape((data_shape['xx'][0], 2, data_shape['xx'][1])) - best_fit
	chi = error * (Ni.reshape((data_shape['xx'][0], 2, data_shape['xx'][1]))) ** .5
print "chi^2 = %.3e, data points %i, pixels %i" % (la.norm(chi) ** 2, len(data), valid_npix)
if not Real_Visibility:
	print "re/im chi2 %.3e, %.3e" % (la.norm(chi[0]) ** 2, la.norm(chi[1]) ** 2)
print "xx/yy chi2 %.3e, %.3e" % (la.norm(chi[:, :, 0]) ** 2, la.norm(chi[:, :, 1]) ** 2)
# plt.clf()
plt.figure(120)
plt.subplot(2, 2, 1)
plt.plot([la.norm(error[:, u]) for u in ubl_sort['x']])
plt.title('Error Norm-bsl')
plt.yscale('symlog')
plt.subplot(2, 2, 2)
plt.plot([la.norm(chi[:, u]) for u in ubl_sort['x']])
plt.title('Chi Norm-bsl')
plt.yscale('symlog')
plt.subplot(2, 2, 3)
plt.plot(lsts, [la.norm(error[..., t]) for t in range(error.shape[-1])])
plt.title('Error Norm-t')
plt.yscale('symlog')
plt.subplot(2, 2, 4)
plt.plot(lsts, [la.norm(chi[..., t]) for t in range(error.shape[-1])])
plt.title('Chi Norm-bsl')
plt.yscale('symlog')
plt.savefig(script_dir + '/../Output/chi-%s-%s-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-%s-recond-%s.png' % (tag, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, S_type, rcond if Add_Rcond else 'none'))
plt.show(block=False)
# plt.gcf().clear()

try:
	print('Additive_sol: %s' % additive_sol[:2])
	print('Rescale_factor: %s' % rescale_factor)
	print ("regularization stength", (maxAtNiA * rcond) ** -.5, "median GSM ranges between", np.median(equatorial_GSM_standard) * min(sizes), np.median(equatorial_GSM_standard) * max(sizes))
except:
	pass

sys.stdout.flush()

# S_type = 'none'
# point spread function:
try:
	if True:  # and S_type == 'none':
		print "Reading Regularized AtNiAi...",
		sys.stdout.flush()
		AtNiAi = sv.InverseCholeskyMatrix.fromfile(AtNiAi_path, len(S_diag), precision)
		
		AtNiA_tag = 'AtNiA_N%s' % vartag
		if not fit_for_additive:
			AtNiA_tag += "_noadd"
		elif crosstalk_type == 'autocorr':
			AtNiA_tag += "_autocorr"
		if pre_ampcal:
			AtNiA_tag += "_ampcal"
		AtNiA_filename = AtNiA_tag + A_filename
		AtNiA_path = datadir + tag + AtNiA_filename
		print "Reading AtNiA...",
		sys.stdout.flush()
		AtNiA = np.fromfile(AtNiA_path, dtype=precision).reshape((Ashape1, Ashape1))
		
		iplot = 0
		valid_thetas_phis = np.array(zip(thetas, phis))
		full_thetas, full_phis = hpf.pix2ang(nside_standard, range(hpf.nside2npix(nside_standard)), nest=True)
		# plt.clf()
		
		for theta in np.arange(0, PI * .9, PI / 6.):
			for phi in np.arange(0, TPI, PI / 3.):
				
				# choose_plots = [1, 6, 12, 18, 24, 30]
				choose_plots = [0, 1, 4, 6, 8, 12, 16, 20, 24, 28, 30, 32, 33, 35]
				
				if iplot in choose_plots:
					np.argmin(la.norm(valid_thetas_phis - [theta, phi], axis=-1))
					point_vec = np.zeros_like(fake_solution).astype('complex128')
					
					point_vec[np.argmin(la.norm(valid_thetas_phis - [theta, phi], axis=-1))] = 1
					if not Only_AbsData:
						spreaded = sol2map(AtNiAi.dotv(AtNiA.dot(point_vec)) + 1.e-6, valid_npix=valid_npix, npix=npix, valid_pix_mask=valid_pix_mask, final_index=final_index, sizes=sizes)
					else:
						spreaded = sol2map(np.diagonal(np.outer(AtNiAi.dotv(AtNiA.dot(point_vec)), (AtNiAi.dotv(AtNiA.dot(point_vec))).conjugate()))**0.5, valid_npix=valid_npix, npix=npix, valid_pix_mask=valid_pix_mask, final_index=final_index, sizes=sizes)
					spreaded /= np.max(spreaded)
					fwhm_mask = np.abs(spreaded) >= .5
					masked_max_ind = np.argmax(spreaded[fwhm_mask])
					fwhm_thetas = full_thetas[fwhm_mask]
					fwhm_phis = full_phis[fwhm_mask]
					# rotate angles to center around PI/2 0
					fwhm_thetas, fwhm_phis = hpr.Rotator(rot=[fwhm_phis[masked_max_ind], PI / 2 - fwhm_thetas[masked_max_ind], 0], deg=False)(fwhm_thetas, fwhm_phis)
					if np.array(fwhm_thetas).shape is ():
						fwhm_thetas = np.array([fwhm_thetas])
						fwhm_phis = np.array([fwhm_phis])
					print fwhm_thetas[masked_max_ind], fwhm_phis[masked_max_ind]  # should print 1.57079632679 0.0 if rotation is working correctly
					
					fwhm_theta = max(fwhm_thetas) - min(fwhm_thetas)
					phi_offset = fwhm_phis[masked_max_ind] - PI
					fwhm_phis = (fwhm_phis - phi_offset) % TPI + phi_offset
					fwhm_phi = max(fwhm_phis) - min(fwhm_phis)
					plt.figure(1300 + iplot)
					# hpv.mollview(np.log10(np.abs(spreaded)), min=-3, max=0, nest=True, coord='CG', title='FWHM = %.3f'%((fwhm_theta*fwhm_phi)**.5*180./PI), sub=(len(choose_plots), 1, choose_plots.index(iplot)+1))
					hpv.mollview(np.log10(np.abs(spreaded)), nest=True, coord='CG', title='FWHM = %.3f' % ((fwhm_theta * fwhm_phi) ** .5 * 180. / PI))
					plt.savefig(script_dir + '/../Output/spreaded_function-CG-%s-%s-%s-dipole-nubl%s-nt%s-mtbin%s-mfbin%s-tbin%s-bnside-%s-nside_standard-%s-%s-recond-%s.png' % (tag, iplot, freq, nUBL_used, nt_used, mocal_time_bin if Absolute_Calibration_dred_mfreq else '_none', mocal_freq_bin if Absolute_Calibration_dred_mfreq else '_none', precal_time_bin if pre_calibrate else '_none', bnside, nside_standard, S_type, rcond if Add_Rcond else 'none'))
					plt.show(block=False)
				# plt.gcf().clear()
				iplot += 1
except:
	print('No point spread function plotted.')

sys.stdout.flush()

Timer_End = time.time()
try:
	print('Programme Ends at: %s' % str(datetime.datetime.now()))
	print('>>>>>>>>>>>>>>>>>> Total Used Time: %s seconds. <<<<<<<<<<<<<<<<<<<<' % (Timer_End - Timer_Start))
except:
	print('No Used Time Printed.')


VariableMemory_Used = {}
for var, obj in locals().items():
	VariableMemory_Used[var] = sys.getsizeof(obj)
	# print (var, sys.getsizeof(obj))
print (sorted(VariableMemory_Used, key=VariableMemory_Used.__getitem__, reverse=True))


# Mac Code