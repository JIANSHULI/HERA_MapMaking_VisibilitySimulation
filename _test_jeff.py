import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as la
import scipy.special as ssp
import math as m
import cmath as cm
from wignerpy._wignerpy import wigner3j, wigner3jvec
from random import random
import healpy as hp
import healpy.pixelfunc as hpf
import simulate_visibility as sv
import matplotlib.pyplot as plt
import unittest, os


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.realpath(__file__)) + '/testing/'
        self.blmequ = sv.read_real_alm(self.test_dir + 'blm5.bin')
        self.blmax = 5
        self.freq = 150./300.*299.792458#my correct answers are computed with c=300,so im rescaing freq here to compensate
        self.zenithequ = sv.ctos(np.fromfile(self.test_dir + 'zenith.bin', dtype = 'float32'))[1:]
        self.blvequ = np.fromfile(self.test_dir + 'blv.bin', dtype = 'float32')
        self.correct_BB = sv.read_alm(self.test_dir + 'BB47.bin')
        self.correct_cm = np.fromfile(self.test_dir + 'cm47.bin', dtype = 'complex64')
        self.correct_result = np.fromfile(self.test_dir + 'final47.bin', dtype = 'complex64') * (len(self.correct_cm))**0.5 #correct result did not use correct normalization n**0.5 in fourier
        self.nside = 16
        healpix = np.zeros(12*self.nside**2)
        healpix[420] = 1
        self.alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(healpix), 3 * self.nside - 1)
        correct_alm = sv.convert_healpy_alm(np.fromfile(self.test_dir + 'datapoint_source_420_n16_alm47.bin', dtype='complex64'), 3 * 16 - 1)
        for key in self.alm.keys():
            self.assertEqual(self.alm[key].astype('complex64'), correct_alm[key])
        self.vs = sv.Visibility_Simulator()
        self.vs.initial_zenith = self.zenithequ
        self.vs.Blm = sv.expand_real_alm(self.blmequ)
        
        #print self.vs.Blm.keys()
        #print self.alm[(0,0)], self.alm[(1,1)]
    def test_BB(self):
        self.BB = self.vs.calculate_Bulm_jf(L=3*self.nside-1,freq=self.freq,d=self.blvequ,L1=self.blmax)
        self.assertEqual(len(self.correct_BB), len(self.BB))
        #self.assertEqual(self.correct_BB, self.BB)
        for l in range(3*self.nside):
            for mm in range(-l, l + 1):
                #print l, mm, self.correct_BB[(l,mm)], self.BB[(l,mm)]
                #self.assertEqual(self.correct_BB[(l,mm)], self.BB[(l,mm)])
                try:
                    self.assertAlmostEqual(la.norm((self.correct_BB[(l,mm)] - self.BB[(l,mm)])/self.correct_BB[(l,mm)]), 0, 2)
                except:
                    print l, mm, self.correct_BB[(l,mm)], self.BB[(l,mm)]
                    self.assertAlmostEqual(la.norm((self.correct_BB[(l,mm)] - self.BB[(l,mm)])/self.correct_BB[(l,mm)]), 0, 2)
    def test_visibility(self):
        #print self.zenithequ
        self.result = self.vs.calculate_visibility_jf(sv.expand_real_alm(self.alm), d=np.array([0,3,0]), freq=self.freq, tlist=np.arange(0,24,24./(6*self.nside-1)), L = 3*self.nside-1, verbose = False)
        #plt.plot(np.real(self.result), 'r--', np.real(self.correct_result), 'bs', np.imag(self.result), 'r--', np.imag(self.correct_result), 'bs')
        #plt.show()
        np.testing.assert_almost_equal(np.abs((self.result-self.correct_result)/self.correct_result), np.zeros(len(self.correct_result)), 5)
if __name__ == '__main__':
    unittest.main()
