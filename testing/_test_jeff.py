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
import simulate_visibilities.simulate_visibilities as sv
import matplotlib.pyplot as plt
import unittest, os, time, sys


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.blmequ = sv.read_real_alm(self.test_dir + 'blm5.bin')
        self.blmax = 5
        self.freq = 150./300.*299.792458#my correct answers are computed with c=300,so im rescaing freq here to compensate
        self.zenithequ = sv.ctos(np.fromfile(self.test_dir + 'zenith.bin', dtype = 'float32'))[1:]
        self.zenithequ[0] = np.pi/2 - self.zenithequ[0]
        self.zenithequ = np.array(self.zenithequ)[::-1]
        self.blvequ = np.fromfile(self.test_dir + 'blv.bin', dtype = 'float32')
        self.correct_BB = sv.read_alm(self.test_dir + 'BB47.bin')
        self.correct_cm = np.fromfile(self.test_dir + 'cm47.bin', dtype = 'complex64')
        self.correct_result = np.fromfile(self.test_dir + 'final47.bin', dtype = 'complex64')# * (len(self.correct_cm))**0.5 #correct result did not use correct normalization n**0.5 in fourier
        self.nside = 16
        healpix = np.zeros(12*self.nside**2)
        healpix[420] = 1
        healpix[752] = 1
        self.alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(healpix), 3 * self.nside - 1)

        self.vs = sv.Visibility_Simulator()
        self.vs.initial_zenith = self.zenithequ
        self.vs.Blm = sv.expand_real_alm(self.blmequ)

        #print self.vs.Blm.keys()
        #print self.alm[(0,0)], self.alm[(1,1)]
    def test_alm(self):
        correct_alm = sv.convert_healpy_alm(np.fromfile(self.test_dir + 'datapoint_source_420_752_n16_alm47.bin', dtype='complex64'), 3 * 16 - 1)

        for key in self.alm.keys():
            self.assertEqual(self.alm[key].astype('complex64'), correct_alm[key])
    def test_BB(self):
        timer = time.time()
        self.BB = self.vs.calculate_Bulm(L=3*self.nside-1,freq=self.freq,d=self.blvequ,L1=self.blmax)
        print "BB time: %f"%(float(time.time() - timer)/60)
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
        self.result = self.vs.calculate_visibility(sv.expand_real_alm(self.alm), d=np.array([0,3,0]), freq=self.freq, tlist=np.arange(0,24,24./(6*self.nside-1)), L = 3*self.nside-1, verbose = False)
        #plt.plot(np.real(self.result), 'r--', np.real(self.correct_result), 'bs', np.imag(self.result), 'r--', np.imag(self.correct_result), 'bs')
        #plt.show()
        np.testing.assert_almost_equal(np.abs((self.result-self.correct_result)/self.correct_result), np.zeros(len(self.correct_result)), 5)
    def test_visibility_fft(self):
        timer = time.time()
        self.result = self.vs.calculate_visibility(sv.expand_real_alm(self.alm), d=np.array([0,3,0]), freq=self.freq, nt=(6*self.nside-1), L = 3*self.nside-1, verbose = False)
        print "Total time: %f"%(float(time.time() - timer)/60)
        np.testing.assert_almost_equal(np.abs((self.result-self.correct_result)/self.correct_result), np.zeros(len(self.correct_result)), 5)

class TestGSM(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.blmequ = sv.read_real_alm(self.test_dir + 'bx125.195.bin')
        self.blmax = 23
        self.pol = 'xx'
        beam_healpixs = {}
        self.freq = 125.195#my correct answers are computed with c=300,so im rescaing freq here to compensate
        self.zenithequ = sv.ctos(np.fromfile(self.test_dir + 'zenith.bin', dtype = 'float32'))[1:]
        self.zenithequ[0] = np.pi/2 - self.zenithequ[0]
        self.zenithequ = np.array(self.zenithequ)[::-1]
        self.rot = np.fromfile(self.test_dir + 'x5rot.bin', dtype = 'float32').reshape((3,3))#fine tune rotation for ant array
        self.correct_result = np.loadtxt(self.test_dir + 'Revised_Location_Visibilties_for_6_m_south_3_m_east_0_m_up_xx_pol_125.195_MHz.dat')
        self.correct_result = self.correct_result[:-1, 1] + 1j * self.correct_result[:-1, 2]


        self.vs = sv.Visibility_Simulator()
        self.vs.initial_zenith = np.array([0,45.2977*np.pi/180])#self.zenithequ

        for f in range(110,200,10):
            beam_healpixs[f] = np.fromfile(self.test_dir + '../data/MWA_beam_in_healpix_horizontal_coor/nside=%i_freq=%i_%s.bin'%((self.blmax + 1)/3, f, self.pol), dtype='float32')
        freqa = int(np.floor(self.freq/10.) * 10)
        freqb = freqa + 10
        beam_healpix = beam_healpixs[freqa] + (self.freq - freqa) * (beam_healpixs[freqb] - beam_healpixs[freqa]) / (freqb - freqa) #linear interpolation
        self.vs.import_beam(beam_healpix)

    def test_josh_gsm(self):
        self.nside = 32
        nside = self.nside
        pca1 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm1.fits' + str(nside))
        pca2 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm2.fits' + str(nside))
        pca3 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm3.fits' + str(nside))
        gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
        equatorial_GSM = np.zeros(12*nside**2,'float')
        #rotate sky map
        for i in range(12*nside**2):
            ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
            pixindex, weight = hpf.get_neighbours(nside,ang[0],ang[1])
            for pix in range(len(pixindex)):
                equatorial_GSM[i] += weight[pix]*gsm[pixindex[pix]]
        self.alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(equatorial_GSM), 3 * nside - 1)
        self.result32 = self.vs.calculate_visibility(sv.expand_real_alm(self.alm), d=self.rot.dot(np.array([6.0,3.0,0.0])), freq=self.freq, nt=len(self.correct_result), L = 3*self.nside-1, verbose = False)

        #self.nside = 64
        #nside = self.nside
        #pca1 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm1.fits' + str(nside))
        #pca2 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm2.fits' + str(nside))
        #pca3 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm3.fits' + str(nside))
        #gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
        #equatorial_GSM = np.zeros(12*nside**2,'float')
        ##rotate sky map
        #for i in range(12*nside**2):
            #ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
            #pixindex, weight = hpf.get_neighbours(nside,ang[0],ang[1])
            #for pix in range(len(pixindex)):
                #equatorial_GSM[i] += weight[pix]*gsm[pixindex[pix]]
        #self.alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(equatorial_GSM), 3 * nside - 1)
        #self.result64 = self.vs.calculate_visibility(sv.expand_real_alm(self.alm), d=self.rot.dot(np.array([6.0,3.0,0.0])), freq=self.freq, nt=len(self.correct_result), L = 3*self.nside-1, verbose = False)
        #plt.plot(np.real(self.result32), 'r--', np.real(self.result64), 'b--', np.real(self.correct_result), 'g--')
        #plt.show()
        #plt.plot(np.imag(self.result32), 'r--', np.imag(self.result64), 'b--', np.imag(self.correct_result), 'g--')
        #plt.show()
        plt.plot(np.real(self.result32), 'r--', np.real(self.correct_result), 'g--')
        plt.show()
        #np.testing.assert_almost_equal(self.result32 , self.correct_result)

class TestSH(unittest.TestCase):
    def test_spherical_harmonics(self):
        #for i in range(0,600):
            #print i,
            #sys.stdout.flush()
            #for mm in range(-i,i+1):
                #sv.spheh(i,mm,1.55146210172, 1.56649120644)
        for l in [153]:
            for mm in range(-l,l+1):
                print mm,
                sys.stdout.flush()
                sv.spheh(l,mm,1.55146210172, 1.56649120644)
            print ""


class TestSpeed(unittest.TestCase):
    def test_speed(self):
        self.test_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
        self.blmequ = sv.read_real_alm(self.test_dir + 'bx125.195.bin')
        self.blmax = 5
        self.freq = 125.195#my correct answers are computed with c=300,so im rescaing freq here to compensate
        self.zenithequ = sv.ctos(np.fromfile(self.test_dir + 'zenith.bin', dtype = 'float32'))[1:]
        self.zenithequ[0] = np.pi/2 - self.zenithequ[0]
        self.zenithequ = np.array(self.zenithequ)[::-1]
        self.correct_result = np.loadtxt(self.test_dir + 'Revised_Location_Visibilties_for_21_m_south_21_m_east_0_m_up_xx_pol_125.195_MHz.dat')
        self.correct_result = self.correct_result[:-1, 1] + 1j * self.correct_result[:-1, 2]
        self.nside = 128

        self.vs = sv.Visibility_Simulator()
        self.vs.initial_zenith = self.zenithequ
        self.vs.Blm = sv.expand_real_alm(self.blmequ)
        self.rot = np.fromfile(self.test_dir + 'x5rot.bin', dtype = 'float32').reshape((3,3))

        nside = self.nside
        print "Reading fits...",
        sys.stdout.flush()
        pca1 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm1.fits' + str(nside))
        pca2 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm2.fits' + str(nside))
        pca3 = hp.fitsfunc.read_map(self.test_dir + '/../data/gsm3.fits' + str(nside))
        gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)
        print "Done reading"
        sys.stdout.flush()
        equatorial_GSM = np.zeros(12*nside**2,'float')

        #rotate sky map
        print "Rotating map...",
        sys.stdout.flush()
        for i in range(12*nside**2):
            ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
            pixindex, weight = hpf.get_neighbours(nside,ang[0],ang[1])
            for pix in range(len(pixindex)):
                equatorial_GSM[i] += weight[pix]*gsm[pixindex[pix]]
        print "Done rotating"
        sys.stdout.flush()

        print "Creating map alm...",
        sys.stdout.flush()
        self.alm = sv.convert_healpy_alm(hp.sphtfunc.map2alm(equatorial_GSM), 3 * nside - 1)
        print "Done alm"
        sys.stdout.flush()

        print "Computing visibilities...",
        sys.stdout.flush()
        timer = time.time()
        self.result = self.vs.calculate_visibility(sv.expand_real_alm(self.alm), d=self.rot.dot(np.array([21.0,21.0,0.0])), freq=self.freq, nt=len(self.correct_result), L = 3*self.nside-1, verbose = True)
        print "done", (time.time() - timer)/60, 'min'
        sys.stdout.flush()
        #print len(self.result), np.argmax(np.real(self.result)) - np.argmax(np.real(self.correct_result)), np.argmax(np.imag(self.result)) - np.argmax(np.imag(self.correct_result))
        plt.plot(np.real(self.result), 'r--', np.real(self.correct_result), 'b--')
        plt.show()
        plt.plot(np.imag(self.result), 'r--', np.imag(self.correct_result), 'b--')
        plt.show()
        #self.assertAlmostEqual(np.mean(abs((self.result-self.correct_result)/self.correct_result))**2, 0, 2)
if __name__ == '__main__':
    unittest.main()
