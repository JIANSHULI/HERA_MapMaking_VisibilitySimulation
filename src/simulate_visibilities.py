import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as sla
import numpy.linalg as la
import scipy.special as ssp
import math as m
import cmath as cm
from wignerpy._wignerpy import wigner3j, wigner3jvec
from boost import spharm, spbessel
from Bulm import compute_Bulm
from random import random
import healpy as hp
import healpy.pixelfunc as hpf
import os, time, sys
#some constant
pi=m.pi
e=m.e

###############################################
#functions for coordinate transformation
############################################
def ctos(cart):
    [x,y,z] = cart
    if [x,y]==[0,0]:
        return [z,0,0]
    return np.array([np.sqrt(x**2+y**2+z**2), np.arctan2(np.sqrt(x**2+y**2),z), np.arctan2(y,x)])

def stoc(spher):
    return spher[0] * np.array([np.cos(spher[2])*np.sin(spher[1]), np.sin(spher[1])*np.sin(spher[2]), np.cos(spher[1])])

def rotatez_matrix(t):
    return np.array([[np.cos(t), -np.sin(t), np.zeros_like(t)], [np.sin(t), np.cos(t), np.zeros_like(t)], [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]])

def rotatez(dirlist,t):
    [theta,phi] = dirlist
    rm = np.array([[np.cos(t),-np.sin(t),np.zeros_like(t)],[np.sin(t),np.cos(t),np.zeros_like(t)],[np.zeros_like(t),np.zeros_like(t),1]])
    return ctos(rm.dot(stoc([1,theta,phi])))[1:3]

def rotatey_matrix(t):
    return np.array([[np.cos(t), np.zeros_like(t), np.sin(t)], [np.zeros_like(t), np.ones_like(t), np.zeros_like(t)], [-np.sin(t), np.zeros_like(t), np.cos(t)]])

def rotatey(dirlist,t):
    [theta,phi] = dirlist
    rm = np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
    return ctos(rm.dot(stoc([1,theta,phi])))[1:3]


def rotationalMatrix(phi,theta,xi):
    m1=np.array([[np.cos(xi),np.sin(xi),0],[-np.sin(xi),np.cos(xi),0],[0,0,1]])
    m2=np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
    m3=np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
    return m1.dot(m2).dot(m3)


#given the Euler angles and [theta,phi], return the [theta',phi'] after the rotation
def rotation(theta,phi,eulerlist):
    return ctos(rotationalMatrix(eulerlist[0],eulerlist[1],eulerlist[2]).dot(stoc([1,theta,phi])))[1:3]

def rotate_healpixmap(healpixmap, z1, y1, z2):#the three rotation angles are (fixed rotation axes and right hand convention): rotate around z axis by z1, around y axis by y1, and z axis again by z2. I think they form a set of Euler angles, but not exactly sure.
    nside = int((len(healpixmap)/12.)**.5)
    if len(healpixmap)%12 != 0 or 12*(nside**2) != len(healpixmap):
        raise Exception('ERROR: Input healpixmap length %i is not 12*nside**2!'%len(healpixmap))
    newmapcoords_in_oldcoords = [rotatez(rotatey(rotatez(hpf.pix2ang(nside, i), -z2), -y1), -z1) for i in range(12*nside**2)]
    newmap = [hpf.get_interp_val(healpixmap, coord[0], coord[1]) for coord in newmapcoords_in_oldcoords]
    return newmap

#Given the 'time' and 'stdtime'(default=2000.0), return the 'transformation matrix' which transforms the coordinates of a vector from (xs,ys,zs) in the coordinate-system built on epoch='stdtime' into (xt,yt,zt) in the coordinate-system built on epoch='time'. This function hitchhicks pyephem so it's not super fast if you need to call it a million times. Precision converting from B2000 to B1950 is tested and appears to be limited by measurement precision. 2 arcsec for CasA and 0.02 arcsec for Crab.
def epoch_transmatrix(time,stdtime=2000.0):
    import ephem as eph
    coorstd=np.zeros((3,3))
    coortime=np.zeros((3,3))

    stars=['Agena','Menkar','Polaris']
    xs={};ys={};zs={}
    xt={};yt={};zt={}

    for starname,i in zip(stars,range(len(stars))):

        star=eph.star(starname)
        star.compute('2000',epoch='%f'%stdtime)
        xs[starname]=np.cos(star.a_dec)*np.cos(star.a_ra)
        ys[starname]=np.cos(star.a_dec)*np.sin(star.a_ra)
        zs[starname]=np.sin(star.a_dec)
        coorstd[i]=[xs[starname],ys[starname],zs[starname]]

        star.compute('2000',epoch='%f'%time)
        xt[starname]=np.cos(star.a_dec)*np.cos(star.a_ra)
        yt[starname]=np.cos(star.a_dec)*np.sin(star.a_ra)
        zt[starname]=np.sin(star.a_dec)
        coortime[i]=[xt[starname],yt[starname],zt[starname]]


    return (coortime.T).dot(la.inv(coorstd.T))

#############################################
#spherical special functions
##############################################
def sphj(l,z):
    #return ssp.sph_jn(l,z)[0][-1]
    #if ssp.sph_jn(l,z)[0][-1]-spbessel(l,z)!=0:
        #print l, "%.30e, %.30e, %.30e"%(z, ssp.sph_jn(l,z)[0][-1], spbessel(l,z))
    return spbessel(l,z)

def spheh(l,m,theta,phi):
    #return ssp.sph_harm(m,l,phi,theta)
    return spharm(l,m,theta,phi)
############################################
#other functions
################################################
##########################################
#get the appropriate nside that has the desired accuracy for a healpix map
##########################################
def check_beam(data, precision = None, verbose = False):
    nside = int((len(data)/12)**0.5)
    print "Input data nside =", nside
    nsidelist = []
    n = 1
    while n <= nside:
        nsidelist.append(n)
        n = 2*n

    truncatemaps = {}
    for n in nsidelist:
        beam_alm = hp.sphtfunc.map2alm(data,lmax = 3*n-1,iter=50)
        truncatemaps[n] = hp.sphtfunc.alm2map(beam_alm,nside,verbose=False)

    error_types = ["RMS diff/RMS data", "RMS diff/max data", "max diff/max data"]
    errorlist = {}
    for n in nsidelist:
        diff = truncatemaps[n] - data
        errorlist[n] = [la.norm(diff)/la.norm(data),la.norm(diff)/(12*nside**2)**0.5/max(data),max(abs(diff))/max(data)]
        if verbose:
            msg =  "nside = %i: "%(n)
            for i in range(len(error_types)):
                msg += (error_types[i] + ": ")
                msg += "%.5f" %errorlist[n][i]
                msg += "; "
            print msg

    if type(precision) == float:
        for n in nsidelist:
            if errorlist[n][0] < precision and errorlist[n][1] < precision:# and errorlist[n][2] < precision:
                if verbose:
                    print 'nside = %.d has the desired precision' %n
                return n
        print 'Need larger nside than the input to have the desired precision'
    return errorlist


#claculate alm from a skymap (each element has the form [theta,phi,intensity])
nside=30                           #increase this for higher accuracy
def get_alm(skymap,lmax=4,dtheta=pi/nside,dphi=2*pi/nside):
    alm={}
    for l in range(lmax+1):
        for mm in range(-l,l+1):
            alm[(l,mm)]=0
            for p in skymap:
                alm[(l,mm)] += np.conj(spheh(l,mm,p[0],p[1]))*p[2]*dtheta*dphi*np.sin(p[0])
    return alm

class InverseCholeskyMatrix:#for a positive definite matrix, Cholesky decomposition is M = L.Lt, where L lower triangular. This decomposition helps computing inv(M).v faster, by avoiding calculating inv(M). Once we have L, the product is simply inv(Lt).inv(L).v, and inverse of triangular matrices multiplying a vector is fast. sla.solve_triangular(M, v) = inv(M).v
    def __init__(self, matrix):
        if type(matrix).__module__ != np.__name__ or len(matrix.shape) != 2:
            raise TypeError("matrix must be a 2D numpy array");
        try:
            self.L = la.cholesky(matrix)#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
            self.Lt = self.L.conjugate().transpose()
            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
        except:
            raise TypeError("cholesky failed. matrix is not positive definite.")

    @classmethod
    def fromfile(cls, filename, n, dtype):
        if not os.path.isfile(filename):
            raise IOError("%s file not found!"%filename)
        matrix = cls(np.array([[1,0],[0,1]]))
        try:
            matrix.L = np.fromfile(filename, dtype=dtype).reshape((n,n))#L.dot(L.conjugate().transpose()) = matrix, L lower triangular
            matrix.Lt = matrix.L.conjugate().transpose()
            #print la.norm(self.L.dot(self.Lt)-matrix)/la.norm(matrix)
        except:
            raise TypeError("cholesky import failed. matrix is not %i by %i with dtype=%s."%(n, n, dtype))
        return matrix

    def dotv(self, vector):
        return sla.solve_triangular(self.Lt, sla.solve_triangular(self.L, vector, lower=True), lower=False)

    def dotM(self, matrix):
        return np.array([self.dotv(v) for v in matrix.transpose()]).transpose()

    def astype(self, t):
        self.L = self.L.astype(t)
        self.Lt = self.Lt.astype(t)
        return self

    def tofile(self, filename, overwrite = False):
        if os.path.isfile(filename) and not overwrite:
            raise IOError("%s file exists!"%filename)
        self.L.tofile(filename)

class Visibility_Simulator:
    def __init__(self):
        self.Blm = np.zeros([3,3],'complex')
        self.initial_zenith = np.array([1000, 1000])     #at t=0, the position of zenith in equatorial coordinate in ra dec radians

    def import_beam(self, beam_healpix_hor):#import beam in horizontal coord in a healpix list and rotate it according to initial_zenith
        if np.array(self.initial_zenith).tolist() == [1000, 1000]:
            raise Exception('ERROR: need to set self.initial_zenith first, which is at t=0, the position of zenith in equatorial coordinate in ra dec radians.')
        beamequ_heal = rotate_healpixmap(beam_healpix_hor, 0, np.pi/2 - self.initial_zenith[1], self.initial_zenith[0])
        self.Blm = expand_real_alm(convert_healpy_alm(hp.sphtfunc.map2alm(beamequ_heal), int(3 * (len(beamequ_heal)/12)**.5 - 1)))

    def calculate_Bulm(self, L, freq, d, L1, verbose = False):
        Blm = np.zeros((L1+1, 2*L1+1), dtype='complex64')
        try:
            _ = self.Blm[(L1, L1)]
        except:
            raise Exception("Error: Existing alm for the beam cannot handle the requested L1 = %i! Make sure you have imported beam."%L1)
        for lm in self.Blm:
            Blm[lm] = self.Blm[lm]
        Bulmarray = compute_Bulm(Blm, L, freq, d, L1)
        Bulmdic = {}
        for l in range(L+1):
            for m in range(-l,l+1):
                Bulmdic[(l,m)] = Bulmarray[(l,m)]
        return Bulmdic

    #from Bulm, return Bulm with given frequency(wave vector k) and baseline vector
    def calculate_Bulm_old(self, L, freq, d, L1, verbose = False):    #L= lmax  , L1=l1max, takes d in equatorial coord
        k = 2*pi*freq/299.792458
        timer = time.time()

        #an array of the comples conjugate of Ylm's * j^l * sphjn
        dth = ctos(d)[1]
        dph = ctos(d)[2]
        if verbose:
            print "Tabulizing spherical harmonics...", dth, dph
            sys.stdout.flush()
        spheharray = np.zeros([L+L1+1,2*(L+L1)+1],'complex64')
        for i in range(0,L+L1+1):
            tmp = spbessel(i, -k*la.norm(d)) * (1.j)**i
            #print pi, freq, la.norm(d), i, spbessel(i, -k*la.norm(d))
            for mm in range(-i,i+1):
                spheharray[i, mm]=(spharm(i,mm,dth,dph)).conjugate() * tmp
                #print spheharray[i, mm]

        if verbose:
            print "Done", float(time.time() - timer)/60
            sys.stdout.flush()

        #an array of m.sqrt((2*l+1)*(2*l1+1)*(2*l2+1)/(4*pi))
        sqrtarray = np.zeros([L+1,L1+1,L+L1+1],'float32')
        for i in range(L+1):
            for j in range(L1+1):
                for kk in range(0,L+L1+1):
                    sqrtarray[i, j, kk] = m.sqrt((2*i+1)*(2*j+1)*(2*kk+1)/(4*pi))
                    #print i,j,kk,sqrtarray[i, j, kk]
        if verbose:
            print float(time.time() - timer)/60
            sys.stdout.flush()

        #Sum over to calculate Bulm
        Bulm={}
        for l in range(L+1):
            for mm in range(-l,l+1):
                Bulm[(l,mm)]=0
                for l1 in range(L1+1):
                    for mm1 in range(-l1,l1+1):
                        mm2=-(-mm+mm1)
                        l2min = max([abs(l-l1),abs(mm2)])
                        diff = max(abs(mm2)-abs(l-l1),0)

                        wignerarray0 = wigner3jvec(l,l1,0,0)
                        wignerarray = wigner3jvec(l,l1,-mm,mm1)

                        delta = 0
                        for l2 in range(l2min,l+l1+1):
                            delta += spheharray[l2, mm2]*sqrtarray[l, l1, l2]*wignerarray0[diff+l2-l2min]*wignerarray[l2-l2min]#(1j**l2)*sphjarray[l2]*
                            #print l,mm,l1,mm1,l2,sqrtarray[l, l1, l2]*wignerarray0[diff+l2-l2min]*wignerarray[l2-l2min],spheharray[l2, mm2], delta, self.Blm[l1,mm1]
                        #print delta, self.Blm[(l1,mm1)], delta * self.Blm[l1,mm1], Bulm[(l,mm)],
                        Bulm[(l,mm)] += delta * self.Blm[l1,mm1]
                        #print Bulm[(l,mm)]
                #print l, mm, Bulm[(l,mm)]
                Bulm[(l,mm)] = 4*pi*(-1)**mm * Bulm[(l,mm)]
                #print l, mm, Bulm[(l,mm)]
        if verbose:
            print float(time.time() - timer)/60
            sys.stdout.flush()
        return Bulm

    def calculate_pointsource_visibility(self, ra, dec, d, freq, beam_healpix_hor = None, beam_heal_equ = None, nt = None, tlist = None, verbose = False):#d in horizontal coord, tlist in lst hours
        if self.initial_zenith.tolist() == [1000, 1000]:
            raise Exception('ERROR: need to set self.initial_zenith first, which is at t=0, the position of zenith in equatorial coordinate in ra dec radians.')
        if tlist is None and nt is None:
                raise Exception("ERROR: neither nt nor tlist was specified. Must input what lst you want in sidereal hours")
        d_equ = stoc(np.append(la.norm(d),rotatez(rotatey(ctos(d)[1:3], (np.pi/2 - self.initial_zenith[1])), self.initial_zenith[0])))
        if beam_healpix_hor is None and beam_heal_equ is None:
            raise Exception("ERROR: conversion from alm for beam to beam_healpix not yet supported, so please specify beam_healpix as a keyword directly, in horizontal coord.")
        elif beam_heal_equ is None:
            beam_heal_equ = np.array(rotate_healpixmap(beam_healpix_hor, 0, np.pi/2 - self.initial_zenith[1], self.initial_zenith[0]))
        if tlist is None:
            tlist = np.arange(0.,24.,24./nt)
        else:
            tlist = np.array(tlist)

        angle_list = tlist/12.*np.pi
        result = np.empty(len(angle_list), dtype='complex64')
        ps_vec = -np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        ik = 2.j*np.pi*freq/299.792458
        ###for i, phi in zip(range(len(angle_list)), angle_list):
            ####print beam_heal_equ.shape, np.pi/2 - dec, ra - phi
            ###result[i] = hpf.get_interp_val(beam_heal_equ, np.pi/2 - dec, ra - phi) * np.exp(ik*rotatez_matrix(phi).dot(d_equ).dot(ps_vec))
        ###return result
        return hpf.get_interp_val(beam_heal_equ, np.pi/2 - dec, ra - np.array(angle_list)) * np.exp(ik * (rotatez_matrix(angle_list).transpose(2,0,1).dot(d_equ).dot(ps_vec)))

    def calculate_pol_pointsource_visibility(self, ra, dec, d_in, freq, beam_healpix_hor = None, beam_heal_equ = None, nt = None, tlist = None, verbose = False):#d_in in horizontal coord, beam is 4 by npix (xx,xy,yx,yy), tlist in lst hours, return 4 by 4 by nt numbers, where first dim of 4 is received xx xy yx yy, and second is xx xy yx yy on sky
        if self.initial_zenith.tolist() == [1000, 1000]:
            raise Exception('ERROR: need to set self.initial_zenith first, which is at t=0, the position of zenith in equatorial coordinate in ra dec radians.')
        if tlist is None and nt is None:
                raise Exception("ERROR: neither nt nor tlist was specified. Must input what lst you want in sidereal hours")
        if np.array(d_in).ndim == 1:
            d_in = [d_in]
        d_equ = np.array([stoc(np.append(la.norm(d),rotatez(rotatey(ctos(d)[1:3], (np.pi/2 - self.initial_zenith[1])), self.initial_zenith[0]))) for d in d_in])
        if beam_healpix_hor is None and beam_heal_equ is None:
            raise Exception("ERROR: conversion from alm for beam to beam_healpix not yet supported, so please specify beam_healpix_hor as a keyword directly, in horizontal coord.")
        elif beam_heal_equ is None:
            beam_heal_equ = np.array([rotate_healpixmap(ibeam_healpix_hor, 0, np.pi/2 - self.initial_zenith[1], self.initial_zenith[0]) for ibeam_healpix_hor in beam_healpix_hor])
        if tlist is None:
            tlist = np.arange(0.,24.,24./nt)
        else:
            tlist = np.array(tlist)

        angle_list = tlist/12.*np.pi

        ps_vec = -np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])#pointing towards observer
        ik = 2.j*np.pi*freq/299.792458


        beamt = np.array([hpf.get_interp_val(ibeam_heal_equ, np.pi/2 - dec, ra - np.array(angle_list)) for ibeam_heal_equ in beam_heal_equ])
        beamt.shape = (2, 2, len(tlist))

        Rut = rotatez_matrix(angle_list).transpose(2,0,1)#rotation matrix for ubl over t
        fringe = np.exp(ik * (Rut.dot(d_equ.transpose()).transpose(2,0,1).dot(ps_vec)))#u by t

        local_zenith_vect = Rut.dot([np.cos(self.initial_zenith[1]) * np.cos(self.initial_zenith[0]), np.cos(self.initial_zenith[1]) * np.sin(self.initial_zenith[0]), np.sin(self.initial_zenith[1])])#over t the zenith in local coord expressed in equitorial coord

        if np.abs(ps_vec[-1]) == 1:
            phi0 = np.array([0, 1, 0])
            alpha0 = np.array([-1, 0, 0])
        else:
            phi0 = np.cross([0,0,1], -ps_vec)
            phi0 = phi0/la.norm(phi0)
            alpha0 = np.cross([0,0,1], phi0)
            alpha0 = alpha0/la.norm(alpha0)
        phi1t = np.cross(local_zenith_vect, -ps_vec)
        if np.min(la.norm(local_zenith_vect-(-ps_vec), axis = -1)) == 0.:
            if np.cross([0,0,1], -ps_vec) != 0:
                phi1t[np.argmin(la.norm(local_zenith_vect-(-ps_vec), axis = -1))] = np.cross([0,0,1], -ps_vec)
            else:
                phi1t[np.argmin(la.norm(local_zenith_vect-(-ps_vec), axis = -1))] = np.array([0, 1, 0])
        phi1t = phi1t / (la.norm(phi1t, axis=-1)[:,None])

        Ranglet = np.arctan2(phi1t.dot(alpha0), phi1t.dot(phi0))#rotation angle for polarization coord over t, from equatotial(phi0,alpha0) to local(phi1,alpha1), around vector -ps_vec
        Ranglet = rotatez_matrix(Ranglet)[:2,:2]#rotation matrix around -ps_vec for polarization coord over t, 3 by 3 by t

        BRt = np.array([beamt[..., i].dot(Ranglet[...,i]) for i in range(len(tlist))])

        result = np.empty((len(d_in), 4 * len(tlist), 4), dtype='complex64')#time is fastest changing in 4 by t
        for truen, (truei, truej) in enumerate([[0,0],[0,1],[1,0],[1,1]]):
            for measuren, (measurei, measurej) in enumerate([[0,0],[0,1],[1,0],[1,1]]):
                result[:, measuren*len(tlist):(measuren+1)*len(tlist), truen] = (np.conjugate(BRt[:, measurei,truei])*BRt[:, measurej,truej])[None,:]*fringe
        return result


    def calculate_visibility(self, skymap_alm, d, freq, L, nt = None, tlist = None, verbose = False):#d in horizontal coord, tlist in [0,24) lst hours
        if self.initial_zenith.tolist() == [1000, 1000]:
            raise Exception('ERROR: need to set self.initial_zenith first, which is at t=0, the position of zenith in equatorial coordinate in ra dec radians.')
        ##rotate d to equatorial coordinate
        drotate = stoc(np.append(la.norm(d),rotatez(rotatey(ctos(d)[1:3], (np.pi/2 - self.initial_zenith[1])), self.initial_zenith[0])))
        if verbose:
            print "Rotated baseline:", drotate
            sys.stdout.flush()

        #calculate Bulm
        L1 = max([key for key in self.Blm])[0]
        if verbose:
            timer = time.time()
            print "Starting Bulm calculation...",
            sys.stdout.flush()
        Bulm = self.calculate_Bulm(L, freq, drotate ,L1, verbose = verbose)
        if verbose:
            print "done in %.2f minutes."%(float(time.time() - timer)/60.)
            sys.stdout.flush()


        #get the intersect of the component of skymap_alm and self.Blm
        commoncomp=list(set([key for key in skymap_alm]) & set([key for key in Bulm]))
        #if verbose:
            #print len(commoncomp)
            #sys.stdout.flush()

        #calculate visibilities
        if tlist is not None:
            vlist = np.zeros(len(tlist),'complex128')
            for i in range(len(tlist)):
                phi=2*pi/24.0*tlist[i]            #turn t (time in hour) to angle of rotation
                v=0
                for comp in commoncomp:
                    v += np.conjugate(skymap_alm[comp]) * Bulm[comp] * e**(-1.0j*comp[1]*phi)
                vlist[i]=v
        else:
            lcommon = max(np.array(commoncomp)[:,0])
            if nt is None:
                nfourier = 2 * lcommon + 1
            elif nt < 2 * lcommon + 1: #make sure nfourier is a multiple of nt and bigger than 2 * lcommon + 1
                nfourier = 0
                while nfourier < 2 * lcommon + 1:
                    nfourier = nfourier +  nt
            else:
                nfourier = nt
            if verbose:
                timer = time.time()
                print "Starting cm calculation...",
                sys.stdout.flush()
            self.cm = np.zeros(nfourier, dtype = 'complex128')
            for mm in range(-lcommon, lcommon + 1):
                for l in range(abs(mm), lcommon + 1):
                    self.cm[mm] += np.conjugate(skymap_alm[(l, mm)]) * Bulm[(l, mm)]
            if verbose:
                print "done in %.2f minutes."%(float(time.time() - timer)/60.)
                sys.stdout.flush()
            vlist = np.fft.fft(self.cm)
            if nt < nfourier:
                vlist = vlist[::(nfourier/nt)]
        return vlist


#################################
####IO functions for alms########
#################################
def read_alm(filename):
    if not os.path.isfile(filename):
        raise Exception("File %s does not exist."%filename)
    raw = np.fromfile(filename, dtype = 'complex64')
    lmax = int(len(raw)**0.5) - 1
    if lmax != len(raw)**0.5 - 1:
        raise Exception("Invalid array length of %i found in file %s. Array must be of length (l+1)^2."%(len(raw), filename))
    result = {}
    cnter = 0
    for l in range(lmax + 1):
        for mm in range(-l, l + 1):
            result[(l, mm)] = raw[cnter]
            cnter = cnter + 1
    return result


def read_real_alm(filename):
    if not os.path.isfile(filename):
        raise Exception("File %s does not exist."%filename)
    raw = np.fromfile(filename, dtype = 'complex64')
    lmax = int(m.floor((2*len(raw))**0.5)) - 1
    if (lmax + 1) * (lmax + 2) / 2 != len(raw):
        raise Exception("Invalid array length of %i found in file %s. Array must be of length (l+1)(l+2)/2."%(len(raw), filename))
    result = {}
    cnter = 0
    for l in range(lmax + 1):
        for mm in range(0, l + 1):
            result[(l, mm)] = raw[cnter]
            cnter = cnter + 1
    return result

def convert_healpy_alm(healpyalm, lmax):
    if len(healpyalm) != (lmax + 1) * (lmax + 2) / 2:
        raise Exception('Length of input 1D healpy alm (%i) does not match the lmax inputed (%i). Length should be (l+1)(l+2)/2 for a real map alm.'%(len(healpyalm), lmax))
    result = {}
    cnter = 0
    for mm in range(lmax + 1):
        for l in range(mm, lmax + 1):
            result[(l, mm)] = healpyalm[cnter]
            cnter = cnter + 1
    return result

def expand_real_alm(real_alm):
    lmax = np.max(np.array(real_alm.keys()))
    if len(real_alm) != (lmax+1)*(lmax+2)/2:
        raise Exception('Input real_alm does not look like a real_alm. Max l %i does not agree with length %i of input real_alm.'%(lmax, len(real_alm)))
    result = {}
    for l in range(lmax + 1):
        for mm in range(0, l + 1):
            result[(l, mm)] = real_alm[(l,mm)]
    for l in range(lmax + 1):
        for mm in range(-l, 0):
            result[(l, mm)] = (-1)**mm * np.conjugate(real_alm[(l, -mm)])

    return result
#add in a line to test github
#add another line

#############################
##test the class
#############################
if __name__ == '__main__':
    btest=Visibility_Simulator()
    btest.initial_zenith=np.array([45.336111/180.0*pi,0])
    #Import the healpix map of the beam, then calculate the Blm of the beam
    with open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/beamhealpix/beamhealpixmap.txt') as f:
        data = np.array([np.array([float(line)]) for line in f])

    data = data.flatten()
    beam_alm = hp.sphtfunc.map2alm(data,iter=10)

    Blm={}
    for l in range(21):
        for mm in range(-l,l+1):
            if mm >= 0:
                Blm[(l,mm)] = (1.0j)**mm*beam_alm[hp.sphtfunc.Alm.getidx(10,l,abs(mm))]
            if mm < 0:
                Blm[(l,mm)] = np.conj((1.0j)**mm*beam_alm[hp.sphtfunc.Alm.getidx(10,l,abs(mm))])

    btest.Blm=Blm


                           #_
      #__ _ ____ __    __ _| |_ __
     #/ _` (_-< '  \  / _` | | '  \
     #\__, /__/_|_|_| \__,_|_|_|_|_|
     #|___/
    #create sky map alm
    pca1 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm1.fits32')
    pca2 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm2.fits32')
    pca3 = hp.fitsfunc.read_map('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/GSM_32/gsm3.fits32')
    gsm = 422.952*(0.307706*pca1+-0.281772*pca2+0.0123976*pca3)

    nside=32
    equatorial_GSM = np.zeros(12*nside**2,'float')
    #rotate sky map
    for i in range(12*nside**2):
        ang = hp.rotator.Rotator(coord='cg')(hpf.pix2ang(nside,i))
        pixindex, weight = hpf.get_neighbours(nside,ang[0],ang[1])
        for pix in range(len(pixindex)):
            equatorial_GSM[i] += weight[pix]*gsm[pixindex[pix]]

    almlist = hp.sphtfunc.map2alm(equatorial_GSM,iter=10)
    alm={}
    for l in range(96):
        for mm in range(-l,l+1):
            if mm >= 0:
                alm[(l,mm)] = (1.0j)**mm*almlist[hp.sphtfunc.Alm.getidx(nside*3-1,l,abs(mm))]
            if mm < 0:
                alm[(l,mm)] = np.conj((1.0j)**mm*almlist[hp.sphtfunc.Alm.getidx(nside*3-1,l,abs(mm))])


    #set frequency and baseline vector
    freq = 125.195
    d=np.array([-6.0,-3.0,0.0])

    timelist = 1/10.0*np.arange(24*10+1)
    v2 = btest.calculate_visibility(alm, d, freq, timelist)
    print v2

    savelist = np.zeros([len(timelist),3],'float')
    for i in range(len(timelist)):
        savelist[i][0] = timelist[i]
        savelist[i][1] = v2[i].real
        savelist[i][2] = v2[i].imag


    f_handle = open('/home/eric/Dropbox/MIT/UROP/simulate_visibilities/visibility_result/sphericalharmonics_L20.txt','w')
    for i in savelist:
        np.savetxt(f_handle, [i])
    f_handle.close()


    ##############################
    ###check the result is the same as mathematica
    ##############################
    #btest=Visibility_Simulator()
    #btest.initial_zenith=np.array([pi/2,0])
    ##Import the healpix map of the beam, then calculate the Blm of the beam
    #Blm={}
    #for l in range(21):
        #for mm in range(-l,l+1):
            #Blm[(l,mm)] = 0

    #Blm[(1, -1)] = Blm[(1, 1)] = -0.36;
    #Blm[(2, 1)] = Blm[(2, -1)] = -0.46;
    #Blm[(3, 1)] = Blm[(3, -1)] = -0.30;

    #btest.Blm=Blm

                           ##_
      ##__ _ ____ __    __ _| |_ __
     ##/ _` (_-< '  \  / _` | | '  \
     ##\__, /__/_|_|_| \__,_|_|_|_|_|
     ##|___/
    ##create sky map alm

    #alm={}
    #alm[(1,1)] = 1
    #alm[(2,1)] = 0.5
    #alm[(3,2)] = 0.1


    ##set frequency and baseline vector
    #freq = 3*299.792458/(2*pi)
    #d=np.array([3.0,6.0,0.0])

    #timelist = [0]
    #v2 = btest.calculate_visibility(alm, d, freq, timelist)
    #print v2
