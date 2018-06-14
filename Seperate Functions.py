import numpy as np
import healpy as hp
from healpy import pixelfunc as hpf

def rotatez_matrix(t):
    return np.array([[np.cos(t), -np.sin(t), np.zeros_like(t)], [np.sin(t), np.cos(t), np.zeros_like(t)], [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]])

def rotatey_matrix(t):
    return np.array([[np.cos(t), np.zeros_like(t), np.sin(t)], [np.zeros_like(t), np.ones_like(t), np.zeros_like(t)], [-np.sin(t), np.zeros_like(t), np.cos(t)]])

def rotate_healpixmap(healpixmap, z1, y1, z2):#the three rotation angles are (fixed rotation axes and right hand convention): rotate around z axis by z1, around y axis by y1, and z axis again by z2. I think they form a set of Euler angles, but not exactly sure.
    nside = int((len(healpixmap)/12.)**.5)
    if len(healpixmap)%12 != 0 or 12*(nside**2) != len(healpixmap):
        raise Exception('ERROR: Input healpixmap length %i is not 12*nside**2!'%len(healpixmap))
    rot_matrix = rotatez_matrix(-z1).dot(rotatey_matrix(-y1).dot(rotatez_matrix(-z2)))
    new_coords = hpf.pix2ang(nside, range(12*nside**2))
    old_coords = hpr.rotateDirection(rot_matrix, new_coords)
    newmap = hpf.get_interp_val(healpixmap, old_coords[0], old_coords[1])
    return newmap


def calculate_pointsource_visibility(initial_zenith=None, ra=None, dec=None, d=None, freq=None, beam_healpix_hor=None, beam_heal_equ=None, nt=None, tlist=None, verbose=False):  # initial_zenith is the 2-D(RA and Declination Angle) vector of zenith in radian, when lst=0 ( typically [0, the declination angle of the array] ), d is the list of baselines in horizontal coord, tlist in lst hours, beam in unites of power
    if initial_zenith is None:
        raise Exception('ERROR: need to set self.initial_zenith first, which is at t=0, the position of zenith in equatorial coordinate in ra dec radians.')
    if tlist is None and nt is None:
        raise Exception("ERROR: neither nt nor tlist was specified. Must input what lst you want in sidereal hours")
    if np.array(d).ndim == 1:
        input_ndim = 1
        d = np.array([d])
    elif np.array(d).ndim == 2:
        input_ndim = 2
    else:
        raise TypeError("Input d has incorrect dimension number of %i." % np.array(d).ndim)
    d_equ = d.dot(np.transpose(rotatez_matrix(initial_zenith[0]).dot(rotatey_matrix(np.pi / 2 - initial_zenith[1]))))
    if beam_healpix_hor is None and beam_heal_equ is None:
        raise Exception("ERROR: conversion from alm for beam to beam_healpix not yet supported, so please specify beam_healpix as a keyword directly, in horizontal coord.")
    elif beam_heal_equ is None:
        beam_heal_equ = np.array(rotate_healpixmap(beam_healpix_hor, 0, np.pi / 2 - initial_zenith[1], initial_zenith[0]))
    if tlist is None:
        tlist = np.arange(0., 24., 24. / nt)
    else:
        tlist = np.array(tlist)
    
    angle_list = tlist / 12. * np.pi
    ps_vec = -np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    ik = 2.j * np.pi * freq / 299.792458
    ###for i, phi in zip(range(len(angle_list)), angle_list):
    ####print beam_heal_equ.shape, np.pi/2 - dec, ra - phi
    ###result[i] = hpf.get_interp_val(beam_heal_equ, np.pi/2 - dec, ra - phi) * np.exp(ik*rotatez_matrix(phi).dot(d_equ).dot(ps_vec))
    ###return result
    try:
        try:
            result = hpf.get_interp_val(beam_heal_equ, np.pi / 2 - dec, ra - np.array(angle_list)) * np.exp(ik * np.dot(rotatez_matrix(angle_list).transpose(0, 2, 1), d_equ.transpose()).transpose(2, 1, 0).dot(ps_vec))
            # print('Use Dot.')
            # print (rotatez_matrix(angle_list).shape, d_equ.shape)
        except:
            result = hpf.get_interp_val(beam_heal_equ, np.pi / 2 - dec, ra - np.array(angle_list)) * np.exp(ik * np.einsum('ijt,uj->uti', rotatez_matrix(angle_list), d_equ).dot(ps_vec))
            print('Use Einsum.')
            # print (rotatez_matrix(angle_list).shape, d_equ.shape)
    except:
        print hpf.get_interp_val(beam_heal_equ, np.pi / 2 - dec, ra - np.array(angle_list)).shape
        print rotatez_matrix(angle_list).shape, d_equ.shape
        print np.exp(ik * np.einsum('ijt,uj->uti', rotatez_matrix(angle_list), d_equ).dot(ps_vec)).shape
        sys.stdout.flush()
    if input_ndim == 1:
        return result[0]
    else:
        return result