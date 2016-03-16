__author__ = 'omniscope'
import numpy as np

kB = 1.38065e-23
c = 2.99792e8
h = 6.62607e-34
T = 2.725
hoverk = h / kB

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (h * nu)* (nu / c)**2 / (np.exp(hoverk * nu / T) - 1)
    conversion_factor = (B_nu * c / nu / T)**2 / 2 * np.exp(hoverk * nu / T) / kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_CMB, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / c)**2 * kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

