from __future__ import division
import numpy as np
import scipy as sp
import scipy.misc as mi
import scipy.sparse as sps
import scipy.linalg as la
import scipy.special as ssp
import math as m
import cmath as cm
from random import random
from wignerpy._wignerpy import wigner3j, wigner3jvec
import time


#some constant
pi=m.pi
e=m.e

###############################################
#functions for coordinate transformation
############################################
def ctos(cart):
	[x,y,z] = cart
	if [x,y]==[0,0]:
		return [1,pi/2,0]
	return np.array([m.sqrt(x**2+y**2+z**2),m.atan2(m.sqrt(x**2+y**2),z),m.atan2(y,x)])

def stoc(spher):
	[r,theta,phi]=spher
	return np.array([r*m.cos(phi)*m.sin(theta),r*m.sin(theta)*m.sin(phi),r*m.cos(theta)])

def rotationalMatrix(phi,theta,xi):
	m1=np.array([[m.cos(xi),m.sin(xi),0],[-m.sin(xi),m.cos(xi),0],[0,0,1]])
	m2=np.array([[1,0,0],[0,m.cos(theta),m.sin(theta)],[0,-m.sin(theta),m.cos(theta)]])
	m3=np.array([[m.cos(phi),m.sin(phi),0],[-m.sin(phi),m.cos(phi),0],[0,0,1]])
	return m1.dot(m2).dot(m3)
	

#given the Euler angles and [theta,phi], return the [theta',phi'] after the rotation
def rotation(theta,phi,eulerlist):
	return ctos(rotationalMatrix(eulerlist[0],eulerlist[1],eulerlist[2]).dot(stoc([1,theta,phi])))[1:3]


#############################################
#spherical special functions
##############################################
def sphj(l,z):
	return ssp.sph_jn(l,z)[0][-1]

def spheh(l,m,theta,phi):
	return ssp.sph_harm(m,l,phi,theta)
	
	
############################################
#other functions
################################################
#claculate alm from a skymap (each element has the form [theta,phi,intensity])
nside=30                           #increase this for higher accuracy
def get_alm(skymap,lmax=4,dtheta=pi/nside,dphi=2*pi/nside):
	alm={}
	for l in range(lmax+1):
		for mm in range(-l,l+1):
			alm[(l,mm)]=0
			for p in skymap:
				alm[(l,mm)] += np.conj(spheh(l,mm,p[0],p[1]))*p[2]*dtheta*dphi*m.sin(p[0])
	return alm
	







######################################################################################            _       _             _    __                  _   _             
           #(_)     (_)           | |  / _|                | | (_)            
  #___  _ __ _  __ _ _ _ __   __ _| | | |_ _   _ _ __   ___| |_ _  ___  _ __  
 #/ _ \| '__| |/ _` | | '_ \ / _` | | |  _| | | | '_ \ / __| __| |/ _ \| '_ \ 
#| (_) | |  | | (_| | | | | | (_| | | | | | |_| | | | | (__| |_| | (_) | | | |
 #\___/|_|  |_|\__, |_|_| |_|\__,_|_| |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|
               #__/ |                                                         
              #|___/                                                          
#from Bulm, return Bulm with given frequency(wave vector k) and baseline vector
def get_Bulm(Blm,freq=137,d=np.array([3.0,6.0,0.0])):
	k = 2*pi*freq/299.792458
	lmax = 4
	Bulm={}
	for l in range(lmax+1):
		for mm in range(-l,l+1):
			Bulm[(l,mm)]=0
			for l1 in range(lmax+1):
				for mm1 in range(-l1,l1+1):
					for l2 in range(abs(l-l1),l+l1+1):
						mm2=-(mm+mm1)
						if abs(mm2)<=l2:
							Bulm[(l,mm)] += 4*pi*(1j**l2)*sphj(l2,k*la.norm(d))*np.conj(spheh(l2,mm2,ctos(d)[1],ctos(d)[2]))*Blm[(l1,mm1)]*m.sqrt((2*l+1)*(2*l1+1)*(2*l2+1)/(4*pi))*wigner3j(l,l1,l2,0,0,0)*wigner3j(l,l1,l2,mm,mm1,mm2)
	return Bulm		
	

##create a beam function from spherical harmonics (these numbers are roughly what I get from using Victor's beam at freq 137MHz)
#Blm = {}
#for i in range(21):
	#for j in range(-i,i+1):
		#Blm[(i,j)] = random()

Blm = np.zeros([21,21*2+1],'complex')
for i in range(21):
	for j in range(-i,i+1):
		Blm[i][i+j] = random()
#Blm[(1,-1)] = -0.36
#Blm[(1,1)] = -0.36
#Blm[(2,1)] = -0.46
#Blm[(2,-1)] = -0.46
#Blm[(3,1)] = -0.30
#Blm[(3,-1)] = -0.30



#bulm=get_Bulm(Blm)

                                 
#(_)                                   
 #_ _ __ ___  _ __  _ __ _____   _____ 
#| | '_ ` _ \| '_ \| '__/ _ \ \ / / _ \
#| | | | | | | |_) | | | (_) \ V /  __/
#|_|_| |_| |_| .__/|_|  \___/ \_/ \___|
            #| |                       
            #|_|                       
#######################################
#pre calculate sphej and spheh
########################################

t0=time.clock()

lmax=10
freq=137
k = 2*pi*freq/299.792458
d=np.array([3.0,6.0,0.0])

L=50
L1=20

#an array of the comples conjugate of Ylm's
spheharray = np.zeros([L+L1+1,L+L1+1],'complex')
for i in range(0,L+L1+1):
	for mm in range(-i,i+1):
		spheharray[i][mm]=(spheh(i,mm,ctos(d)[1],ctos(d)[2])).conjugate()
		
		
print 'spheharray'
print time.clock()-t0

#an dictionary of spherical Bessel functions
sphjdict={}
for l in range(2*lmax+1):
	sphjdict[l] = sphj(l,k*la.norm(d))
	
sphjarray = np.zeros(L+L1+1,'complex')
for l in range(L+L1+1):
	sphjarray[l] = sphj(l,k*la.norm(d))
	

print 'sphjdict'
print time.clock()-t0

#an array of m.sqrt((2*l+1)*(2*l1+1)*(2*l2+1)/(4*pi))
sqrtarray = np.zeros([L+1,L1+1,L+L1+1],'complex')
for i in range(L+1):
	for j in range(L1+1):
		for kk in range(0,L+L1+1):
			sqrtarray[i][j][kk] = m.sqrt((2*i+1)*(2*j+1)*(2*kk+1)/(4*pi))

print 'sqrtarray'
print time.clock()-t0


t1=time.clock()
Bulm={}
for l in range(L+1):
	for mm in range(-l,l+1):
		Bulm[(l,mm)]=0
		for l1 in range(L1+1):
			for mm1 in range(-l1,l1+1):
				mm2=-(mm+mm1)
				wignerarray0 = wigner3jvec(l,l1,0,0)
				wignerarray = wigner3jvec(l,l1,mm,mm1)
				l2min = max([abs(l-l1),abs(mm2)])
				diff = max(abs(mm2)-abs(l-l1),0)
				for l2 in range(l2min,l+l1+1):
					#Bulm[(l,mm)] += 4*pi*(1j**(l2%4))*sphjarray[l2]*spheharray[l2][mm2]*Blm[(l1,mm1)]*sqrtarray[l][l1][l2]*wigner3j(l,l1,l2,0,0,0)*wigner3j(l,l1,l2,mm,mm1,mm2)
					#Bulm[(l,mm)] += 4*pi*(1j**(l2%4))*sphjarray[l2]*spheharray[l2][mm2]*Blm[(l1,mm1)]*sqrtarray[l][l1][l2]*wignerarray0[diff+l2-l2min]*wignerarray[l2-l2min]
					Bulm[(l,mm)] += 4*pi*(1j**(l2%4))*sphjarray[l2]*spheharray[l2][mm2]*Blm[l1][l1+mm1]*sqrtarray[l][l1][l2]*wignerarray0[diff+l2-l2min]*wignerarray[l2-l2min]


print time.clock() - t1
print time.clock() - t0










#Bulm={}
#for l in range(lmax+1):
	#for mm in range(-l,l+1):
		#Bulm[(l,mm)]=0
		#for l1 in range(lmax+1):
			#for mm1 in range(-l1,l1+1):
				#for l2 in range(abs(l-l1),l+l1+1):
					#mm2=-(mm+mm1)
					#if Blm[(l1,mm1)] ==0 :
						#Bulm[(l,mm)] +=0
					#elif abs(mm2)<=l2:
						#Bulm[(l,mm)] += 4*pi*(1j**l2)*sphj(l2,k*la.norm(d))*np.conj(spheh(l2,mm2,ctos(d)[1],ctos(d)[2]))*Blm[(l1,mm1)]*m.sqrt((2*l+1)*(2*l1+1)*(2*l2+1)/(4*pi))*wigner3j(l,l1,l2,0,0,0)*wigner3j(l,l1,l2,mm,mm1,mm2)



#print Bulm[(0,0)]-Bulmnew[(0,0)]

#t1=time.clock()
#for i in range(100):
	#a = wigner3j(100,100,100,20,-20,0)
#t2=time.clock()
#for i in range(100):
	#float(wigner_3j(500,500,500,20,-20,0))
#t3=time.clock()

#print [t2-t1,t3-t2]

	
	
	
