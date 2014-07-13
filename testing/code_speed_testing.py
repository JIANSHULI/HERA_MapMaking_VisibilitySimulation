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
#from sympy.physics.wigner import wigner_3j
from wignerpy._wignerpy import wigner3j, wigner3jvec
import time
#from  symbol_3j import wigner_3j


#a = wigner3jvec(10,10,1,0)
#print a






t1=time.clock()
for i in range(10000):
	a = wigner3jvec(50,50,0,0,)
	
t2=time.clock()
for i in range(10000):
	a = wigner3jvec(200,200,0,0)

t3=time.clock()


print [t2-t1,t3-t2]
