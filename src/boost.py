import _boost_math

def spharm(l, m, theta, phi):
	real, imag = _boost_math.spharm(l, m, theta, phi)
	return real + 1.j*imag

def spbessel(n, x):
	return _boost_math.spbessel(n, x)
