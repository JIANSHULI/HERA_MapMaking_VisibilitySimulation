import _boost_math
def spharm(l, m, theta, phi):
	real, imag = _boost_math.spharm(l, m, theta, phi)
	return real + 1.j*imag
