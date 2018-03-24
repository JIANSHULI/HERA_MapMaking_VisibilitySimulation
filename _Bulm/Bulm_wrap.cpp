#include "include/Bulm_wrap.h"
#define QUOTE(a) # a
#define uint unsigned int
#define CHK_NULL(a) \
    if (a == NULL) { \
        PyErr_Format(PyExc_MemoryError, "Failed to allocate %s", QUOTE(a)); \
        return NULL; }




/*_  __           _       _                       _   _               _
|  \/  | ___   __| |_   _| | ___   _ __ ___   ___| |_| |__   ___   __| |___
| |\/| |/ _ \ / _` | | | | |/ _ \ | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
| |  | | (_) | (_| | |_| | |  __/ | | | | | |  __/ |_| | | | (_) | (_| \__ \
|_|  |_|\___/ \__,_|\__,_|_|\___| |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/ */



PyObject *compute_Bulm(PyObject *self, PyObject *args, PyObject *kwds) {
    int dummy = 0;
    uint L, L1;
    double freq, dx, dy, dz;
    PyArrayObject *Blm; // input arrays
    static char *kwlist[] = {"Blm", "L", "freq", "dx", "dy", "dz", "L1", "dummy"};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,"O!IddddI|i", kwlist,
            &PyArray_Type, &Blm, &L, &freq, &dx, &dy, &dz, &L1, &dummy))
        return NULL;
    // check shape and type of data
    if (PyArray_NDIM(Blm) != 2 || PyArray_TYPE(Blm) != PyArray_CFLOAT) {
        PyErr_Format(PyExc_ValueError, "data must be a 1D array of floats");
        return NULL;
    }
//C
    PyArrayObject *out=NULL; // output arrays
    npy_intp dims[2] = {L+1, 2*L+1}; //
    PyObject *rv;
    out = (PyArrayObject *) PyArray_SimpleNew(2, dims, PyArray_CFLOAT);
    CHK_NULL(out);

    float pi = 3.141592653589793238462643383279502884197169399;
    double k = 2*pi*freq/299.792458;

    double d = sqrt(dz*dz+dy*dy+dx*dx);
    double dth = acos(dz / d);
    double dph = atan2(dy, dx);
    std::complex<double> sphharm;
    //Tabulate an array of complex conjugate of Ylm * jl * sphjn
    std::vector<std::vector<std::vector<double> > > spheharray (L + L1 + 1, std::vector<std::vector<double> > (2*(L + L1) + 1, std::vector<double> (2, 0)));
//C
    float tmp, tmpr, tmpi;
    uint mp, m1p, m2p;
    for (uint i = 0; i < spheharray.size(); i ++){
        //if (i&1){
            //tmpr = - boost::math::sph_bessel(i, k*d);
            //tmpi = 0;
        //} else{
            //tmpr = boost::math::sph_bessel(i, k*d);
            //tmpi = 0;
        //}
        //std::cout << pi << " " << freq << " " << d << " " << i << ": " << tmpr << "+" << tmpi << "j" << std::endl;
        //std::cout << i << std::flush;
        if (i%4 == 0){
            tmpr = boost::math::sph_bessel(i, k*d);
            tmpi = 0;
        } else if (i%4 == 1){
            tmpi = - boost::math::sph_bessel(i, k*d);
            tmpr = 0;
        } else if (i%4 == 2){
            tmpr = -boost::math::sph_bessel(i, k*d);
            tmpi = 0;
        } else{
            tmpi = boost::math::sph_bessel(i, k*d);
            tmpr = 0;
        }
        //std::cout << " " << i << std::flush;
        for (int m = -i; m < (int)(i + 1); m++){
            //std::cout << i << " " << m << " " << dth << " " << dph << std::flush;
            sphharm = boost::math::spherical_harmonic(i, m, dth, dph);
            mp = m + ((int)(m<0)) * spheharray[i].size();
            spheharray[i][mp][0] = tmpr * real(sphharm) - tmpi * (-imag(sphharm));
            spheharray[i][mp][1] = tmpr * (-imag(sphharm)) + tmpi * real(sphharm);
            //std::cout << i << " " << m << ": " << spheharray[i][mp][0]<< "+" << spheharray[i][mp][1] << "j" << std::endl;
        }
        //std::cout << " " << i << std::endl << std::flush;
    }
//C
    //Tabulate an array of square roots
    std::vector<std::vector<std::vector<float> > > sqrtarray (L + 1, std::vector<std::vector<float> > (L1 + 1, std::vector<float> (L + L1 + 1, 0)));
    for (uint i = 0; i < sqrtarray.size(); i ++){
        for (uint j = 0; j < sqrtarray[0].size(); j++){
            for (uint k = 0; k < sqrtarray[0][0].size(); k++){
                sqrtarray[i][j][k] = sqrt((2*i + 1) * (2*j + 1) * (2*k + 1) / (4 * pi));
                //std::cout << i << " " << j << " " << k << " " << sqrtarray[i][j][k] << std::endl;
            }
        }
    }
//C

    //Start loop l,m,l1,m1,l2
    for (uint l = 0; l < L+1; l++){
        for (int m = -l; m < (int)l+1; m++){
            mp = m + ((int)(m<0)) * ((uint)(dims[1]));
//C
            ((float *) PyArray_GETPTR2(out,l,mp))[0] = 0;//C
            ((float *) PyArray_GETPTR2(out,l,mp))[1] = 0;
            for (uint l1 = 0; l1 < L1+1; l1++){
                for (int m1 = -l1; m1 < (int)l1+1; m1++){
                    m1p = m1 + ((int)(m1<0)) * ((uint)(PyArray_DIM(Blm,1)));
                    int m2 = -(-m+m1);
                    uint l2min = std::max(fabs((int)l-(int)l1), fabs(m2));
                    uint diff = std::max((int)(fabs(m2) - fabs((int)l-(int)l1)), 0);
                    std::vector<double> wignerarray0 = WignerSymbols::wigner3j(l, l1, 0, 0, 0);
                    std::vector<double> wignerarray = WignerSymbols::wigner3j(l, l1, m2, -m, m1);
                    float deltar = 0, deltai = 0;
                    //std::cout << "l2min" << l << " " << l1 << " " << m2 << " " << l2min << " " << l+l1+1 << std::endl;
                    for (uint l2 = l2min; l2 < l+l1+1; l2++){
                        tmp = sqrtarray[l][l1][l2] * wignerarray0[diff+l2-l2min] * wignerarray[l2-l2min];
                        m2p = m2 + ((int)(m2<0)) * ((int)(spheharray[l2].size()));
                        deltar += spheharray[l2][m2p][0] * tmp;
                        deltai += spheharray[l2][m2p][1] * tmp;
                        //std::cout << l << " " << m << " " << mp << " " << l1 << " " << m1 << " " << m1p << " " << l2 << " " << tmp << " " << spheharray[l2][m2p][0] << "+" << spheharray[l2][m2p][1] << "j" << " " << ((float *) PyArray_GETPTR2(Blm,l1,m1p))[0] << "+" << ((float *) PyArray_GETPTR2(Blm,l1,m1p))[1] << "j" << std::endl;
                    }

                    ((float *) PyArray_GETPTR2(out,l,mp))[0] += deltar * ((float *) PyArray_GETPTR2(Blm,l1,m1p))[0] - deltai * ((float *) PyArray_GETPTR2(Blm,l1,m1p))[1];
                    ((float *) PyArray_GETPTR2(out,l,mp))[1] += deltar * ((float *) PyArray_GETPTR2(Blm,l1,m1p))[1] + deltai * ((float *) PyArray_GETPTR2(Blm,l1,m1p))[0];
                }
            }
            if (m&1){
                ((float *) PyArray_GETPTR2(out,l,mp))[0] *= -(4 * pi);
                ((float *) PyArray_GETPTR2(out,l,mp))[1] *= -(4 * pi);
            } else{
                ((float *) PyArray_GETPTR2(out,l,mp))[0] *= (4 * pi);
                ((float *) PyArray_GETPTR2(out,l,mp))[1] *= (4 * pi);
            }
        }
    }
    rv = Py_BuildValue("O", PyArray_Return(out));
    Py_DECREF(out);
    return rv;
}

// Module methods
static PyMethodDef Bulm_methods[] = {
    {"compute_Bulm", (PyCFunction)compute_Bulm, METH_VARARGS | METH_KEYWORDS,
        "Compute Bulm, the core step in simulating visibilities."},
    {NULL, NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC init_Bulm(void) {
    PyObject* m;
    m = Py_InitModule3("_Bulm", Bulm_methods,
    "Wrapper for fast C++ code computing Bulm.");

    import_array();
}
