#include "include/boost_math_wrap.h"
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
PyObject *spharm_wrap(PyObject *self, PyObject *args){
    uint l;
    int m;
    double theta, phi, res_real, res_imag;
    if (!PyArg_ParseTuple(args, "Iidd", &l, &m, &theta, &phi))
        return NULL;
    std::complex<double> res = boost::math::spherical_harmonic(l, m, theta, phi);
    res_real = real(res);
    res_imag = imag(res);
    return Py_BuildValue("dd", res_real, res_imag);
}

PyObject *spbessel_wrap(PyObject *self, PyObject *args){
    uint n;
    double x, res;
    if (!PyArg_ParseTuple(args, "Id", &n, &x))
        return NULL;
    res = boost::math::sph_bessel(n, fabs(x)) * (((x<0)&&(n&1))*(-2)+1);
    return Py_BuildValue("d", res);
}


// Module methods
static PyMethodDef boost_math_methods[] = {
    {"spharm", (PyCFunction) spharm_wrap, METH_VARARGS,
        "Return the spherical harmonics Y of l, m, theta, phi."},
    {"spbessel", (PyCFunction) spbessel_wrap, METH_VARARGS,
        "Return the spherical bessel j of l, x."},

    {NULL, NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC init_boost_math(void) {
    PyObject* m;
    m = Py_InitModule3("_boost_math", boost_math_methods,
    "Wrapper for boost C++ math code.");

    import_array();
}
