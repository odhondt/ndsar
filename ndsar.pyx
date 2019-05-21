cdef extern from "src/ndsar_lib.h":
  cdef void ndsar_blf_cpp(float complex* arr, float complex* arr2, int* c_shp, float gs, float gr, bint trick, bint flat, int method)
  cdef void ndsar_nlm_cpp(float complex* arr, float complex* arr2, int* c_shp, float gs, float gr, int Psiz, bint trick, bint flat, int method)

import numpy as np

def ndsarblf(float complex[:,:,:,::1] varr, float gs=2.8, float gr=1.4,
             method = 'ai', bint trick=True, bint flat=False):
  """ndsarblf(float complex[:,:,:,::1] arr, float gs=2.8, float gr=1.4,
             method = 'ai', bint trick=True, bint flat=False)
  """

  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], varr.shape[2],
                                    varr.shape[3]], dtype=np.int32)

  arr2 = np.zeros_like(varr)
  cdef float complex[:,:,:,::1] varr2 = arr2

  if method == 'ai':
    meth_int = 1
  elif method == 'le':
    meth_int = 2
  elif method == 'ld':
    meth_int = 3
  elif method == 'gl':
    meth_int = 4
  else:
    raise ValueError("Method does not exist")
  ndsar_blf_cpp(&varr[0,0,0,0], &varr2[0,0,0,0], &c_shp[0], gs, gr, trick, flat, meth_int)

  return arr2

def ndsarnlm(float complex[:,:,:,::1] varr, float gs=2.8, float gr=1.4,
             int  psiz=3 ,method = 'ai', bint trick=True, bint flat=False):
  """ndsarnlm(float complex[:,:,:,::1] varr, float gs=2.8, float gr=1.4,
             int  psiz=3 ,method = 'ai', bint trick=True, bint flat=False)
  """
  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], varr.shape[2],
                                    varr.shape[3]], dtype=np.int32)

  arr2 = np.zeros_like(varr)
  cdef float complex[:,:,:,::1] varr2 = arr2

  if method == 'ai':
    meth_int = 1
  elif method == 'le':
    meth_int = 2
  elif method == 'ld':
    meth_int = 3
  elif method == 'gl':
    meth_int = 4
  else:
    raise ValueError("Method does not exist")
  ndsar_nlm_cpp(&varr[0,0,0,0], &varr2[0,0,0,0], &c_shp[0], gs, gr, psiz ,trick, flat, meth_int)

  return arr2

def sarblf(float[:,::1] varr, float gs=2.8, float gr=1.4, method = 'ai', bint trick=True, bint flat=False):
  """sarblf(np.ndarray[np.float32_t, ndim=2] arr, float gs=2.8, float gr=1.4, method = 'ai', bint trick=True, bint flat=False)
  Version for single-channel images (intensity, amplitude). Takes float as an input.
  This is a helper function to be able to use scalar images.
  """

  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], varr.shape[2],
                                    varr.shape[3]], dtype=np.int32)

  cdef float complex[:,::1] varrclx = np.zeros_like(varr, dtype=np.complex64)
  cdef float complex[:,::1] varr2 = np.zeros_like(varr, dtype=np.complex64)

  varrclx.real = varr

  ndsar_blf_cpp(&varrclx[0,0], &varr2[0,0], &c_shp[0], gs, gr, trick, flat, 0)

  return np.real(varr2)

def sarnlm(float[:,::1] varr, float gs=2.8, float gr=1.4, int ps=3, method = 'ai', bint trick=True, bint flat=False):
  """
    sarnlm(np.ndarray[np.float32_t, ndim=2] arr, float gs=2.8, float gr=1.4, int ps, method = 'ai', bint trick=True, bint flat=False):
  Version for single-channel images (intensity, amplitude). Takes float as an input.
  This is a helper function to be able to use scalar images.
  """

  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], varr.shape[2],
                                    varr.shape[3]], dtype=np.int32)

  cdef float complex[:,::1] varrclx = np.zeros_like(varr, dtype=np.complex64)
  cdef float complex[:,::1] varr2 = np.zeros_like(varr, dtype=np.complex64)

  varrclx.real = varr

  ndsar_nlm_cpp(&varrclx[0,0], &varr2[0,0], &c_shp[0], gs, gr, ps, trick, flat, 0)
  #ndsar_nlm_cpp(<float complex*> arrclx.data, <float complex*> arr2.data, <int*> c_shp.data, gs, gr, ps, trick, flat, 0)

  return np.real(varr2)

