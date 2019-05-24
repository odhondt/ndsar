cdef extern from "src/ndsar_lib.h":
  cdef void ndsar_nlm_cpp(float complex* arr, float complex* arr2, int* c_shp,
                          float gs, float gr, int Psiz, bint trick, bint flat,
                          int method)
  cdef void ndsar_blf_cpp(float complex* arr, float complex* arr2, int* c_shp,
                          float gs, float gr, bint trick, bint flat, int method)

import numpy as np

def ndsarnlm(float complex[:,:,:,::1] varr, float gs=2.8, float gr=1.4,
             int  psiz=3 ,method = 'ai', bint trick=True, bint flat=False):
  '''ndsarnlm(cov, gs=2.8, gr=1.4, psiz=3 , method='ai', trick=True, flat=False)

        NDSAR-NLM: Nonlocal means filtering of N-dimensional SAR images.

        Parameters
        ----------
        cov: array, shape (naz, nrg, dim, dim), dtype: np.complex64
            complex covariance image computed from SLC data.
            naz and nrg are the number of pixels in azimuth and range.
            dim is the matrix dimension.

        gs: float
            spatial scale parameter.

        gr: float
            radiometric scale parameter.

        psiz: int
            size of the patches to compute pixel similarities.

        method: string
            distance used to compute radiometric similarities.
            'ai': Affine Invariant
            'le': Log-Euclidean (much faster, recommended)
            'ld': Log Diagonal (equivalent to 'ai' and 'le' but assumes
            covariance is diagonal, uses only intensity information)

        trick: boolean
            underweights central pixel to enforce more filtering (recommended)
            default: True

        flat: boolean
            uses uniform spatial weights instead of Gaussian ones.
            default: False

        Returns
        -------
        fcov: array, shape (naz, nrg, dim, dim)
            filtered covariance image.

        Notes
        -----
        Covariance matrices must have full rank for the 'ai' and 'le' distances
        to be defined. Please ensure that the number of looks of your data is at
        least of dim. Pre-summing is preferred to boxcar multilooking to avoid
        introducing spatial correlation between the pixels.
        '''

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
  else:
    raise ValueError("Method does not exist")
  ndsar_nlm_cpp(&varr[0,0,0,0], &varr2[0,0,0,0], &c_shp[0], gs, gr, psiz,
                trick, flat, meth_int)

  return arr2

def ndsarblf(float complex[:,:,:,::1] varr, float gs=2.8, float gr=1.4,
             method = 'ai', bint trick=True, bint flat=False):
  '''ndsarblf(cov, gs=2.8, gr=1.4, method = 'ai', trick=True, flat=False)

        NDSAR-BLF: Bilateral filtering of N-dimensional SAR images.

        Parameters
        ----------
        cov: array, shape (naz, nrg, dim, dim), dtype: np.complex64
            complex covariance image computed from SLC data.
            naz and nrg are the number of pixels in azimuth and range.
            dim is the matrix dimension.

        gs: float
            spatial scale parameter.

        gr: float
            radiometric scale parameter.

        method: string
            distance used to compute radiometric similarities.
            'ai': Affine Invariant
            'le': Log-Euclidean (much faster, recommended)
            'ld': Log Diagonal (equivalent to 'ai' and 'le' but assumes
            covariance is diagonal, uses only intensity information)

        trick: boolean
            underweights central pixel to enforce more filtering (recommended)
            default: True

        flat: boolean
            uses uniform spatial weights instead of Gaussian ones.
            default: False

        Returns
        -------
        fcov: array, shape (naz, nrg, dim, dim)
        filtered covariance image.

        Notes
        -----
        Covariance matrices must have full rank for the 'ai' and 'le' distances
        to be defined. Please ensure that the number of looks of your data is at
        least of dim. Pre-summing is preferred to boxcar multilooking to avoid
        introducing spatial correlation between the pixels.

        The NDSAR-BLF filter is equivalent to NDSAR-NLM with a patch size of 1.
  '''

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
  else:
    raise ValueError("Method does not exist")
  ndsar_blf_cpp(&varr[0,0,0,0], &varr2[0,0,0,0], &c_shp[0], gs, gr, trick,
                flat, meth_int)

  return arr2


def sarnlm(float[:,::1] varr, float gs=2.8, float gr=1.4, int psiz=3,
           bint trick=True, bint flat=False):
  '''sarnlm(img, gs=2.8, gr=1.4, psiz=3 , trick=True, flat=False)

        SAR-NLM: Nonlocal means filtering of single channel SAR images.

        Parameters
        ----------
        img: array, shape (naz, nrg), dtype: np.float32
            intensity image computed from SLC data.
            naz and nrg are the number of pixels in azimuth and range.

        gs: float
            spatial scale parameter.

        gr: float
            radiometric scale parameter.

        psiz: int
            size of the patches to compute pixel similarities.

        trick: boolean
            underweights central pixel to enforce more filtering (recommended)
            default: True

        flat: boolean
            uses uniform spatial weights instead of Gaussian ones.
            default: False

        Returns
        -------
        fimg: array, shape (naz, nrg)
            filtered image.

        Notes
        -----
        This function calls a c++ routine which is optimized for single
        channel images.

        For single channel images, 'ai', 'le' and 'ld' distances are equivalent
        to the Euclidean distance on the log of intensity.
  '''

  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], 1, 1],
                                   dtype=np.int32)

  cdef float complex[:,::1] varrclx = np.asarray(varr, dtype=np.complex64)
  cdef float complex[:,::1] varr2 = np.zeros_like(varr, dtype=np.complex64)

  ndsar_nlm_cpp(&varrclx[0,0], &varr2[0,0], &c_shp[0], gs, gr, psiz, trick,
                flat, 0)

  return np.ascontiguousarray(np.real(varr2))

def sarblf(float[:,::1] varr, float gs=2.8, float gr=1.4, bint trick=True,
           bint flat=False):
  '''sarblf(img, gs=2.8, gr=1.4, trick=True, flat=False)

        SAR-BLF: Bilateral filtering of single channel SAR images.

        Parameters
        ----------
        img: array, shape (naz, nrg), dtype: np.float32
            intensity image computed from SLC data.
            naz and nrg are the number of pixels in azimuth and range.

        gs: float
            spatial scale parameter.

        gr: float
            radiometric scale parameter.

        trick: boolean
            underweights central pixel to enforce more filtering (recommended)
            default: True

        flat: boolean
            uses uniform spatial weights instead of Gaussian ones.
            default: False

        Returns
        -------
        fimg: array, shape (naz, nrg)
            filtered image.

        Notes
        -----
        This function calls a c++ routine which is optimized for single
        channel images.

        For single channel images, 'ai', 'le' and 'ld' distances are equivalent
        to the Euclidean distance on the log of intensity.
  '''

  cdef int[::1] c_shp = np.asarray([varr.shape[0], varr.shape[1], 1, 1],
                                   dtype=np.int32)

  cdef float complex[:,::1] varrclx = np.asarray(varr, dtype=np.complex64)
  cdef float complex[:,::1] varr2 = np.zeros_like(varr, dtype=np.complex64)

  ndsar_blf_cpp(&varrclx[0,0], &varr2[0,0], &c_shp[0], gs, gr, trick, flat, 0)

  return np.ascontiguousarray(np.real(varr2))


