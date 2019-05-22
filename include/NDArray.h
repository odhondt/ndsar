/*
 #
 #  File        : NDArray.h
 #                ( C++ header file )
 #
 #  Description : A minimal C++ class for multidimensional arrays
 #                inspired from the CImg library.
 #
 #  Copyright   : Olivier D'Hondt
 #                (https://odhondt.github.io)
 #                (https://github.com/odhondt)
 #
 #  License     : CeCILL v2.0
 #                ( http://www.cecill.info/licences/Licence_CeCILL_V2-en.html )
 #
 #  This software is governed by the CeCILL  license under French law and
 #  abiding by the rules of distribution of free software.  You can  use,
 #  modify and/ or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and  rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
 #
*/


#define arr_for1(bound,p) for (int p = 0; p<(int)(bound); ++p)
#define arr_forI(img,i) arr_for1((img)._dimi,i)
#define arr_forJ(img,j) arr_for1((img)._dimj,j)
#define arr_forK(img,k) arr_for1((img)._dimk,k)
#define arr_forL(img,l) arr_for1((img)._diml,l)
#define arr_forIJ(img,i,j) arr_forI(img,i) arr_forJ(img,j) 
#define arr_forIJK(img,i,j,k) arr_forIJ(img,i,j) arr_forK(img,k)
#define arr_forIJL(img,i,j,l) arr_forIJ(img,i,j) arr_forL(img,l)
#define arr_forIJKL(img,i,j,k,l) arr_forIJK(img,i,j,k) arr_forL(img,l)
#define arr_forKL(img,k,l) arr_forK(img,k) arr_forL(img,l)

template <typename T> 
struct NDArray 
{

  unsigned int _dimi, _dimj, _dimk, _diml;
  T* _data;
  bool _is_shared;
 
  NDArray():_dimi(0),_dimj(0),_dimk(0),_diml(0),_data(0),_is_shared(false) {}


  // constructor for extisting data buffer
  NDArray(T* const dataptr, int dimi, int dimj=0, int dimk=0, int diml=0) {
    _dimi = dimi;
    _dimj = dimj;
    _dimk = dimk;
    _diml = diml;
    _data = dataptr;
    _is_shared = true;
  }

  // constructor with initial value
  NDArray(int dimi, int dimj=0, int dimk=0, int diml=0, T val=0) {
    _dimi = dimi;
    _dimj = dimj;
    _dimk = dimk;
    _diml = diml;
    _data = new T[_dimi*_dimj*_dimk*_diml];
    _is_shared = false;
    arr_forIJKL((*this), i, j, k, l) (*this)(i,j,k,l) = val;
  }

  
  NDArray<T>& assign(int dimi, int dimj=0, int dimk=0, int diml=0, T val=0) {
    _dimi = dimi;
    _dimj = dimj;
    _dimk = dimk;
    _diml = diml;
    _data = new T[_dimi*_dimj*_dimk*_diml];
    _is_shared = false;
    arr_forIJKL((*this), i, j, k, l) (*this)(i,j,k,l) = val;
    return (*this);
  }

  ~NDArray() {
    if(!_is_shared) delete _data;
  }

  T& operator()(const unsigned int i) {
    return _data[_diml*_dimk*_dimj*i];
  }
  const T& operator()(const unsigned int i) const {
    return _data[_diml*_dimk*_dimj*i];
  }

  T& operator()(const unsigned int i, const unsigned int j) {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j];
  }
  const T& operator()(const unsigned int i, const unsigned int j) const {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j];
  }

  T& operator()(const unsigned int i, const unsigned int j, const unsigned int k) {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j + _diml*k];
  }
  const T& operator()(const unsigned int i, const unsigned int j, const unsigned int k) const {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j + _diml*k];
  }

  T& operator()(const unsigned int i, const unsigned int j, const unsigned int k, const unsigned int l) {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j + _diml*k + l];
  }
  const T& operator()(const unsigned int i, const unsigned int j, const unsigned int k, const unsigned int l) const {
    return _data[_diml*_dimk*_dimj*i + _diml*_dimk*j + _diml*k + l];
  }

  int dimI() const {
    return (int)_dimi;
  }
  int dimJ() const {
    return (int)_dimj;
  }
  int dimK() const {
    return (int)_dimk;
  }
  int dimL() const {
    return (int)_diml;
  }

  // copy constructor
  NDArray(const NDArray<T> &arr) {
    _dimi = arr._dimi;
    _dimj = arr._dimj;
    _dimk = arr._dimk;
    _diml = arr._diml;
    _is_shared = false;
    arr_forIJKL((*this), i, j, k, l) (*this)(i,j,k,l) = arr(i,j,k,l);
  }

    // Return a pointer to the first value.
    T* data() {
      return _data;
    }

    // Return a pointer to the first value const.
    const T* data() const {
      return _data;
    }
};

