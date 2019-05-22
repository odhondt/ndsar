# NDSAR filters for multidimensional SAR speckle filtering

## Description

Python/C++ implementations of two speckle filters based on the nonlocal principle which can be applied to any type of SAR data (Polarimetric, Tomographic, Inteferometric, Multi-temporal, PolInSAR). 

- The **NDSAR-BLF** is a bilateral filter adapted to covariance matrices obtained from SLC multi-dimensional images.
- The **NDSAR-NLM** is a generalization of the previous method which computes similarities on square patches instead of individual pixels. It is more robust to speckle than the bilateral but requires more computational power, depending on the user selected patch size.

If you use one of these methods in your paper, please cite the following publication:

O. D’Hondt, C. López-Martínez , S. Guillaso and O. Hellwich.
**Nonlocal Filtering Applied to 3-D Reconstruction of Tomographic SAR Data.**
_IEEE Transactions on Geoscience and Remote Sensing, 2018, 56, 272-285_   

If you use only the NDSAR-BLF on PolSAR data, you may also cite:

D'Hondt, O., Guillaso, S. and Hellwich, O. 
**Iterative Bilateral Filtering of Polarimetric SAR Data.** 
_IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,  2013, 6, 1628-1639_

## Installation

- Clone the repository in a folder contained in you python path.
- Build the functions with `./cl_build.sh`

### Requirements

- numpy
- cython
- gcc
- openmp

## Usage

In your favorite python environment, import the filters with

```python
from ndsar import *
```

Then, type the name of the function followed by `?` to get help on how to use the function. Ex: `ndsarnlm?`. 

There are four available filters:

- `ndsarnlm`: nonlocal filter for covariance matrices (multi-dimensional SAR images)
- `ndsarblf`: bilateral filter for covariance matrices (multi-dimensional SAR images)
- `sarnlm`: nonlocal filter for intensity images  (single-channel SAR images)
- `sarblf`: bilateral filter for intensiity images  (single-channel SAR images)


