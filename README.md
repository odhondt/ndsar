# NDSAR filters for multidimensional SAR speckle filtering

## Description

Implementations of two speckle filters based on the nonlocal principle which can be applied to any type of SAR data (Polarimetric, Tomographic, Inteferometric, Multi-temporal, PolInSAR). 

- The NDSAR-BLF is a bilateral filter adapted to covariance matrices obtained from SLC multi-dimensional images.
- The NDSAR-NLM is a generalization of the previous method which computes similarities on square patches instead of individual pixels. It is more robust to speckle than the bilateral but requires more computational power, depending on the user selected patch size.

If you use one of these methods in your paper, please cite the following publication:

O. D’Hondt, C. López-Martínez , S. Guillaso and O. Hellwich.
**Nonlocal Filtering Applied to 3-D Reconstruction of Tomographic SAR Data.**
_IEEE Transactions on Geoscience and Remote Sensing, 2018, 56, 272-285_ 


