.. _preprocessing:

Pre Processing
==============
In order to improve and/or reduce the computational time needed, we implemented two
methods of pre processing the time series used. These can be found in the portfolioML.data.preprocessing
module.

| Scikit-learn package sklearn.decomposition.PCA is used for PCA (https://pywavelets.readthedocs.io/en/latest/)
| PyWavelets package is used for Discrete Wavelet Transform (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)


PCA - Principal Component Analysis
----------------------------------
Principal Component Analysis is a common way to reduce the dimensionality of the data
through a projection onto a less-dimensional space composed by the so-called principal
components, i.e. the eigenvectors of the covariance matrix of the data.

Our implementation articulates as follow:
Starting from a 365-dimensional space (equal to the number of companies we tracked from
the available data of the S&P500 index in our entire period of study), we choosed to keep
the first 250 components that correspond to the eigenvectors of the covariance matrix
with the greatest eigenvalues. This resulted in a preserved variance of 0.94. Then, since
these orthogonal vectors are linear combinations of the ones from the higher dimensional
space, we selected the features that most contribute to them.


DWT - Discrete Wavelet Transform
--------------------------------
Some clever stuff on DWT


.. code-block:: python

   some smart code
