.. -*- mode: rst -*-

|Travis|_ |Codecov|_ |Python27| |Python35|


.. |Travis| image:: https://travis-ci.com/CherkaouiHamza/pyta.svg?branch=master
.. _Travis: https://travis-ci.com/CherkaouiHamza/pyta

.. |Codecov| image:: https://codecov.io/gh/CherkaouiHamza/pyta/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/CherkaouiHamza/pyta

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/scikit-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn


pyTA
====
py Total Activation (pyTA) implements the temporal proximal operator of the
Total Activation approach (ie deconvolution via a total variation constraint
for fMRI data).

 [1] F. I. Karahanoglu, C. Caballero Gaudes, F. Lazeyras, D. Van De Ville,
 "Total Activation: FMRI Deconvolution through Spatio-Temporal Regularization",
 NeuroImage, vol. 73, pp. 121-134, 2013


Important links
===============

- Official source code repo: https://github.com/CherkaouiHamza/pyta
- Original Matlab implementation: https://miplab.epfl.ch/index.php/software/total-activation


Dependencies
============

* numpy >= 1.14.0
* numba >= 0.40.1
* scipy >= 1.0.0
* joblib >= 0.11
* nibabel >= 2.3.2
* matplotlib >= 2.1.2 (for examples)


Install
=======

In order to perform the installation, run the following command from the pyTA directory::

    pip install -r 'numpy>=1.14.0' 'scipy>=1.0.0' 'joblib>=0.11' 'matplotlib>=2.1.2' 'numba>=0.40.1'
    python setup.py install --user


Testing
=======
In order to perform the unit tests, run the following command from the pyTA directory::

    pip install pytest --user
    pytest


Development
===========

You can check the latest sources with the command::

    git clone https://github.com/CherkaouiHamza/pyta
