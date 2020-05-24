.. -*- mode: rst -*-

|Travis|_ |Codecov|_ |Python36|


.. |Travis| image:: https://travis-ci.com/hcherkaoui/pyta.svg?branch=master
.. _Travis: https://travis-ci.com/hcherkaoui/pyta

.. |Codecov| image:: https://codecov.io/gh/hcherkaoui/pyta/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hcherkaoui/pyta

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/scikit-learn


pyTA
====
py Total Activation (pyTA) implements the Total Activation approach
(ie deconvolution via a spatial and temporal total variation constraint
for fMRI data).

 [1] F. I. Karahanoglu, C. Caballero Gaudes, F. Lazeyras, D. Van De Ville,
 "Total Activation: FMRI Deconvolution through Spatio-Temporal Regularization",
 NeuroImage, vol. 73, pp. 121-134, 2013


Important links
===============

- Official source code repo: https://github.com/hcherkaoui/pyta
- Original Matlab implementation: https://miplab.epfl.ch/index.php/software/total-activation


Dependencies
============

* numpy >= 1.14.0
* scipy >= 1.0.0
* joblib >= 0.11
* prox_tv
* carpet
* matplotlib >= 2.1.2 (for examples)


Install
=======

In order to perform the installation, run the following command from the pyTA directory::

    pip install -r 'joblib>=0.11' 'numpy>=1.14.0' 'scipy>=1.0.0' 'scikit-learn>=0.19.2' 'matplotlib>=2.1.2' 'prox_tv' 'carpet'
    python setup.py install --user


Testing
=======
In order to perform the unit tests, run the following command from the pyTA directory::

    pip install pytest --user
    pytest


Development
===========

You can check the latest sources with the command::

    git clone https://github.com/hcherkaoui/pyta
