# coding: utf-8
""" Implement the temporal proximal operator of the Total Activation method."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import collections
import numpy as np
from joblib import Parallel, delayed

from .utils import _prox_t, _prox_s, spm_hrf, fbs


def prox_t(Y, t_r, lbda, lbda_type='ratio', h=None, len_h=None,
           unbiased=False, solver_kwargs=dict(), n_jobs=1, verbose=0):
    """ The temporal proximal operator for full brain. """

    # define the HRF
    msg = "Either provide the HRF or its dimension"
    assert (h is not None) or (len_h is not None), msg
    if h is None:
        h = spm_hrf(t_r, len_h)

    # fix temporal regularization
    if lbda_type == 'ratio':
        lbda_max = np.array([np.max(np.abs(
                   np.cumsum(np.convolve(h[::-1], y, mode='valid')[::-1])[::-1]
                            ))
                   for y in Y])
        lbda = lbda * lbda_max
        if verbose > 0:
            print("Lambda temporal max = {0:.3e}, "
                  "fixed lambda temporal = {1:.3e}".format(max(lbda_max),
                                                           max(lbda)))
    elif lbda_type == 'fixed':
        lbda = lbda
        if verbose > 0:
            print("Fixed lambda temporal = {0:.3e}".format(max(lbda)))
    else:
        raise ValueError("lbda_type should be ['ratio', 'fixed']"
                         ", not {0}".format(lbda_type))

    if not isinstance(lbda, (collections.Sequence, np.ndarray)):
        lbda = [lbda] * Y.shape[0]

    # Estimate the z signal (neural activation signal)
    list_z = Parallel(n_jobs=n_jobs)(delayed(_prox_t)(
                                    y=y, lbda=lbda_, h=h, verbose=0,
                                    solver_kwargs=solver_kwargs)
                                    for y, lbda_ in zip(Y, lbda))

    if unbiased:
        # freeze the support after a thresholding to increase sparsity
        if verbose > 0:
            print("Re-run the estimation with freezed support")
        l_mask_support = []
        for z in list_z:
            s = 0.1  # 10% of sparsity for z
            n = int(s * len(z))
            mask = np.zeros_like(z, dtype=int)
            mask_ = np.argpartition(np.abs(z), -n)[-n:]
            mask[mask_] = 1
            l_mask_support.append(mask)
        list_z = Parallel(n_jobs=n_jobs)(delayed(_prox_t)(
                                    y=y, lbda=0.0, h=h,
                                    verbose=0, freeze_support=True,
                                    mask_support=mask_support)
                                    for mask_support, y
                                    in zip(l_mask_support, Y))

    z = np.vstack([z_ for z_ in list_z])
    Lz = np.vstack([np.cumsum(z_) for z_ in list_z])
    x = np.vstack([np.convolve(h, np.cumsum(z_)) for z_ in list_z])

    return x, Lz, z


def prox_s(Y, lbda, vol_shape, n_jobs=1, verbose=0):
    """ The spatial proximal operator for full brain. """
    list_x = Parallel(n_jobs=n_jobs, verbose=verbose)(
             delayed(_prox_s)(y.reshape(vol_shape), lbda) for y in Y.T)
    x = np.vstack([x_.flatten() for x_ in list_x]).T
    return x


def ta(Y, t_r, h, vol_shape, lbda_t, lbda_s, update_weights=[0.5, 0.5],
       max_iter=30, n_jobs=1, verbose=True):
    """ Main Total Activation interface function. """

    lbdas_max = np.array([np.max(np.abs(
                 np.cumsum(np.convolve(h[::-1], y, mode='valid')[::-1])[::-1]
                              )) for y in Y])
    lbda_t *= lbdas_max

    def _prox_t(Y):
        return prox_t(Y, t_r, lbda_t, lbda_type='fixed', h=h, n_jobs=n_jobs,
                      solver_kwargs=dict(max_iter=200), verbose=0)[0]

    def _prox_s(Y):
        return prox_s(Y, lbda_s, vol_shape, n_jobs=n_jobs, verbose=0)

    X = fbs(Y, _prox_t, _prox_s, update_weights=update_weights,
            max_iter=max_iter, verbose=verbose)

    return prox_t(X, t_r, lbda_t, lbda_type='fixed', h=h, n_jobs=n_jobs,
                  unbiased=True, solver_kwargs=dict(max_iter=600), verbose=0)
