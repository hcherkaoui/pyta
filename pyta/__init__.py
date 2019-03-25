# coding: utf-8
""" Implement the temporal proximal operator of the Total Activation method."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

from joblib import Parallel, delayed
import numpy as np

from .opt import fista
from .loss_grad import _grad_t, _obj_t
from .hrf import spm_hrf
from .prox import _soft_th


def _prox_t(y, lbda, h, freeze_support=False, mask_support=None,
            solver_kwargs=dict(), verbose=0):
    """ The temporal proximal operator for one voxel.
    """
    hth = np.convolve(h[::-1], h)
    hty = np.convolve(h[::-1], y, mode='valid')

    if freeze_support:
        msg = "if freeze_support is True, mask_support should be given"
        assert mask_support is not None, msg

        def _grad(x):
            # global: y, hth, hty
            return mask_support * _grad_t(x, hth, hty)

    else:
        def _grad(x):
            # global: y, hth, hty
            return _grad_t(x, hth, hty)

    def _obj(x):
        # global: y, h, lbda
        return _obj_t(x, y, h, lbda)

    def _prox(x, step_size):
        # global: lbda
        return _soft_th(x, lbda, step_size)

    if 'x0' not in solver_kwargs:
        solver_kwargs['x0'] = np.zeros(len(y) - len(h) + 1)
    solver_kwargs.update(dict(grad=_grad, obj=_obj, prox=_prox, max_iter=500,
                              name='_prox_t'))
    return fista(**solver_kwargs)


def prox_t(Y, t_r, lbda, lbda_type='ratio', h=None, len_h=None,
           unbiased=False, solver_kwargs=dict(), n_jobs=1, verbose=0):
    """ The temporal proximal operator for full brain.
    """
    # define the HRF
    msg = "Either provide the HRF or its dimension"
    assert (h is not None) or (len_h is not None), msg
    if h is None:
        h = spm_hrf(t_r, len_h)

    # fix temporal regularization
    if lbda_type == 'ratio':
        lbda_max = np.max([np.max(np.abs(
                   np.cumsum(np.convolve(h[::-1], y, mode='valid')[::-1])[::-1]
                            )) for y in Y])
        lbda = lbda * lbda_max
        if verbose > 0:
            print("Lambda temporal max = {0:.3e}, "
                  "fixed lambda temporal = {1:.3e}".format(lbda_max, lbda))
    elif lbda_type == 'fixe':
        lbda = lbda
        if verbose > 0:
            print("Fixed lambda temporal = {0:.3e}".format(lbda))
    else:
        raise ValueError("lbda_type should be ['ratio', 'fixe']"
                         ", not {0}".format(lbda_type))

    # Estimate the z signal (neural activation signal)
    list_z = Parallel(n_jobs=n_jobs)(delayed(_prox_t)(
                                    y=y, lbda=lbda, h=h, verbose=0,
                                    solver_kwargs=solver_kwargs)
                                    for y in Y)

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
