# coding: utf-8
""" This module gathers usefull usefull functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import sys
import time
import warnings
import cProfile
import numpy as np
import numba
from scipy.stats import gamma
from scipy.optimize.linesearch import line_search_armijo
from prox_tv import tvgen


MIN_DELTA = 0.5
MAX_DELTA = 2.0


def spm_hrf(t_r, n_times_atom):
    """ Interface HRF function.
    """
    _hrf = _spm_hrf(delta=1.0, t_r=t_r, dur=n_times_atom * t_r)[0]
    n = n_times_atom - len(_hrf)
    if n < 0:
        _hrf = _hrf[:n]
    elif n > 0:
        _hrf = np.hstack([_hrf, np.zeros(n)])
    return _hrf


def _spm_hrf(delta, t_r=1.0, dur=60.0, normalized_hrf=True, dt=0.001,
             p_delay=6, undershoot=16.0, p_disp=1.0, u_disp=1.0,
             p_u_ratio=0.167, onset=0.0):
    """ SPM canonical HRF with a time scaling parameter.
    """
    if (delta < MIN_DELTA) or (delta > MAX_DELTA):
        raise ValueError("delta should belong in [{0}, {1}]; wich correspond"
                         " to a max FWHM of 10.52s and a min FWHM of 2.80s"
                         ", got delta = {2}".format(MIN_DELTA, MAX_DELTA,
                                                    delta))

    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, int(float(dur) / dt)) - float(onset) / dt
    scaled_time_stamps = delta * t

    peak = gamma.pdf(scaled_time_stamps, p_delay/p_disp, loc=dt/p_disp)
    undershoot = gamma.pdf(scaled_time_stamps, undershoot/u_disp,
                           loc=dt/u_disp)
    hrf = peak - p_u_ratio * undershoot

    if normalized_hrf:
        hrf /= np.max(hrf + 1.0e-30)

    hrf = hrf[::int(t_r/dt)]
    t_hrf = t[::int(t_r/dt)]

    return hrf, t_hrf


def _soft_th(x, lbda, step_size):
    """ Prox for the temporal prox for one voxels.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lbda * step_size, 0)


@numba.jit((numba.float64[:], numba.float64[:], numba.int64), cache=True,
           nopython=True)
def hth_(hth, x, n):
    """ HRF convolve and time-reversed convolution operator. """
    hth_x = np.empty(n)
    for t in range(n):
        hth_x[t] = np.sum(hth * x[t: t + len(hth)])
    return hth_x


def _grad_t(x, hth, hty=None):
    """ Gradient for the temporal prox for one voxels.
    """
    Lx = np.r_[np.zeros(int(len(hth)/2)),
               np.cumsum(x),
               np.zeros(int(len(hth)/2))]
    grad = hth_(hth, Lx, len(x))

    if hty is not None:
        grad -= hty

    return np.cumsum(grad[::-1])[::-1]


def _obj_t(x, y, h, lbda):
    """ Cost func for the temporal prox for one voxels.
    """
    return (0.5 * np.sum(np.square(np.convolve(h, np.cumsum(x)) - y))
            + lbda * np.sum(np.abs(x)))


def estimate_Lipsch_cst(AtA, x_len, max_iter=100):
    """ Estimate the Lipschcitz constant associated with the op 'LththL'.
    """
    x_old = np.random.randn(x_len)
    for ii in range(max_iter):
        x_new = AtA(x_old) / np.linalg.norm(x_old)
        diff = np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old))
        diff /= np.linalg.norm(x_old)
        if diff < np.finfo(np.float32).eps:
            warnings.warn("Spectral radius estimation converge in"
                          "{0} iters.".format(ii))
            return np.linalg.norm(x_new)
    warnings.warn("Spectral radius estimation did not converge.")
    return np.linalg.norm(x_new)


def fista(grad, obj, prox, x0, momentum=True, max_iter=100, step_size=None,
          early_stopping=True, eps=np.finfo(np.float32).eps, times=False,
          debug=False, verbose=0, name="Optimization"):
    """ F/ISTA algorithm.
    """
    if verbose and not debug:
        warnings.warn("Can't have verbose if cost-func is not computed, "
                      "enable it by setting debug=True")

    adaptive_step_size = False
    if step_size is None:
        adaptive_step_size = True
        step_size = 1.0

    # prepare the iterate
    t = t_old = 1
    z_old = np.zeros_like(x0)
    x = np.copy(x0)

    # saving variables
    pobj_, times_ = [], []

    # precompute L.op(y)
    if adaptive_step_size:
        old_fval = obj(x)

    # main loop
    for ii in range(max_iter):

        if times:
            t0 = time.time()

        grad_ = grad(x)

        # step-size
        if adaptive_step_size:
            step_size, _, old_fval = line_search_armijo(
                    obj, x.ravel(), -grad_.ravel(), grad_.ravel(),
                    old_fval, c1=1.0e-5, alpha0=step_size)
            if step_size is None:
                step_size = 0.0

        # descent step
        z = prox(x - step_size * grad_, step_size)

        # fista acceleration
        if momentum:
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_old**2))
            x = z + (t_old - 1.0) / t * (z - z_old)
        else:
            x = z

        # savings
        if debug:
            if adaptive_step_size:
                pobj_.append(old_fval)
            else:
                pobj_.append(obj(x))

        # printing
        if debug and verbose > 0:
            print("[{0}] Iteration {1} / {2}, "
                  "loss = {3}".format(name, ii+1, max_iter, pobj_[ii]))

        # early-stopping
        l1_diff = np.sum(np.abs(z - z_old))
        if l1_diff <= eps and early_stopping:
            if debug:
                print("\n-----> [{0}] early-stopping "
                      "done at {1}/{2}".format(name, ii+1, max_iter))
            break
        if l1_diff > np.finfo(np.float64).max:
            raise RuntimeError("[{}] {} have diverged during.".format(name,
                               ["ISTA", "FISTA"][momentum]))

        # update iterates
        t_old = t
        z_old = z

        # savings
        if times:
            times_.append(time.time() - t0)

    if not times and not debug:
        return x
    if times and not debug:
        return x, times_
    if not times and debug:
        return x, pobj_
    if times and debug:
        return x, pobj_, times_


def fbs(Y, prox_t, prox_s, update_weights=[0.5, 0.5], max_iter=10,
        verbose=False):
    """ Forward Backward Splitting algorithm. """
    X_s = np.zeros_like(Y)
    X_t = np.zeros_like(Y)
    X = np.zeros_like(Y)
    w_t, w_s = update_weights

    for ii in range(max_iter):

        if w_t > 0:
            X_t = X_t + prox_t(X - X_t + Y) - X

        if w_s > 0:
            X_s = X_s + prox_s(X - X_s + Y) - X

        X = w_t * X_t + w_s * X_s

        if verbose:
            sys.stdout.write("*" * ii)
            sys.stdout.flush()

    sys.stdout.write("\n")

    if verbose:
        print()

    return X


def _prox_t(y, lbda, h, freeze_support=False, mask_support=None,
            solver_kwargs=dict(), verbose=0):
    """ The temporal proximal operator for one voxel. """

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
    solver_kwargs.update(dict(grad=_grad, obj=_obj, prox=_prox,
                              name='_prox_t'))
    return fista(**solver_kwargs)


def _prox_s(y, lbda):
    """ The spatial proximal operator for one voxel. """
    return tvgen(y, [lbda, lbda, lbda], [1, 2, 3], [1, 1, 1])


def profile_this(fn):
    """ Profiling decorator.
    Example of usage:
    >>> def f1():
    >>>     return [x * x for x in range(10000)]
    >>> def f2():
    >>>     return [x * x for x in range(30000)]
    >>> @profile_this
    >>> def test():
    >>>     f1()
    >>>     f2()
    >>> test()
    'snakeviz' to view the produced report.
    """
    def profiled_fn(*args, **kwargs):
        filename = fn.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        prof.dump_stats(filename)
        return ret

    return profiled_fn
