""" This module gathers usefull functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import warnings
import cProfile
import numpy as np


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance

    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{0} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def compute_lbda_max(y, h, per_sample=False):
    """ Compute lambda max. """
    dim_x, dim_y, dim_z, n_times = y.shape
    n_samples = dim_x * dim_y * dim_z
    y_ravel = y.reshape(n_samples, n_times)

    lbdas_max = np.array([np.max(np.abs(
        np.cumsum(np.convolve(h[::-1], y_, mode='valid')[::-1])[::-1]
        )) for y_ in y_ravel])

    if per_sample:
        return lbdas_max
    else:
        return np.max(lbdas_max)


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
