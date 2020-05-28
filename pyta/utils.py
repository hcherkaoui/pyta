""" This module gathers usefull functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import warnings
import cProfile
import numpy as np


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of different number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))


def compute_lbda_max(H, y, per_sample=False):
    """ Compute lambda max. """
    n_times_valid, n_times = H.shape
    dim_x, dim_y, dim_z, _ = y.shape
    y_ravel = y.reshape(dim_x * dim_y * dim_z, n_times)
    u_shape = (y_ravel.shape[0], n_times_valid)
    L = np.triu(np.ones((n_times_valid, n_times_valid)))
    S = H.sum(axis=0)
    c = (y_ravel.dot(S) / (S ** 2).sum())[:, None] * np.ones(u_shape)
    lmbd_max = np.abs((y_ravel - c.dot(H)).dot(H.T).dot(L.T))
    if per_sample:
        return lmbd_max.max(axis=1, keepdims=True)
    else:
        return lmbd_max.max()


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
