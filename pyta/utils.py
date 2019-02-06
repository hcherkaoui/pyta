# coding: utf-8
""" This module gathers usefull usefull functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import warnings
import cProfile
import numpy as np


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
