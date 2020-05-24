""" This module gathers optimization functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import time
import warnings
import numpy as np
from scipy.optimize.linesearch import line_search_armijo


def fista(grad, obj, prox, x0, momentum=True, max_iter=100, step_size=None,
          early_stopping=True, eps=np.finfo(np.float32).eps, times=False,
          debug=False, verbose=0, name="Optimization"):
    """ F/ISTA algorithm. """
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


def fbs(y, prox_t, prox_s, update_weights=[0.5, 0.5], max_iter=10,
        name='Optimization', obj=lambda x: -1, verbose=False):
    """ Forward Backward Splitting algorithm. """
    x_s = np.zeros_like(y)
    x_t = np.zeros_like(y)
    x = np.zeros_like(y)
    w_t, w_s = update_weights

    if verbose:
        print(f"[{name}] progress {0}%", end="\r")

    l_loss = []
    for ii in range(max_iter):

        if w_t > 0:
            x_t = x_t + prox_t(x - x_t + y) - x

        if w_s > 0:
            x_s = x_s + prox_s(x - x_s + y) - x

        x = w_t * x_t + w_s * x_s

        l_loss.append(obj(x))

        if verbose > 0:
            print(f"[{name}] progress {100. * (ii + 1) / max_iter}%"
                  f" - loss={l_loss[-1]:.4e}", end="\r")

    return x, l_loss
