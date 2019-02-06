# coding: utf-8
""" Unittests module."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

from pyta import _prox_t
from pyta.loss_grad import _obj_t, _grad_t
from pyta.prox import _soft_th
from pyta.utils import estimate_Lipsch_cst
from pyta.hrf import spm_hrf
from pyta.opt import fista


def test_gradient():
    """ Test the gradients.
    """
    h = spm_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    y = np.random.randn(n_times)
    x0 = np.random.randn(n_times_valid)

    hth = np.convolve(h[::-1], h)
    hty = np.convolve(h[::-1], y, mode='valid')

    def grad(x):
        # global: y, hth, hty
        return _grad_t(x, hth, hty, h)

    def approx_grad(x):
        return approx_fprime(x, _obj_t, 1.0e-12, y, h, 0.0)

    err = np.linalg.norm(approx_grad(x0) - grad(x0))
    err /= np.linalg.norm(approx_grad(x0))
    try:
        assert err < 1.0e-2
    except AssertionError as e:
        plt.figure("[{0}] Failed estimation".format("approx-grad/grad"))
        plt.plot(approx_grad(x0), 'b--', lw=3.0, label="approx-grad")
        plt.plot(grad(x0), 'y-', lw=3.0, label="grad")
        plt.title("Approximated gradient / closed-form gradient")
        plt.legend()
        plt.show()
        raise e


@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('step_size', [None, 1.0e-6])
@pytest.mark.parametrize('max_iter', [1, 10, 100])
def test_fista_decrease(momentum, step_size, max_iter):
    """ Test that the BOLD-TV cost function deacrease.
    """
    h = spm_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    lbda = 1.0

    y = np.random.randn(n_times)
    x0 = np.random.randn(n_times_valid)

    hth = np.convolve(h[::-1], h)
    hty = np.convolve(h[::-1], y, mode='valid')

    def grad(x):
        # global: y, hth, hty
        return _grad_t(x, hth, hty, h=h)

    def obj(x):
        # global: y, h, lbda
        return _obj_t(x, y, h, lbda=0.0)

    def prox(x, step_size):
        # global: lbda
        return _soft_th(x, lbda, step_size)

    x1 = fista(grad, obj, prox, x0, momentum=momentum, max_iter=max_iter,
               step_size=step_size)

    assert _obj_t(x0, y, h, lbda) >= _obj_t(x1, y, h, lbda)


def test_estimate_Lipsch_cst():
    """ Dummy test for estimate_Lipsch_cst.
    """
    def AtA(x):
        return x
    step_size = estimate_Lipsch_cst(AtA, 100)
    np.testing.assert_allclose(step_size, 1.0)


@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('step_size', [None, '1/Lipsch', 1.0])
def test_fista_optimum(momentum, step_size):
    """ Test the F/ISTA implementation of a dummy cost-function
    (with a inversible observation operator: the identity).
    """
    h = np.array([1.0])
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    y = np.random.randn(n_times_valid)

    def grad(x):
        # global: y
        return x - y

    def obj(x):
        # global: y
        return 0.5 * np.sum(np.square(x - y))

    def prox(x, step_size):
        return x

    if step_size == '1/Lipsch':
        def AtA(x):
            return x

        step_size = 0.9 / estimate_Lipsch_cst(AtA, n_times_valid)
    x = fista(grad, obj, prox, np.zeros_like(y), momentum=momentum,
              max_iter=100, step_size=step_size)

    try:
        np.testing.assert_allclose(x, y)
    except AssertionError as e:
        plt.figure("[{0}] Failed estimation".format(__file__))
        plt.plot(y, 'b--', lw=3.0, label="Ref")
        plt.plot(x, 'y-', lw=3.0, label="Test")
        plt.title("Expected vs estimated signal")
        plt.legend()
        plt.show()
        raise e


@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('step_size', [None, '1/Lipsch', 1.0e-6])
def test_prefect_recover(momentum, step_size):
    """ With True HRF, no regularization, prox_t call and an init at the
    True value.
    """
    h = spm_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    lbda = 0.0

    x = np.random.randn(n_times_valid)
    y = np.convolve(h, np.cumsum(x))

    hth = np.convolve(h[::-1], h)

    if step_size == '1/Lipsch':
        def AtA(x):
            return _grad_t(x, hth, hty=None, h=h)
        step_size = 0.9 / estimate_Lipsch_cst(AtA, n_times_valid)

    solver_kwargs = dict(max_iter=100, step_size=step_size,
                         momentum=momentum, x0=x)
    x_ = _prox_t(y, lbda, h, solver_kwargs=solver_kwargs)

    try:
        np.testing.assert_allclose(x, x_)
    except AssertionError as e:
        plt.figure("[{0}][{1}, {2}] Failed estimation".format(
                                          __file__, step_size, momentum))
        plt.plot(x, 'b--', lw=3.0, label="Ref")
        plt.plot(x_, 'y-', lw=3.0, label="Test")
        plt.title("Expected vs estimated signal")
        plt.legend()
        plt.show()
        raise e
