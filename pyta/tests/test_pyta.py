# coding: utf-8
""" Unittests module."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from prox_tv import tv1_1d
from pyta.loss_and_grad import _obj_t_analysis, _grad_t_analysis
from pyta.optim import fista
from pyta.hrf_model import double_gamma_hrf, make_toeplitz
from pyta.utils import estimate_Lipsch_cst


def test_gradient():
    """ Test the gradients.
    """
    h = double_gamma_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    y = np.random.randn(1, n_times)
    x0 = np.random.randn(1, n_times_valid)

    H = make_toeplitz(h, n_times_valid).T
    HtH = H.dot(H.T)
    Hty = y.dot(H.T)

    def grad(x):
        return _grad_t_analysis(x, HtH, Hty)

    def approx_grad(x):
        def obj(x):
            return _obj_t_analysis(x, y, H, 0.0)
        return approx_fprime(x.ravel(), obj, epsilon=1e-10)

    err = np.linalg.norm(approx_grad(x0) - grad(x0))
    err /= np.linalg.norm(approx_grad(x0))
    try:
        assert err < 1.0e-2
    except AssertionError as e:
        plt.figure("[{0}] Failed estimation".format("approx-grad/grad"))
        plt.plot(approx_grad(x0).T, lw=3.0, label="approx-grad")
        plt.plot(grad(x0).T, ls='--', lw=3.0, label="grad")
        plt.title("Approximated gradient / closed-form gradient")
        plt.legend()
        plt.show()
        raise e


@pytest.mark.parametrize('momentum', [True, False])
@pytest.mark.parametrize('max_iter', [1, 10, 100])
def test_fista_decrease(momentum, max_iter):
    """ Test that the BOLD-TV cost function deacrease.
    """
    h = double_gamma_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    y = np.random.randn(1, n_times)
    x0 = np.random.randn(1, n_times_valid)

    H = make_toeplitz(h, n_times_valid).T
    HtH = H.dot(H.T)
    Hty = y.dot(H.T)

    step_size = np.linalg.norm(HtH, ord=2) ** 2
    lbda = 1.0

    def grad(x):
        return _grad_t_analysis(x, HtH, Hty)

    def obj(x):
        return _obj_t_analysis(x, y, H, lbda=lbda)

    def prox(x, step_size):
        return np.r_[[tv1_1d(x_, lbda * step_size) for x_ in x]]

    x1 = fista(grad, obj, prox, x0, momentum=momentum, max_iter=max_iter,
               step_size=step_size)

    assert _obj_t_analysis(x0, y, H, lbda) >= _obj_t_analysis(x1, y, H, lbda)


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
        return x - y

    def obj(x):
        return 0.5 * np.sum(np.square(x - y))

    def prox(x, step_size):
        return x

    if step_size == '1/Lipsch':
        def AtA(x):
            return x
        step_size = 0.9 / estimate_Lipsch_cst(AtA, n_times_valid,
                                              max_iter=1000)

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
