# coding: utf-8
""" Unittests module on loss and grad functions."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from carpet.checks import check_tensor
from pyta.loss_and_grad import _obj_t_analysis, _grad_t_analysis
from pyta.hrf_model import double_gamma_hrf
from pyta.lista_analysis import LpgdTautStringHRF


def test_loss_coherence():
    """ Test the loss function coherence. """
    t_r = 1.0
    n_time_hrf = 30
    n_time_valid = 100
    n_time = n_time_valid + n_time_hrf - 1
    n_samples = 10
    h = double_gamma_hrf(t_r, n_time_hrf)
    u = np.random.randn(n_samples, n_time_valid)
    x = np.random.randn(n_samples, n_time)
    u_ = check_tensor(u)
    x_ = check_tensor(x)
    lbda = 0.1
    kwargs = dict(h=h, n_times_valid=n_time_valid, n_layers=10)
    net_solver = LpgdTautStringHRF(**kwargs)
    loss = float(net_solver._loss_fn(x_, lbda, u_))
    loss_ = _obj_t_analysis(u, x, h, lbda)
    np.testing.assert_allclose(loss, loss_, rtol=1e-3)


def test_gradient():
    """ Test the gradients. """
    h = double_gamma_hrf(1.0, 30)
    n_times = 100
    n_times_valid = n_times - len(h) + 1

    y = np.random.randn(1, n_times)
    x0 = np.random.randn(1, n_times_valid)

    hth = np.convolve(h[::-1], h)
    htY = np.r_[[np.convolve(h[::-1], y_, mode='valid') for y_ in y]]

    def grad(x):
        return _grad_t_analysis(x, hth, htY)

    def approx_grad(x):
        def obj(x):
            return _obj_t_analysis(x, y, h, 0.0)
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
