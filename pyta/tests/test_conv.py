# coding: utf-8
""" Unittests module on convolution function."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np
import torch
from pyta.hrf_model import double_gamma_hrf
from pyta.convolution import (hu_numpy, hu_tensor, make_toeplitz, htx_numpy,
                              htx_tensor, hthu_numpy, hthu_tensor,
                              hth_id_u_numpy, hth_id_u_tensor)


def test_convolution():
    """ Test the convolution. """
    t_r = 1.0
    n_time_hrf = 30
    n_time_valid = 100
    n_samples = 10
    h = double_gamma_hrf(t_r, n_time_hrf)
    H = make_toeplitz(h, n_time_valid).T
    u = np.random.randn(n_samples, n_time_valid)
    h_, u_ = torch.Tensor(h), torch.Tensor(u)
    np.testing.assert_allclose(hu_numpy(h, u), u.dot(H), rtol=1e-3)
    np.testing.assert_allclose(hu_tensor(h_, u_).numpy(), u.dot(H), rtol=1e-3)


def test_rev_convolution():
    """ Test the reverse convolution. """
    t_r = 1.0
    n_time_hrf = 30
    n_time_valid = 100
    n_samples = 10
    h = double_gamma_hrf(t_r, n_time_hrf)
    u = np.random.randn(n_samples, n_time_valid)
    x = hu_numpy(h, u)
    h_, x_ = torch.Tensor(h), torch.Tensor(x)
    np.testing.assert_allclose(htx_numpy(h, x), htx_tensor(h_, x_).numpy(),
                               rtol=1e-3)


def test_hth_convolution():
    """ Test the hth-convolution. """
    t_r = 1.0
    n_time_hrf = 30
    n_time_valid = 100
    n_samples = 10
    h = double_gamma_hrf(t_r, n_time_hrf)
    u = np.random.randn(n_samples, n_time_valid)
    hth = np.convolve(h[::-1], h)
    hth_, u_ = torch.Tensor(hth), torch.Tensor(u)
    np.testing.assert_allclose(hthu_numpy(hth, u), hthu_tensor(hth_, u_),
                               rtol=1e-3)


def test_hth_id_convolution():
    """ Test the hth-convolution. """
    t_r = 1.0
    n_time_hrf = 30
    n_time_valid = 100
    n_samples = 10
    h = double_gamma_hrf(t_r, n_time_hrf)
    u = np.random.randn(n_samples, n_time_valid)
    hth = np.convolve(h[::-1], h)
    hth_, u_ = torch.Tensor(hth), torch.Tensor(u)
    np.testing.assert_allclose(hth_id_u_numpy(hth, u),
                               hth_id_u_tensor(hth_, u_),
                               rtol=1e-3)
