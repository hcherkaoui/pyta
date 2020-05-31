""" (Pseudo monkey patched) Module to define Optimization Neural Net. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# Authors: Thomas Moreau <thomas.moreau@inria.fr>
# License: BSD (3-clause)

import warnings
import numpy as np
import torch
from carpet.checks import check_tensor
from carpet.utils import init_vuz
from carpet.proximity_tv import ProxTV_l1, RegTV
from carpet.lista_base import DOC_LISTA
from carpet.lista_analysis import _ListaAnalysis
from .utils import lipsch_cst_from_kernel
from .convolution import hth_id_u_tensor, htx_tensor, hu_tensor, make_toeplitz


LEARN_PROX_PER_LAYER = 'per-layer'
LEARN_PROX_GLOBAL = 'global'
LEARN_PROX_FALSE = 'none'
ALL_LEARN_PROX = [LEARN_PROX_FALSE, LEARN_PROX_GLOBAL, LEARN_PROX_PER_LAYER]


class LpgdTautStringHRF(_ListaAnalysis):
    __doc__ = DOC_LISTA.format(
        type='learned-PGD with taut-string for prox operator',
        problem_name='TV',
        descr='unconstrained parametrization'
    )

    def __init__(self, h, n_times_valid, n_layers, learn_th=False,
                 use_moreau=False, max_iter=100, net_solver_type="recursive",
                 initial_parameters=None, name="LPGD - Taut-string",
                 verbose=0, device=None):
        if device is not None and 'cuda' in device:
            warnings.warn("Cannot use LpgdTautString on cuda device. "
                          "Falling back to CPU.")
            device = 'cpu'

        self.use_moreau = use_moreau

        self.h = np.array(h)
        self.h_ = check_tensor(h, device=device)
        self.l_ = lipsch_cst_from_kernel(h, n_times_valid)

        self.A = make_toeplitz(h, n_times_valid).T
        self.A_ = check_tensor(self.A, device=device)
        self.inv_A_ = torch.pinverse(self.A_)
        self.D = (np.eye(n_times_valid, k=-1)
                  - np.eye(n_times_valid, k=0))[:, :-1]

        super().__init__(n_layers=n_layers, learn_th=learn_th,
                         max_iter=max_iter, net_solver_type=net_solver_type,
                         initial_parameters=initial_parameters, name=name,
                         verbose=verbose, device=device)

    def get_initial_layer_parameters(self, layer_id):
        layer_params = dict()
        layer_params['hth'] = np.convolve(self.h[::-1], self.h) / self.l_
        layer_params['h'] = self.h / self.l_

        if self.learn_th:
            layer_params['threshold'] = np.array(1.0 / self.l_)
        return layer_params

    def forward(self, x, lbda, output_layer=None):
        """ Forward pass of the network. """
        output_layer = self.check_output_layer(output_layer)

        # initialized variables
        _, u, _ = init_vuz(self.A, self.D, x, lbda, inv_A=self.inv_A_,
                           device=self.device)

        for layer_id in range(output_layer):
            layer_params = self.parameter_groups[f'layer-{layer_id}']
            # retrieve parameters
            h = layer_params['h']
            hth = layer_params['hth']
            mul_lbda = layer_params.get('threshold', 1.0 / self.l_)
            mul_lbda = check_tensor(mul_lbda, device=self.device)

            # apply one 'iteration'
            u = hth_id_u_tensor(hth, u) + htx_tensor(h, x)
            u = ProxTV_l1.apply(u, lbda * mul_lbda)

        return u

    def _loss_fn(self, x, lbda, u):
        r"""Loss function for the primal.

            :math:`L(u) = 1/2 ||x - Au||_2^2 - lbda ||D u||_1`
        """
        n_samples = x.shape[0]
        residual = hu_tensor(self.h_, u) - x
        loss = 0.5 * (residual * residual).sum()
        if self.use_moreau:
            loss = RegTV.apply(loss, u, lbda)
        else:
            loss = loss + lbda * torch.abs(u[:, 1:] - u[:, :-1]).sum()
        return loss / n_samples
