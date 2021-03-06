""" pyTA."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import warnings
import torch
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin
from prox_tv import tvgen, tv1_1d
from carpet.checks import check_tensor
from carpet.utils import init_vuz
from .lista_analysis import LpgdTautStringHRF
from .optim import fista, fbs
from .loss_and_grad import _grad_t_analysis, _obj_t_analysis
from .utils import lipsch_cst_from_kernel


class TA(TransformerMixin):
    """ Total Activation transformer

    implements the method proposed in

    'Total activation: fMRI deconvolution through spatio-temporal
     regularization'

    Fikret Isik Karahanoglu, Cesar Caballero-Gaudes, François Lazeyras,
    Dimitri Van De Ville

    Parameters
    ----------
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    h : array (hrf_n_time_frames,), HRF
    n_times_valid : int, neural signal activity length
    solver_type : str, (default='fista-z-step'), solder type for the
        z-step, valid option are ['ista-z-step', 'fista-z-step' 'learn-z-step']
    update_weights : tuple of two float, weights to balance the constrained
        between the spatial and temporal update
    max_iter : int, number of iterations for the main loop
    n_jobs : int, number of CPU to use for the temporal regularization if the
        iterative alorithm is used
    name : str, id name of the TA instance
    device : str, (default='cpu'), Cuda device to use the network for the
        z-step when solver_type is 'learn-z-step'

    net_solver_training_type : str (default='recursive'), define the method of
        optimization for the z-step when solver_type is 'learn-z-step'
    max_iter_training_net : int, (default=200), number of iteration to train
        the network for the z-step when solver_type is 'learn-z-step'
    max_iter_z : int, (default=40), number of iterations/layers for the z-step
    inner_max_iter_net : int, (default=50) number of layers for the inner prox
        net of the net solver
    verbose : int, (default=1), verbosity level
    """
    def __init__(self, t_r, h, n_times_valid, solver_type='fista-z-step',
                 update_weights=[0.5, 0.5], max_iter=50, n_jobs=1, name='TA',
                 device='cpu', init_net_parameters=None,
                 inner_max_iter_net=50, net_solver_training_type='recursive',
                 max_iter_training_net=100, max_iter_z=40, verbose=1):

        # model parameters
        self.t_r = t_r
        self.h = h

        # usefull dimension
        self.n_times_valid = n_times_valid
        self.n_times = n_times_valid + len(self.h) - 1

        # network parameters
        self.device = device
        self.max_iter_training_net = max_iter_training_net
        self.inner_max_iter_net = inner_max_iter_net
        self.max_iter_z = max_iter_z
        self.net_solver_training_type = net_solver_training_type
        self.init_net_parameters = init_net_parameters

        # solver parameters
        self.solver_type = solver_type
        kwargs = dict(h=self.h, n_times_valid=self.n_times_valid,
                      n_layers=self.max_iter_z, learn_th=False,
                      max_iter=self.max_iter_training_net,
                      net_solver_type=self.net_solver_training_type,
                      initial_parameters=self.init_net_parameters,
                      name="Temporal prox", verbose=1, device=self.device)
        self.net_solver = LpgdTautStringHRF(**kwargs)

        # optimization parameter
        self.update_weights = update_weights
        self.max_iter = max_iter
        self.name = name
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, y, lbda_t):
        """ Fit the neural net for the prox_t step.

        Parameters
        ----------
        Y : array (n_samples, n_time_frames), observed BOLD signal
        """
        dim_x, dim_y, dim_z, n_times = y.shape
        y_ravel = y.reshape(dim_x * dim_y * dim_z, n_times)

        if self.solver_type == 'learn-z-step':
            self.net_solver.fit(y_ravel, lbda=lbda_t)
        else:
            warnings.warn("Fitting is not necessary if learn_temporal_prox is "
                          "set to False")
        return self

    def transform(self, y, lbda_t, lbda_s):
        """ Estimate the neural activation signal from the observed BOLD signal

        Parameters
        ----------
        y : array (n_samples, n_time_frames), observed BOLD signal

        Return
        ------
        X : array (n_samples, n_time_frames), denoised BOLD signal
        U : array (n_samples, n_time_frames - hrf_n_time_frames + 1),
            block signal
        Z : array (n_samples, n_time_frames - hrf_n_time_frames + 1),
            source signal
        """
        def _prox_t(x):
            x, _, _ = self.prox_t(x, lbda_t)
            return x

        def _prox_s(x):
            return self.prox_s(x, lbda_s)

        x, l_time, l_loss = fbs(
            y, _prox_t, _prox_s, update_weights=self.update_weights,
            max_iter=self.max_iter, name=self.name,
            verbose=self.verbose)

        self.l_time = np.cumsum(np.array(l_time))
        self.l_loss = np.array(l_loss)

        x, u, z = self.prox_t(x, lbda_t)  # XXX forget last point in l_loss

        return x, u, z

    def prox_s(self, y, lbda_s, reshape_4d=True):
        """ The spatial proximal operator for full brain. """
        dim_x, dim_y, dim_z, n_times = y.shape

        def _prox_s(x, reg):
            return tvgen(x, [reg, reg, reg], [1, 2, 3], [1, 1, 1])

        x = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_prox_s)(y[..., i], lbda_s) for i in range(n_times))
        x = np.vstack([x_.flatten() for x_ in x]).T

        if reshape_4d:
            return x.reshape(dim_x, dim_y, dim_z, n_times)
        return x

    def prox_t(self, y, lbda_t, reshape_4d=True):
        """ The temporal proximal operator for full brain. """
        dim_x, dim_y, dim_z, n_times = y.shape
        n_samples = dim_x * dim_y * dim_z
        y_ravel = y.reshape(n_samples, n_times)

        if self.solver_type == 'learn-z-step':

            if hasattr(self.net_solver, 'training_loss_'):
                warnings.warn("Learn-prox-TV seems to not have been trained, "
                              "call the 'fit' method to that purpose")

            self.l_loss_prox_t = self._compute_loss(y_ravel, lbda_t)
            u = self.net_solver.transform(y_ravel, lbda=lbda_t)
            z = np.diff(u, axis=1)
            x = u.dot(self.net_solver.A)

        elif self.solver_type in ['ista-z-step', 'fista-z-step']:

            momentum = self.solver_type == 'fista-z-step'

            hth = np.convolve(self.h[::-1], self.h)
            htY = np.r_[[np.convolve(self.h[::-1], y_, mode='valid')
                         for y_ in y_ravel]]

            _, u0 = self._get_init(y_ravel, lbda_t, force_numpy=True)

            lipsch_cst = lipsch_cst_from_kernel(self.h, self.n_times_valid)
            step_size = 1.0 / lipsch_cst

            def _grad(x):
                return _grad_t_analysis(x, hth, htY=htY)

            def _obj(x):
                return _obj_t_analysis(x, y_ravel, self.h, lbda_t)

            def _prox(x, step_size):
                return np.array([tv1_1d(x_, lbda_t * step_size) for x_ in x])

            kwargs = dict(x0=u0, grad=_grad, obj=_obj, prox=_prox,
                step_size=step_size, name='_prox_t',  # noqa: E128
                max_iter=self.max_iter_z, debug=True,  # noqa: E128
                momentum=momentum, verbose=0)  # noqa: E128
            u, self.l_loss_prox_t = fista(**kwargs)
            z = np.diff(u, axis=1)
            x = np.r_[[np.convolve(self.h, u[i, :])
                       for i in range(u.shape[0])]]

        else:
            raise ValueError(f"solver_type should belong to [learn-z-step, "
                             f"ista-z-step, fista-z-step]"
                             f", got {self.solver_type}")

        if reshape_4d:  # XXX seems to mess up the order
            x = x.reshape(dim_x, dim_y, dim_z, self.n_times)
            u = u.reshape(dim_x, dim_y, dim_z, self.n_times_valid)
            z = z.reshape(dim_x, dim_y, dim_z, self.n_times_valid - 1)
            return x, u, z
        return x, u, z

    def fit_transform(self, y, lbda_t, lbda_s):
        """ Fit and transform given observed data. """
        if self.solver_type == 'learn-z-step':
            self.fit(y, lbda_t)
        return self.transform(y, lbda_t, lbda_s)

    def _get_init(self, x, lbda, force_numpy=False):
        x0, u0, _ = init_vuz(self.net_solver.A,
                             self.net_solver.D, x, lbda)
        if force_numpy:
            return np.array(x0), np.array(u0)
        else:
            return x0, u0

    def _compute_loss(self, x, lbda):
        """ Return the loss evolution in the forward pass of x. """
        _, u0 = self._get_init(x, lbda)
        x_ = check_tensor(x, device=self.net_solver.device)
        l_loss = [self.net_solver._loss_fn(x_, lbda, u0)]
        with torch.no_grad():
            for n_layers in range(self.net_solver.n_layers):
                u = self.net_solver(x_, lbda, output_layer=n_layers+1)
                loss_ = self.net_solver._loss_fn(x_, lbda, u)
                l_loss.append(loss_.cpu().numpy())
        return np.array(l_loss)
