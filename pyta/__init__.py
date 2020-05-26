""" pyTA."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import warnings
import numpy as np
from numpy.fft import fftn, ifftn
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin
from prox_tv import tvgen, tv1_1d
from carpet.lista_analysis import LpgdTautString
from .hrf_model import double_gamma_hrf, make_toeplitz
from .optim import fista, fbs
from .loss_and_grad import _grad_t_analysis, _obj_t_analysis


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
    h : array (hrf_n_time_frames,), (default=None), HRF
    len_h : int, (default=None), desired length of the HRF
    solver_type : str, (default='iterative-z-step'), solder type for the
        z-step, valid option are ['iterative-z-step', 'learn-z-step']
    update_weights : tuple of two float, weights to balance the constrained
        between the spatial and temporal update
    max_iter : int, number of iterations for the main loop
    n_jobs : int, number of CPU to use for the temporal regularization if the
        iterative alorithm is used
    name : str, id name of the TA instance
    device : str, (default='cpu'), Cuda device to use the network for the
        z-step when solver_type is 'learn-z-step'
    net_solver_type : str (default='recursive'), define the method of
        optimization for the z-step when solver_type is 'learn-z-step'
    max_iter_training_net : int, (default=200), number of iteration to train
        the network for the z-step when solver_type is 'learn-z-step'
    max_iter_z : int, (default=40), number of iterations/layers for the z-step
    verbose : int, (default=1), verbosity level
    """
    def __init__(self, t_r, h=None, len_h=None, solver_type='iterative-z-step',
                 update_weights=[0.5, 0.5], max_iter=50, n_jobs=1, name='TA',
                 device='cpu', net_solver_type='recursive',
                 max_iter_training_net=100, max_iter_z=40, verbose=1):

        # model parameters
        self.t_r = t_r
        if (h is not None) and (len_h is not None):
            raise ValueError("Either provide the HRF or its dimension")
        if h is None:
            h = double_gamma_hrf(t_r, len_h)
        self.h = h

        # network parameters
        self.device = device
        self.max_iter_training_net = max_iter_training_net
        self.max_iter_z = max_iter_z
        self.net_solver_type = net_solver_type

        # solver parameters
        if solver_type in ['learn-z-step', 'iterative-z-step']:
            self.solver_type = solver_type
        else:
            raise ValueError(f"solver_type should belong to ['learn-z-step',"
                             f" 'iterative-z-step'], got {solver_type}")

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
        if self.solver_type == 'learn-z-step':
            dim_x, dim_y, dim_z, n_times = y.shape
            n_samples = dim_x * dim_y * dim_z
            y_ravel = y.reshape(n_samples, n_times)
            n_times_valid = y_ravel.shape[1] - len(self.h) + 1

            self.H = make_toeplitz(self.h, n_times_valid)

            kwargs = dict(A=self.H.T, n_layers=self.max_iter_z,
                          initial_parameters=None, learn_th=False,
                          max_iter=self.max_iter_training_net,
                          net_solver_type=self.net_solver_type,
                          name="Temporal prox", verbose=1,
                          device=self.device)

            self.pretrained_network = LpgdTautString(**kwargs)
            self.pretrained_network.fit(y_ravel, lbda=lbda_t)

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
        # to compute spatial regularization
        dim_x, dim_y, dim_z, n_times = y.shape
        vol_shape = (dim_x, dim_y, dim_z)

        l1 = l3 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        l2 = [[0, 1, 0], [1, -6, 1], [0, 1, 0]]
        fft_D = fftn(np.array([l1, l2, l3]), vol_shape)

        n_times_valid = n_times - len(self.h) + 1
        H = make_toeplitz(self.h, n_times_valid).T
        pinv_H = np.linalg.pinv(H)

        def _prox_t(x):
            x, _, _ = self.prox_t(x, lbda_t)
            return x

        def _prox_s(x):
            return self.prox_s(x, lbda_s)

        def _reg_t(x):
            # XXX might not be equivalent to the 'real' temporal regularization
            x = x.reshape(dim_x * dim_y * dim_z, n_times)
            u = x.dot(pinv_H)  # XXX
            z = np.diff(u, axis=1)
            return np.sum(np.abs(z))

        def _reg_s(x):
            # XXX might not be equivalent to the'real' spatial regularization
            Dx = ifftn(fft_D * fftn(x, vol_shape)).real
            return np.sum(np.abs(Dx))

        def _obj(x):
            # XXX might not be equivalent to the 'real' loss function
            res = (x - y).ravel()
            reg_t = _reg_t(x)
            reg_s = _reg_s(x)
            return 0.5 * res.dot(res) + lbda_t * reg_t + lbda_s * reg_s

        x, l_time, l_loss = fbs(
            y, _prox_t, _prox_s, update_weights=self.update_weights,
            max_iter=self.max_iter, name=self.name, obj=_obj,
            verbose=self.verbose)

        self.l_time = np.cumsum(np.array(l_time))
        self.l_loss = np.array(l_loss)

        x, u, z = self.prox_t(x, lbda_t)  # XXX forget last point in l_loss

        return x, u, z

    def prox_s(self, y, lbda_s):
        """ The spatial proximal operator for full brain. """
        dim_x, dim_y, dim_z, n_times = y.shape

        def _prox_s(x, reg):
            return tvgen(x, [reg, reg, reg], [1, 2, 3], [1, 1, 1])

        x = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_prox_s)(y[..., i], lbda_s) for i in range(n_times))
        x = np.vstack([x_.flatten() for x_ in x]).T

        return x.reshape(dim_x, dim_y, dim_z, n_times)

    def prox_t(self, y, lbda_t):
        """ The temporal proximal operator for full brain. """
        dim_x, dim_y, dim_z, n_times = y.shape
        n_samples = dim_x * dim_y * dim_z
        y_ravel = y.reshape(n_samples, n_times)

        if self.solver_type == 'learn-z-step':
            u = self.pretrained_network.transform(y_ravel, lbda=lbda_t)

            n_times_valid = y_ravel.shape[1] - len(self.h) + 1
            H = make_toeplitz(self.h, n_times_valid).T

            z = np.diff(u, axis=1)
            x = u.dot(H)

        elif self.solver_type == 'iterative-z-step':
            n_times_valid = y_ravel.shape[1] - len(self.h) + 1
            H = make_toeplitz(self.h, n_times_valid).T

            HtH = H.dot(H.T)
            Hty = y_ravel.dot(H.T)

            x0 = np.zeros((n_samples, n_times_valid))

            step_size = 1.0 / np.linalg.norm(HtH, ord=2) ** 2

            def _grad(x):
                return _grad_t_analysis(x, HtH, Hty=Hty)

            def _obj(x):
                return _obj_t_analysis(x, y_ravel, H, lbda_t)

            def _prox(x, step_size):
                return np.array([tv1_1d(x_, lbda_t * step_size) for x_ in x])

            kwargs = dict(x0=x0, grad=_grad, obj=_obj, prox=_prox,
                step_size=step_size, name='_prox_t',  # noqa: E128
                max_iter=self.max_iter_z, verbose=0)  # noqa: E128
            u = fista(**kwargs)
            z = np.diff(u, axis=1)
            x = u.dot(H)

        else:
            raise ValueError(f"solver_type should belong to [synthesis, "
                             f"analysis, pretrained-net]"
                             f", got {self.solver_type}")

        x = x.reshape(dim_x, dim_y, dim_z, n_times)
        u = u.reshape(dim_x, dim_y, dim_z, n_times_valid)
        z = z.reshape(dim_x, dim_y, dim_z, n_times_valid - 1)

        return x, u, z

    def fit_transform(self, y, lbda_t, lbda_s):
        """ Fit and transform given observed data. """
        if self.solver_type == 'learn-z-step':
            self.fit(y, lbda_t)
        return self.transform(y, lbda_t, lbda_s)
