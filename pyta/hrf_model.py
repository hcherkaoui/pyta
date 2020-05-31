""" This module gathers HRF models."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import numpy as np
from scipy.special import gammaln, xlogy


# double gamma HRF model constants
DT = 0.001
P_DELAY = 6.0
UNDERSHOOT = 16.0
P_DISP = 1.0
U_DISP = 1.0
P_U_RATIO = 0.167

# usefull precomputed HRF peak constants
LOC_PEAK = DT / P_DISP
A_1_PEAK = P_DELAY / P_DISP - 1
GAMMA_LN_A_PEAK = gammaln(P_DELAY / 1.0)

# usefull precomputed HRF undershoot constants
LOC_U = DT / U_DISP
A_1_U = UNDERSHOOT / U_DISP - 1
GAMMA_LN_A_U = gammaln(UNDERSHOOT / 1.0)


def _gamma_pdf_hrf_peak(x):
    """ Precomputed gamma pdf for HRF peak (double gamma HRF model).

    Parameters
    ----------
    x : float, quantiles

    Return
    ------
    p : float, probability density function evaluated at x
    """
    x = np.copy(x)
    x -= LOC_PEAK
    support = x > 0.0
    x_valid = x[support]
    p = np.zeros_like(x)
    p[support] = np.exp(xlogy(A_1_PEAK, x_valid) - x_valid - GAMMA_LN_A_PEAK)
    return p


def _gamma_pdf_hrf_undershoot(x):
    """ Precomputed gamma pdf for HRF undershoot (double gamma HRF model).

    Parameters
    ----------
    x : float, quantiles

    Return
    ------
    p : float, probability density function evaluated at x
    """
    x = np.copy(x)
    x -= LOC_U
    support = x > 0.0
    x_valid = x[support]
    p = np.zeros_like(x)
    p[support] = np.exp(xlogy(A_1_U, x_valid) - x_valid - GAMMA_LN_A_U)
    return p


def _double_gamma_hrf(delta, t_r=1.0, dur=60.0, onset=0.0):
    """ Double Gamma HRF model.

    From Nistats package
   https://github.com/nistats/nistats/blob/master/nistats/hemodynamic_models.py

    Parameters
    ----------
    delta : float, temporal dilation to pilot the HRF inflation
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    dur : float, (default=60.0), the time duration on which to represent the
        HRF
    onset : float, (default=0.0), onset of the HRF

    Return
    ------
    hrf : array, shape (dur / t_r, ), HRF
    """
    # dur: the (continious) time segment on which we represent all
    # the HRF. Can cut the HRF too early. The time scale is second.
    t = np.linspace(0, dur, int(float(dur) / DT)) - float(onset) / DT
    t = t[::int(t_r/DT)]

    peak = _gamma_pdf_hrf_peak(delta * t)
    undershoot = _gamma_pdf_hrf_undershoot(delta * t)
    hrf = peak - P_U_RATIO * undershoot

    return hrf, t


def check_len_hrf(h, n_times_atom):
    """ Check that the HRF has the proper length.

    Parameters
    ----------
    h : array, shape (n_times_atom, ), HRF
    n_times_atom : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    h : array, shape (n_times_atom, ), HRF with a correct length
    """
    n = n_times_atom - len(h)
    if n < 0:
        h = h[:n]
    elif n > 0:
        h = np.hstack([h, np.zeros(n)])
    return h


def double_gamma_hrf(t_r, n_times_atom=60):
    """ Double gamma HRF.

    Parameters
    ----------
    t_r : float, Time of Repetition, fMRI acquisition parameter, the temporal
        resolution
    n_times_atom : int, number of components on which to decompose the neural
        activity (number of temporal components and its associated spatial
        maps).

    Return
    ------
    hrf : array, shape (n_times_atom, ), HRF
    """
    _hrf = _double_gamma_hrf(delta=1.0, t_r=t_r, dur=n_times_atom * t_r)[0]
    return check_len_hrf(_hrf, n_times_atom)
