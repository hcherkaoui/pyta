"""Convergence rate between iterative-step-z and learn-step-z algorithm for
TA decomposition.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import shutil
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from carpet.utils import init_vuz
from pyta import TA
from pyta.data import little_brain
from pyta.hrf_model import double_gamma_hrf, make_toeplitz
from pyta.utils import check_random_state, compute_lbda_max
from pyta.loss_and_grad import _obj_t_analysis


def logspace_layers(n_layers=10, max_depth=50):
    """ Return n_layers, from 1 to max_depth of different number of layers to
    define networks """
    all_n_layers = np.logspace(0, np.log10(max_depth), n_layers).astype(int)
    return list(np.unique(all_n_layers))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=20,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=100,
                        help='Max number of iterations for the z-step.')
    parser.add_argument('--load-net', type=str, default=None, nargs='+',
                        help='Load pretrained network parameters.')
    parser.add_argument('--max-training-iter', type=int, default=1000,
                        help='Max number of iterations to train the '
                        'learnable networks for the z-step.')
    parser.add_argument('--n-time-frames', type=int, default=100,
                        help='Number of timeframes to retain from the the '
                        'data fMRI.')
    parser.add_argument('--plots-dir', type=str, default='outputs',
                        help='Outputs directory for plots')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    args = parser.parse_args()

    print(__doc__)
    print('*' * 80)

    t0_global = time.time()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    print("archiving '{0}' under '{1}'".format(__file__, args.plots_dir))
    shutil.copyfile(__file__, os.path.join(args.plots_dir, __file__))

    rng = check_random_state(args.seed)
    print(f'Seed used = {args.seed}')  # noqa: E999

    ###########################################################################
    # Real data loading
    t_r = 1.0
    hrf_time_frames = 30
    nx = ny = nz = 10
    n_times_valid = 2 * args.n_time_frames - hrf_time_frames + 1
    h = double_gamma_hrf(t_r, hrf_time_frames)
    params = dict(tr=t_r, nx=nx, ny=ny, nz=nz, N=n_times_valid, h=h, seed=rng)
    y, _, _, _, _ = little_brain(**params)

    n_times_valid = args.n_time_frames - hrf_time_frames + 1
    D = (np.eye(n_times_valid, k=-1) - np.eye(n_times_valid, k=0))[:, :-1]
    H = make_toeplitz(h, n_times_valid).T

    y_train = y[..., slice(args.n_time_frames, 2 * args.n_time_frames)]
    y_test = y[..., slice(0, args.n_time_frames)]

    print(f"Shape of the train-set : {y_train.shape}")
    print(f"Shape of the test-set : {y_test.shape}")

    ###########################################################################
    # Main experimentation
    lbda = 0.1 * compute_lbda_max(H, y_train)
    all_layers = logspace_layers(n_layers=10, max_depth=args.max_iter_z)

    params = dict(t_r=t_r, H=H, name='Iterative-z',
                  max_iter_z=10*args.max_iter_z,
                  solver_type='iterative-z-step', verbose=1)
    ta_iter = TA(**params)

    t0 = time.time()
    _, _, _ = ta_iter.prox_t(y_test, lbda)
    print(f"ta_iterative.prox_t finished : {time.time() - t0:.2f}s")

    y_train_ravel = y_train.reshape(nx * ny * nz, args.n_time_frames)
    y_test_ravel = y_test.reshape(nx * ny * nz, args.n_time_frames)
    _, u0, _ = init_vuz(H, D, y_train_ravel, lbda)
    loss_ta_learn = [_obj_t_analysis(u0, y_test_ravel, H, lbda)]

    for i, n_layers in enumerate(all_layers):
        params = dict(t_r=t_r, H=H, max_iter_z=n_layers,
                      net_solver_type='recursive', name='Learned-z',
                      max_iter_training_net=args.max_training_iter,
                      solver_type='learn-z-step', verbose=1)

        if args.load_net is not None:
            filename = args.load_net[i]
            with open(filename, 'rb') as pfile:
                init_params = pickle.load(pfile)
            params['init_net_parameters'] = init_params
            print(f"Loading parameters from '{filename}'")
            ta_learn = TA(**params)

        else:
            ta_learn = TA(**params)
            ta_learn.fit(y_train, lbda)

        fitted_params = ta_learn.pretrained_network.export_parameters()
        filename = f'fitted_params_n_layers_{n_layers}.pkl'
        filename = os.path.join(args.plots_dir, filename)
        with open(filename, 'wb') as pfile:
            pickle.dump(fitted_params, pfile)
        print(f"Saving fitted parameters under '{filename}'")

        t0 = time.time()
        _, u, _ = ta_learn.prox_t(y_test, lbda)
        print(f"ta_learn.prox_t finished : {time.time() - t0:.2f}s")

        u = np.array(u)
        loss_ta_learn.append(_obj_t_analysis(u, y_test, H, lbda))
    loss_ta_learn = np.array(loss_ta_learn)

    ###########################################################################
    # Plotting
    params = dict(t_r=t_r, H=H, max_iter_z=1000, name='Ref-z',
                  solver_type='iterative-z-step', verbose=0)
    ta_ref = TA(**params)

    t0 = time.time()
    _, _, _ = ta_ref.prox_t(y_test, lbda)
    print(f"ta_ref.prox_t finished : {time.time() - t0:.2f}s")

    min_loss = ta_ref.l_loss_prox_t[-1]
    eps = 1.0e-10
    lw = 6

    loss_ta_iter = ta_iter.l_loss_prox_t - min_loss
    loss_ta_learn = loss_ta_learn - min_loss

    plt.figure(f"[{__file__}] Loss functions", figsize=(6, 3))
    plt.semilogy(loss_ta_iter, lw=lw, label='iterative')
    plt.semilogy(loss_ta_learn, lw=lw, label='learn')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
               borderaxespad=0.0, fontsize=16)
    plt.grid()
    plt.xlabel("Iters [-]", fontsize=16)
    plt.ylabel('$F(.) - F(u^*)$', fontsize=16)
    title_ = f'Loss function comparison'
    plt.title(title_, fontsize=18)

    plt.tight_layout()

    filename = os.path.join(args.plots_dir, "loss_comparison.pdf")
    plt.savefig(filename, dpi=300)

    delta_t = time.time() - t0_global
    delta_t = time.strftime("%H h %M min %S s", time.gmtime(delta_t))
    print("Script runs in: {}".format(delta_t))

    plt.show()
