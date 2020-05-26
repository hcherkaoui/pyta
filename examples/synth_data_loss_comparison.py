"""Convergence rate between iterative-step-z and learn-step-z algorithm for
TA decomposition.
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import datasets, image
from pyta import TA
from pyta.data import little_brain
from pyta.hrf_model import double_gamma_hrf
from pyta.utils import check_random_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=20,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=100,
                        help='Max number of iterations for the z-step.')
    parser.add_argument('--max-training-iter', type=int, default=1000,
                        help='Max number of iterations to train the '
                        'learnable networks for the z-step.')
    parser.add_argument('--n-time-frames', type=int, default=50,
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

    rng = check_random_state(args.seed)
    print(f'Seed used = {args.seed}')  # noqa: E999

    ###########################################################################
    # Real data loading
    t_r = 1.0
    hrf_time_frames = 20
    nx, ny, nz, N_orig = 3, 3, 3, args.n_time_frames - hrf_time_frames + 1
    snr = 1.0
    h = double_gamma_hrf(t_r, hrf_time_frames)
    params = dict(tr=t_r, nx=nx, ny=ny, nz=nz, N=N_orig, snr=snr, h=h,
                  seed=rng)
    y, _, _, _, _ = little_brain(**params)

    y_train = y[..., int(args.n_time_frames/2):]
    y_test = y[..., :int(args.n_time_frames/2)]

    print(f"Shape of y-train : {y_train.shape}")
    print(f"Shape of y-test : {y_test.shape}")

    ###########################################################################
    # Main experimentation
    lbda_t = 0.1
    lbda_s = 0.1

    params = dict(t_r=t_r, h=h, max_iter=args.max_iter, name='Iterative-z',
                  max_iter_z=args.max_iter_z, solver_type='iterative-z-step',
                  verbose=1)
    ta_iterative = TA(**params)

    params = dict(t_r=t_r, h=h, max_iter=args.max_iter,
                  max_iter_z=args.max_iter_z,
                  net_solver_type='recursive', name='Learned-z',
                  max_iter_training_net=args.max_training_iter,
                  solver_type='learn-z-step', verbose=1)
    ta_learn = TA(**params)

    t0 = time.time()
    _, _, _ = ta_iterative.transform(y_test, lbda_t, lbda_s)
    print(f"ta_iterative.transform finished : {time.time() - t0:.2f}s")

    ta_learn.fit(y_train, lbda_t)
    t0 = time.time()
    _, _, _ = ta_learn.transform(y_test, lbda_t, lbda_s)
    print(f"ta_learn.transform finished : {time.time() - t0:.2f}s")

    ###########################################################################
    # Plotting
    params = dict(t_r=t_r, h=h, max_iter=20, max_iter_z=100, name='Ref-z',
                  solver_type='iterative-z-step', verbose=1)
    ta_ref = TA(**params)
    t0 = time.time()
    _, _, _ = ta_ref.transform(y_test, lbda_t, lbda_s)
    print(f"ta_ref.transform finished : {time.time() - t0:.2f}s")

    min_loss = ta_ref.l_loss[-1]
    eps = 1.0e-20
    lw = 5

    plt.figure(f"[{__file__}] Loss functions", figsize=(8, 4))
    plt.semilogy(ta_iterative.l_time,
                 ta_iterative.l_loss - min_loss, lw=lw, label='iterative')
    plt.semilogy(ta_learn.l_time, ta_learn.l_loss - min_loss,
                 ls='--', lw=lw, label='learn')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
               borderaxespad=0.0, fontsize=16)
    plt.grid()
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel('$F(.) - F(x^*)$', fontsize=16)
    title_ = f'Loss function comparison on UKBB subjects'
    plt.title(title_, fontsize=18)

    plt.tight_layout()

    filename = os.path.join(args.plots_dir, "loss_comparison.pdf")
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s",
        time.gmtime(time.time() - t0_global))
    print("Script runs in: {}".format(delta_t))

    plt.show()