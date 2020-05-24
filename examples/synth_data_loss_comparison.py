""" Convergence rate between iterative-step-z and learn-step-z for TA
decomposition
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import argparse
import matplotlib.pyplot as plt
from pyta import TA
from pyta.data import little_brain
from pyta.hrf_model import double_gamma_hrf
from pyta.utils import compute_lbda_max, check_random_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=300,
                        help='Max number of iterations to for the z-step.')
    parser.add_argument('--max-training-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks for the z-step.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--plots-dir', type=str, default='outputs',
                        help='Outputs directory for plots')
    args = parser.parse_args()

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    rng = check_random_state(args.seed)
    print(f'Seed used = {args.seed}')  # noqa: E999

    ###########################################################################
    # Synthetic data generation
    t_r = 1.0
    nx, ny, nz, N_orig = 5, 5, 5, 200
    snr = 1.0
    h = double_gamma_hrf(t_r, 30)
    params = dict(tr=t_r, nx=nx, ny=ny, nz=nz, N=N_orig, snr=snr, h=h,
                  seed=rng)
    y, _, _, _, _ = little_brain(**params)
    y_train, y_test = y[..., int(N_orig/2):], y[..., :int(N_orig/2)]

    ###########################################################################
    # Main experimentation
    lbda_t = 5e-1  # 0.5 * compute_lbda_max(y, h)
    lbda_s = 5e-3

    params = dict(t_r=t_r, h=h, max_iter=args.max_iter,
                  max_iter_z=args.max_iter_z, solver_type='iterative-z-step',
                  verbose=1)
    ta_iterative = TA(**params)

    params = dict(t_r=t_r, h=h, max_iter=args.max_iter,
                  max_iter_z=args.max_iter_z, n_inner_layers=300,
                  net_solver_type='recursive',
                  max_iter_training_net=args.max_training_iter,
                  solver_type='learn-z-step', verbose=1)
    ta_learn = TA(**params)

    _, _, _ = ta_iterative.transform(y, lbda_t, lbda_s)

    ta_learn.fit(y_train, lbda_t)
    _, _, _ = ta_learn.transform(y_test, lbda_t, lbda_s)

    ###########################################################################
    # Plotting
    params = dict(t_r=t_r, h=h, max_iter=200, max_iter_z=5000,
                  solver_type='iterative-z-step', verbose=1)
    ta_ref = TA(**params)
    _, _, _ = ta_ref.transform(y_test, lbda_t, lbda_s)
    min_loss = ta_ref.l_loss[-1]
    eps = 1.0e-20
    lw = 4

    plt.figure(f"[{__file__}] Loss functions", figsize=(10, 5))
    plt.loglog(list(range(1, args.max_iter + 1)),
               ta_iterative.l_loss - min_loss, lw=lw, label='iterative')
    plt.loglog(list(range(1, args.max_iter + 1)), ta_learn.l_loss - min_loss,
               lw=lw, label='learn')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
               borderaxespad=0.0, fontsize=15)
    plt.grid()
    plt.xlabel("Layers [-]", fontsize=15)
    plt.ylabel('$F(.) - F(z^*)$', fontsize=15)
    title_ = f'Loss function comparison on UKBB subject'
    plt.title(title_, fontsize=15)

    plt.tight_layout()

    filename = os.path.join(args.plots_dir, "loss_comparison.pdf")
    plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))
