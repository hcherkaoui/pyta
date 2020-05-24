""" Example of the TA decomposition on the synthetic data experimentation from
the paper:

'Total activation: fMRI deconvolution through spatio-temporal regularization'

Fikret Isik Karahanoglu, Cesar Caballero-Gaudes, François Lazeyras,
Dimitri Van De Ville
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
from pyta.utils import check_random_state


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=300,
                        help='Max number of iterations for the z-step.')
    parser.add_argument('--max-training-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks for the z-step.')
    parser.add_argument('--solver-type', type=str, default='iterative-z-step',
                        help="Solver type for the z-step, possible choice are"
                        " ['iterative-z-step', 'learn-z-step'].")
    parser.add_argument('--plots-dir', type=str, default='outputs',
                        help='Outputs directory for plots')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
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
    nx, ny, nz, N_orig = 5, 5, 5, 100
    snr = 1.0
    h = double_gamma_hrf(t_r, 30)
    params = dict(tr=t_r, nx=nx, ny=ny, nz=nz, N=N_orig, snr=snr, h=h,
                  seed=rng)
    y, x, u, _, _ = little_brain(**params)

    ###########################################################################
    # Main experimentation
    lbda_t = 5e-1
    lbda_s = 0.005

    params = dict(
        t_r=t_r, h=h, update_weights=[0.5, 0.5], max_iter=args.max_iter,
        max_iter_z=args.max_iter_z, n_inner_layers=150,
        max_iter_training_net=args.max_training_iter,
        net_solver_type='recursive', solver_type=args.solver_type,
        verbose=True,
        )
    ta = TA(**params)

    est_x, est_u, _ = ta.fit_transform(y, lbda_t, lbda_s)

    ###########################################################################
    # Plotting
    nx, ny, nz, N = est_x.shape

    voxels_of_interest = [
                (int(0.15*nx), int(0.5*ny), int(0.5*nz), slice(None)),  # c1
                (int(0.65*nx), int(0.15*ny), int(0.5*nz), slice(None)),  # c2
                (int(0.65*nx), int(0.65*ny), int(0.25*nz), slice(None)),  # c3
                (int(0.65*nx), int(0.65*ny), int(0.75*nz), slice(None)),  # c4
                (int(0.3*nx), int(0.15*ny), int(0.5*nz), slice(None)),  # b12
                (int(0.65*nx), int(0.3*ny), int(0.25*nz), slice(None)),  # b23
                (int(0.65*nx), int(0.3*ny), int(0.75*nz), slice(None)),  # b24
                (int(0.65*nx), int(0.65*ny), int(0.5*nz), slice(None)),  # b34
                (int(0.3*nx), int(0.3*ny), nz-1, slice(None)),  # i124
                (int(0.3*nx), int(0.3*ny), int(0.5*nz), slice(None)),  # i1234
                (int(0.3*nx), int(0.3*ny), 0, slice(None)),  # i123
                (0, 0, 0, slice(None)),  # f1
                (nx-1, ny-1, nz-1, slice(None)),  # f4
                            ]
    name_of_interest = ["center region 1",
                        "center region 2",
                        "center region 3",
                        "center region 4",
                        "border between region 1 and 2",
                        "border between regions 2 and 3",
                        "border between regions 2 and 4",
                        "border between regions 3 and 4",
                        "intersection between regions 1, 2 and 4",
                        "intersection between regions 1, 2, 3 and 4",
                        "intersection between regions 1, 2 and 3",
                        "border corner region 1",
                        "border corner region 4",
                        ]

    for i, voxel in enumerate(voxels_of_interest):
        fig = plt.figure(i, figsize=(12, 6))

        y_ = y[voxel]
        x_ = x[voxel]
        est_x_ = est_x[voxel]
        u_ = u[voxel]
        est_u_ = est_u[voxel]

        ax1 = fig.add_subplot(211)
        ax1.plot(y_, ls='-', label="noisy AR signal", lw=1,
                 color='tab:red')
        ax1.plot(x_, ls='-', label="original AR signal", lw=2,
                 color='tab:blue')
        ax1.plot(est_x_, ls='--', label="estimated AR signal", lw=2,
                 color='tab:orange')
        plt.grid()
        plt.xlabel("Time-frames [-]", fontsize=15)
        plt.ylabel("BOLD [-]", fontsize=15)
        plt.legend(fontsize=12)
        ax1.set_title(f"Activity related signal (voxel "
                      f"'{name_of_interest[i]}')", fontsize=15)

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(100. * u_, ls='-', label="original AI signal", lw=2,
                 color='tab:blue')
        ax2.plot(100. * est_u_, ls='-', label="estimated AI signal", lw=2,
                 color='tab:orange')
        plt.grid()
        plt.xlabel("Time-frames [-]", fontsize=15)
        plt.ylabel("(de-)Activation [%]", fontsize=15)
        plt.legend(fontsize=12)
        ax2.set_title(f"Activity inducing signal (voxel "
                      f"'{name_of_interest[i]}')", fontsize=15)

        plt.tight_layout()

        filename = "voxel_{0}_{1}_{2}.pdf".format(*voxel[:-1])
        filename = os.path.join(args.plots_dir, filename)
        plt.savefig(filename, dpi=300)

    delta_t = time.strftime("%H h %M min %S s", time.gmtime(time.time() - t0))
    print("Script runs in: {}".format(delta_t))
