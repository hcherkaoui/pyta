""" Convergence rate between iterative-step-z and learn-step-z for TA
decomposition
"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

import os
import time
import argparse
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from pyta import TA
from pyta.hrf_model import double_gamma_hrf
from pyta.utils import compute_lbda_max


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=300,
                        help='Max number of iterations to for the z-step.')
    parser.add_argument('--max-training-iter', type=int, default=300,
                        help='Max number of iterations to train the '
                        'learnable networks for the z-step.')
    parser.add_argument('--plots-dir', type=str, default='outputs',
                        help='Outputs directory for plots')
    args = parser.parse_args()

    print(__doc__)
    print('*' * 80)

    t0 = time.time()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    ###########################################################################
    # Real data loading
    t_r = 0.735

    len_h = 30
    h = double_gamma_hrf(t_r, len_h)

    sub1_img = 'data/6017587_20227_MNI_RS.nii'
    sub2_img = 'data/6015996_20227_MNI_RS.nii'

    masker = NiftiMasker(smoothing_fwhm=6.0, standardize=True, detrend=True,
                         low_pass=0.1, high_pass=0.01, t_r=t_r,
                         memory='__cache_dir__')
    masker.fit([sub1_img, sub2_img])
    y_train = masker.inverse_transform(masker.transform(sub1_img)).get_data()
    y_test = masker.inverse_transform(masker.transform(sub2_img)).get_data()

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

    _, _, _ = ta_iterative.transform(y_test, lbda_t, lbda_s)

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
