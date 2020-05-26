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
from pyta.hrf_model import double_gamma_hrf


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
    args = parser.parse_args()

    print(__doc__)
    print('*' * 80)

    t0_global = time.time()

    if not os.path.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    ###########################################################################
    # Real data loading
    t_r = 0.735

    hrf_time_frames = 20
    h = double_gamma_hrf(t_r, hrf_time_frames)

    sub1_img = 'data/6025086_20227_MNI_RS.nii.gz'
    sub2_img = 'data/6025837_20227_MNI_RS.nii.gz'

    # load data
    masker = NiftiMasker(standardize=True, detrend=True, low_pass=0.1,
        high_pass=0.01, t_r=t_r, memory='__cache_dir__')
    masker.fit([sub1_img, sub2_img])
    y_train = masker.inverse_transform(masker.transform(sub1_img)).get_data()
    y_test = masker.inverse_transform(masker.transform(sub2_img)).get_data()

    # mask data
    msg = (f"n_time_frames should be between {2 * hrf_time_frames} and "
           f"{y_train.shape[-1]}, got {args.n_time_frames}")
    assert ((2 * hrf_time_frames <= args.n_time_frames)
            and (args.n_time_frames <= y_train.shape[-1])), msg

    start_, length_x_, length_y_, length_z_ = 10, 20, 20, 20
    mask_roi = (slice(start_, start_ + length_x_),
                slice(start_, start_ + length_y_),
                slice(start_, start_ + length_z_),
                slice(0, args.n_time_frames))
    y_train = y_train[mask_roi]
    y_test = y_test[mask_roi]

    print(f"Shape of chosen region to decompose : {y_test.shape}")

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