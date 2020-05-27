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
from nilearn.input_data import NiftiMasker
from pyta import TA
from pyta.hrf_model import double_gamma_hrf, make_toeplitz


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-iter', type=int, default=20,
                        help='Max number of iterations for the global loop.')
    parser.add_argument('--max-iter-z', type=int, default=100,
                        help='Max number of iterations for the z-step.')
    parser.add_argument('--load-net', type=str, default=None,
                        help='Load pretrained network parameters.')
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

    print("archiving '{0}' under '{1}'".format(__file__, args.plots_dir))
    shutil.copyfile(__file__, os.path.join(args.plots_dir, __file__))

    ###########################################################################
    # Real data loading
    t_r = 0.735

    hrf_time_frames = 30
    h = double_gamma_hrf(t_r, hrf_time_frames)

    n_times_valid = args.n_time_frames - hrf_time_frames + 1
    H = make_toeplitz(h, n_times_valid).T

    sub1_img = 'data/6025086_20227_MNI_RS.nii.gz'
    sub2_img = 'data/6025837_20227_MNI_RS.nii.gz'

    # load data
    masker = NiftiMasker(standardize=True, detrend=True, low_pass=0.1,
        high_pass=0.01, t_r=t_r, memory='__cache_dir__')  # noqa: E128
    masker.fit([sub1_img, sub2_img])
    y_train = masker.inverse_transform(masker.transform(sub1_img)).get_data()
    y_test = masker.inverse_transform(masker.transform(sub2_img)).get_data()

    start_, length_x_, length_y_, length_z_ = 10, 10, 10, 10
    mask_roi = (slice(start_, start_ + length_x_),
                slice(start_, start_ + length_y_),
                slice(start_, start_ + length_z_),
                slice(0, args.n_time_frames))
    y_train = y_train[mask_roi]
    y_test = y_test[mask_roi]

    print(f"Shape of the train-set : {y_train.shape}")
    print(f"Shape of the test-set : {y_test.shape}")

    ###########################################################################
    # Main experimentation
    lbda_t = 0.1

    params = dict(t_r=t_r, H=H, name='Iterative-z',
                  max_iter_z=10*args.max_iter_z,
                  solver_type='iterative-z-step', verbose=1)
    ta_iter = TA(**params)

    t0 = time.time()
    _, _, _ = ta_iter.prox_t(y_test, lbda_t)
    print(f"ta_iterative.prox_t finished : {time.time() - t0:.2f}s")

    params = dict(t_r=t_r, H=H, max_iter_z=args.max_iter_z,
                  net_solver_type='recursive', name='Learned-z',
                  max_iter_training_net=args.max_training_iter,
                  solver_type='learn-z-step', verbose=1)

    if args.load_net is not None:
        with open(args.load_net, 'rb') as pfile:
            init_params = pickle.load(pfile)
        params['init_net_parameters'] = init_params
        print(f"Loading parameters from '{args.load_net}'")
        ta_learn = TA(**params)

    else:
        ta_learn = TA(**params)
        ta_learn.fit(y_train, lbda_t)

    fitted_params = ta_learn.pretrained_network.export_parameters()
    filename = os.path.join(args.plots_dir, 'fitted_parameters.pkl')
    with open(filename, 'wb') as pfile:
        pickle.dump(fitted_params, pfile)
    print(f"Saving fitted parameters under '{filename}'")

    t0 = time.time()
    _, _, _ = ta_learn.prox_t(y_test, lbda_t)
    print(f"ta_learn.prox_t finished : {time.time() - t0:.2f}s")

    ###########################################################################
    # Plotting
    params = dict(t_r=t_r, H=H, max_iter_z=1000, name='Ref-z',
                  solver_type='iterative-z-step', verbose=0)
    ta_ref = TA(**params)

    t0 = time.time()
    _, _, _ = ta_ref.prox_t(y_test, lbda_t)
    print(f"ta_ref.prox_t finished : {time.time() - t0:.2f}s")

    min_loss = ta_ref.l_loss_prox_t[-1]
    eps = 1.0e-10
    lw = 6

    plt.figure(f"[{__file__}] Loss functions", figsize=(6, 3))
    plt.semilogy(ta_iter.l_loss_prox_t - min_loss, lw=lw, label='iterative')
    plt.semilogy(ta_learn.l_loss_prox_t - min_loss, lw=lw, label='learn')
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
