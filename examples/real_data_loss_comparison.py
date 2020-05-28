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
from nilearn.input_data import NiftiMasker
from carpet.utils import init_vuz
from pyta import TA
from pyta.hrf_model import double_gamma_hrf, make_toeplitz
from pyta.utils import compute_lbda_max, logspace_layers
from pyta.loss_and_grad import _obj_t_analysis


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

    ###########################################################################
    # Real data loading
    t_r = 0.735
    hrf_time_frames = 30
    n_times_valid = args.n_time_frames - hrf_time_frames + 1

    h = double_gamma_hrf(t_r, hrf_time_frames)
    D = (np.eye(n_times_valid, k=-1) - np.eye(n_times_valid, k=0))[:, :-1]
    H = make_toeplitz(h, n_times_valid).T

    # load data
    sub1_img = 'data/6025086_20227_MNI_RS.nii.gz'
    sub2_img = 'data/6025837_20227_MNI_RS.nii.gz'

    masker = NiftiMasker(standardize=True, detrend=True, low_pass=0.1,
        high_pass=0.01, t_r=t_r, memory='__cache_dir__')  # noqa: E128
    masker.fit([sub1_img, sub2_img])

    y_train = masker.inverse_transform(masker.transform(sub1_img)).get_data()
    y_test = masker.inverse_transform(masker.transform(sub2_img)).get_data()

    # reduce dimensionality
    start_, length_x_, length_y_, length_z_ = 10, 10, 10, 10
    mask_roi = (slice(start_, start_ + length_x_),
                slice(start_, start_ + length_y_),
                slice(start_, start_ + length_z_),
                slice(0, args.n_time_frames))
    y_train = y_train[mask_roi]
    y_test = y_test[mask_roi]

    # lbda-max scale data
    lbda_max = compute_lbda_max(H, y_train, per_sample=True)
    lbda_max = lbda_max.reshape(nx, ny, nz)
    y_train /= np.repeat(lbda_max[..., None], args.n_time_frames, axis=-1)

    lbda_max = compute_lbda_max(H, y_test, per_sample=True)
    lbda_max = lbda_max.reshape(nx, ny, nz)
    y_test /= np.repeat(lbda_max[..., None], args.n_time_frames, axis=-1)

    print(f"Shape of the train-set : {y_train.shape}")
    print(f"Shape of the test-set : {y_test.shape}")

    ###########################################################################
    # Main experimentation
    lbda = 0.9
    multi_iter = 2
    all_layers = logspace_layers(n_layers=10, max_depth=args.max_iter_z)

    params = dict(t_r=t_r, H=H, name='Iterative-z',
                  max_iter_z=multi_iter * args.max_iter_z,
                  solver_type='fista-z-step', verbose=1)
    ta_iter = TA(**params)

    t0 = time.time()
    _, _, _ = ta_iter.prox_t(y_test, lbda)
    print(f"ta_iterative.prox_t finished : {time.time() - t0:.2f}s")
    loss_ta_iter = ta_iter.l_loss_prox_t

    n_samples = length_x_ * length_y_ * length_z_
    y_test_ravel = y_test.reshape(n_samples, args.n_time_frames)
    _, u0, _ = init_vuz(H, D, y_test_ravel, lbda)
    loss_ta_learn = [_obj_t_analysis(u0, y_test_ravel, H, lbda)]

    init_net_params = None
    params = dict(t_r=t_r, H=H, net_solver_training_type='recursive',
                  name='Learned-z', solver_type='learn-z-step', verbose=1,
                  max_iter_training_net=args.max_training_iter)

    for i, n_layers in enumerate(all_layers):

        params['max_iter_z'] = n_layers

        if args.load_net is not None:
            # load and re-used pre-fitted parameters case
            filename = args.load_net[i]
            with open(filename, 'rb') as pfile:
                init_net_params = pickle.load(pfile)
            print(f"Loading parameters from '{filename}'")
            params['init_net_parameters'] = init_net_params
            ta_learn = TA(**params)

        else:
            # fit parameters and save parameters case
            params['init_net_parameters'] = init_net_params
            ta_learn = TA(**params)
            ta_learn.fit(y_train, lbda)
            init_net_params = ta_learn.net_solver.export_parameters()
            filename = f'fitted_params_n_layers_{n_layers}.pkl'
            filename = os.path.join(args.plots_dir, filename)
            with open(filename, 'wb') as pfile:
                pickle.dump(init_net_params, pfile)
            print(f"Saving fitted parameters under '{filename}'")

        t0 = time.time()
        _, u, _ = ta_learn.prox_t(y_test, lbda, reshape_4d=False)
        print(f"ta_learn.prox_t finished : {time.time() - t0:.2f}s")
        loss_ta_learn.append(_obj_t_analysis(u, y_test_ravel, H, lbda))

    loss_ta_learn = np.array(loss_ta_learn)

    ###########################################################################
    # Plotting
    params = dict(t_r=t_r, H=H, max_iter_z=1000, name='Ref-z',
                  solver_type='fista-z-step', verbose=0)
    ta_ref = TA(**params)

    t0 = time.time()
    _, _, _ = ta_ref.prox_t(y_test, lbda)
    print(f"ta_ref.prox_t finished : {time.time() - t0:.2f}s")
    min_loss = ta_ref.l_loss_prox_t[-1]

    all_layers = [0] + all_layers
    eps = 1.0e-10
    lw = 6

    plt.figure(f"[{__file__}] Loss functions", figsize=(6, 3))
    plt.semilogy(np.arange(start=0, stop=multi_iter * args.max_iter_z + 1),
                 loss_ta_iter - min_loss, lw=lw, label='iterative')
    plt.semilogy(all_layers, loss_ta_learn - min_loss, lw=lw,
                 label='learn')
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
