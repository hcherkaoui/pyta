# coding: utf-8
""" Example of the tempral proximal operator TA on the synthetic experimentaion
of the paper."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import os
import shutil
import time
from datetime import datetime, timedelta
from pprint import pformat
import matplotlib.pyplot as plt
import numpy as np

from pyta import prox_t
from pyta.data import little_brain


print(__doc__)

d = datetime.now()
root_outdir = 'results_simu_{0}{1}{2}{3}{4}'.format(d.year, d.month, d.day,
                                                    d.hour, d.minute)
if not os.path.exists(root_outdir):
    os.makedirs(root_outdir)

###########################################################################
# Saving the parameters of launch
print("Saving python file under '{0}/'\n".format(root_outdir))
shutil.copyfile(__file__, os.path.join(root_outdir, __file__))

###########################################################################
# Synthetic signal generation
tr = 1.0
nx, ny, nz = 10, 10, 10
N = 200
input_snr = 1.0
random_state = 42
params = dict(tr=tr, nx=nx, ny=ny, nz=nz, N=N,
              snr=input_snr, random_state=random_state)
noisy_AR_s, AR_s, AI_s, I_s, _ = little_brain(**params)

Y = noisy_AR_s.get_data()
noisy_AR_s = noisy_AR_s.get_data()
AR_s = AR_s.get_data()
AI_s = AI_s.get_data()
I_s = I_s.get_data()

snr_num = np.linalg.norm(AR_s)
snr_deno = np.linalg.norm(Y - AR_s)
input_snr = 20 * np.log10(snr_num / snr_deno)

print("-"*80)
print("Input 4D fMRI data has dimension of: {0}".format(Y.shape))
print("Input SNR of {0:.2f}dB".format(input_snr))

###########################################################################
# Parameters definition
len_h = 30
params = dict(t_r=tr, len_h=len_h, n_jobs=-2, lbda_type='ratio', lbda=0.02,
              unbiased=False, verbose=1)
print("-"*80)
print("Will launch the analysis with params:\n{0}\n".format(pformat(params)))

###########################################################################
# Signal formating
dim_x, dim_y, dim_z, N = Y.shape
Y = Y.reshape(dim_x*dim_y*dim_z, N)
params['Y'] = Y  # add here to avoid ackward pprint

###########################################################################
# Signal recovery
print("-"*80)
print("Running the temporal proximal operator analysis, please wait...\n")
t0 = time.time()
est_AR_s, est_AI_s, est_I_s = prox_t(**params)
delta_t = int(time.time() - t0)

###########################################################################
# reshapping
est_AR_s = est_AR_s.reshape(dim_x, dim_y, dim_z, N)
est_AI_s = est_AI_s.reshape(dim_x, dim_y, dim_z, N - len_h + 1)
est_I_s = est_I_s.reshape(dim_x, dim_y, dim_z, N - len_h + 1)

###########################################################################
# Display duration
print("-"*80)
print("runtime: {0}".format(timedelta(seconds=delta_t)))

###########################################################################
# Plotting
nx, ny, nz, N = est_AR_s.shape

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

outdir_plots = os.path.join(root_outdir, "plots")
print("Saving plots files under '{0}/'\n".format(outdir_plots))
if not os.path.exists(outdir_plots):
    os.makedirs(outdir_plots)

for i, voxel in enumerate(voxels_of_interest):
    fig = plt.figure(i, figsize=(20, 10))

    noisy_ar_s = noisy_AR_s[voxel]
    ar_s = AR_s[voxel]
    est_ar_s = est_AR_s[voxel]
    ai_s = AI_s[voxel]
    est_ai_s = est_AI_s[voxel]
    i_s = I_s[voxel]
    est_i_s = est_I_s[voxel]

    ax1 = fig.add_subplot(311)
    ax1.plot(noisy_ar_s, '-y', label="noisy AR signal")
    ax1.plot(ar_s, '-b', label="original AR signal")
    ax1.plot(est_ar_s, '--g', label="estimated AR signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax1.set_title("Activity related signal")

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(ai_s, '-b', label="original AI signal")
    ax2.plot(est_ai_s, '-g', label="estimated AI signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax2.set_title("Activity inducing signal")

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.stem(i_s, '-b', label="original I signal")
    ax3.stem(est_i_s, '-g', label="estimated I signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax3.set_title("Innovation signal")

    plt.tight_layout()

    fig.suptitle("Voxel '{0}', "
                 "located at {1}".format(name_of_interest[i],
                                         voxel[:-1]), fontsize=18)

    filename = os.path.join(outdir_plots,
                            "voxel_{0}_{1}_{2}.png".format(*voxel[:-1]))
    plt.savefig(filename)
