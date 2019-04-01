# coding: utf-8
""" Example of the tempral proximal operator TA on the synthetic experimentaion
of the paper."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import os
is_travis = ('TRAVIS' in os.environ)
if is_travis:
    import matplotlib
    matplotlib.use('Agg')

import os
import shutil
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from pyta import ta
from pyta.data import little_brain
from pyta.utils import spm_hrf


print(__doc__)

d = datetime.now()
root_outdir = 'results_ta_{0}{1}{2}{3}{4}'.format(d.year, d.month, d.day,
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
N_orig = 200
input_snr = 1.0
random_state = 42
len_h = 20
h = spm_hrf(tr, len_h)
params = dict(tr=tr, nx=nx, ny=ny, nz=nz, N=N_orig,
              snr=input_snr, h=h, random_state=random_state)
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
print("fMRI data: shape={0}, SNR={1:.2f}dB".format(Y.shape, input_snr))

###########################################################################
# Parameters definition
params = dict(t_r=tr, h=h, vol_shape=(nx, ny, nz), lbda_t=5.0e-2,
              lbda_s=5.0e-2, update_weights=[0.5, 0.5], max_iter=10,
              n_jobs=-2)

###########################################################################
# Signal formating
dim_x, dim_y, dim_z, N = Y.shape
Y = Y.reshape(dim_x*dim_y*dim_z, N)
params['Y'] = Y  # add here to avoid ackward pprint

###########################################################################
# Signal recovery
print("-"*80)
print("Running TA analysis...\n")
t0 = time.time()
est_AR_s, est_AI_s, est_I_s = ta(**params)
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
# saving
filename = "est_AR_s.nii"
img = nib.Nifti1Image(est_AR_s, np.eye(4))
img.to_filename(os.path.join(root_outdir, filename))
filename = "est_AI_s.nii"
img = nib.Nifti1Image(est_AI_s, np.eye(4))
img.to_filename(os.path.join(root_outdir, filename))
filename = "est_I_s.nii"
img = nib.Nifti1Image(est_I_s, np.eye(4))
img.to_filename(os.path.join(root_outdir, filename))

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

for i, voxel in enumerate(voxels_of_interest):
    fig = plt.figure(i, figsize=(15, 7))

    noisy_ar_s = noisy_AR_s[voxel]
    ar_s = AR_s[voxel]
    est_ar_s = est_AR_s[voxel]
    ai_s = AI_s[voxel]
    est_ai_s = est_AI_s[voxel]
    i_s = I_s[voxel]
    est_i_s = est_I_s[voxel]

    ax1 = fig.add_subplot(311)
    ax1.plot(noisy_ar_s, '-y', label="noisy AR signal", lw=1)
    ax1.plot(ar_s, '-b', label="original AR signal", lw=3)
    ax1.plot(est_ar_s, '--g', label="estimated AR signal", lw=3)
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax1.set_title("Activity related signal")

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(ai_s, '-b', label="original AI signal", lw=3)
    ax2.plot(est_ai_s, '-g', label="estimated AI signal", lw=3)
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax2.set_title("Activity inducing signal")

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.stem(i_s, label="original I signal")
    ax3.stem(est_i_s, label="estimated I signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax3.set_title("Innovation signal")

    plt.tight_layout()

    fig.suptitle("Voxel '{0}', "
                 "located at {1}".format(name_of_interest[i],
                                         voxel[:-1]), fontsize=18)

    filename = os.path.join(root_outdir,
                            "voxel_{0}_{1}_{2}.png".format(*voxel[:-1]))
    plt.savefig(filename)
