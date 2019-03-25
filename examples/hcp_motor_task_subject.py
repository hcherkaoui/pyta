# coding: utf-8
""" Simple fMRI example: example to recover the different spontanious tasks
involved in the BOLD signal."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import os
import time
from datetime import datetime, timedelta
from pprint import pformat
import shutil
import pickle

import matplotlib.pyplot as plt
from nilearn import input_data, datasets, image

from pyta import prox_t
from utils import (fetch_subject_list, get_hcp_fmri_fname, TR_HCP)


###############################################################################
# results management
print(__doc__)

date = datetime.now()
dirname = 'results_hcp_#{0}{1}{2}{3}{4}'.format(date.year, date.month,
                                                date.day, date.hour,
                                                date.minute)

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# define data
harvard_oxford = datasets.fetch_atlas_harvard_oxford(
        'cort-maxprob-thr50-2mm', symmetric_split=True)
atlas_data = harvard_oxford.maps.get_data()
mask = atlas_data == harvard_oxford.labels.index('Right Precentral Gyrus')
mask_img = image.new_img_like(harvard_oxford.maps, mask)
subject_id = fetch_subject_list()[0]
fmri_img = get_hcp_fmri_fname(subject_id)
mask_img = image.resample_to_img(
    mask_img, fmri_img, interpolation='nearest')
masker = input_data.NiftiMasker(mask_img, t_r=TR_HCP, standardize=True)
X = masker.fit_transform(fmri_img).T
P, T = X.shape
print("Data loaded shape: {}".format(X.shape))

###########################################################################
# Parameters definition
len_h = 30
params = dict(t_r=TR_HCP, len_h=len_h, n_jobs=-2, lbda_type='ratio', lbda=0.1,
              unbiased=False, verbose=1)
print("-"*80)
print("Will launch the analysis with params:\n{0}\n".format(pformat(params)))
params['Y'] = X  # add here to avoid ackward pprint

###########################################################################
# Signal recovery
print("-"*80)
print("Running the temporal proximal operator analysis, please wait...\n")
t0 = time.time()
est_AR_s, est_AI_s, est_I_s = prox_t(**params)
delta_t = int(time.time() - t0)

###########################################################################
# Display duration
print("-"*80)
print("runtime: {0}".format(timedelta(seconds=delta_t)))

###############################################################################
# archiving results
res = dict(est_AR_s=est_AR_s, est_AI_s=est_AI_s, est_I_s=est_I_s)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
# u
voxels_of_interest = [(0, slice(None)),
                      (50, slice(None)),
                      (100, slice(None)),
                      ]

for i, voxel in enumerate(voxels_of_interest):
    fig = plt.figure(i, figsize=(20, 10))

    noisy_ar_s = X[voxel]
    est_ar_s = est_AR_s[voxel]
    est_ai_s = est_AI_s[voxel]
    est_i_s = est_I_s[voxel]

    ax1 = fig.add_subplot(311)
    ax1.plot(noisy_ar_s, '-y', label="noisy AR signal")
    ax1.plot(est_ar_s, '--g', label="estimated AR signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax1.set_title("Activity related signal")

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(est_ai_s, '-g', label="estimated AI signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax2.set_title("Activity inducing signal")

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.stem(est_i_s, '-g', label="estimated I signal")
    plt.xlabel("n scans")
    plt.ylabel("amplitude")
    plt.legend()
    ax3.set_title("Innovation signal")

    plt.tight_layout()

    fig.suptitle("Voxel-{0}".format(voxel[:-1]), fontsize=18)

    filename = os.path.join(dirname, "voxel_{0}.png".format(*voxel[:-1]))
    print("Saving plot under '{0}'".format(filename))

    plt.savefig(filename)
