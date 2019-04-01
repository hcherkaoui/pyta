# coding: utf-8
""" Simple example for the spatial proximal operator."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import os
import shutil
import time
from datetime import datetime, timedelta
from pprint import pformat
import numpy as np
import nibabel as nib

from pyta import _prox_s


print(__doc__)

d = datetime.now()
root_outdir = 'results_prox_s_{0}{1}{2}{3}{4}'.format(d.year, d.month, d.day,
                                                      d.hour, d.minute)
if not os.path.exists(root_outdir):
    os.makedirs(root_outdir)

###########################################################################
# Saving the parameters of launch
print("Saving python file under '{0}/'\n".format(root_outdir))
shutil.copyfile(__file__, os.path.join(root_outdir, __file__))

###############################################################################
# Data loading
img = nib.load('data/T1.nii')
X = img.get_data().astype(float)
X -= np.mean(X)
X /= np.std(X)

###########################################################################
# Parameters definition
params = dict(lbda=0.5)
print("-"*80)
print("Will launch the analysis with params:\n{0}\n".format(pformat(params)))

###########################################################################
# Signal formating
params['y'] = X  # add here to avoid ackward pprint

###########################################################################
# Signal recovery
print("-"*80)
print("Running the spatial proximal operator analysis...\n")
t0 = time.time()
est_AR_s = _prox_s(**params)
delta_t = int(time.time() - t0)

###########################################################################
# Display duration
print("-"*80)
print("runtime: {0}".format(timedelta(seconds=delta_t)))

###########################################################################
# Plotting
filename = "tv_T1.nii"
img = nib.Nifti1Image(est_AR_s, np.eye(4))
img.to_filename(os.path.join(root_outdir, filename))
