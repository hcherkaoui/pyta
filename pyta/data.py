# coding: utf-8
""" Dataset module."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: new BSD

import numpy as np
from numpy.linalg import norm as norm_2
import nibabel as nib

from .hrf import spm_hrf


def gen_random_events(N, h, nb_events=4, avg_dur=5, var_dur=1,
                      middle_spike=False, tr=1.0, nb_try=10000,
                      nb_try_duration=10000, overlapping=False,
                      random_state=None):
    """ Generate synthetic x, Lz, z signals.
    """
    center_neighbors = [int(N/2)-1, int(N/2)+1]

    if isinstance(random_state, int):
        r = np.random.RandomState(random_state)
    elif random_state is None:
        r = np.random  # tweak to call directly the np.random module
    else:
        raise ValueError("random_state could only be seed-int or None, "
                         "got {0}".format(type(random_state)))

    for _ in range(nb_try):
        offsets = r.randint(0, N, nb_events)
        for _ in range(nb_try_duration):
            durations = avg_dur + var_dur*r.randn(nb_events)
            if any(durations < 1):
                continue  # null or negative duration events: retry
            else:
                break

        durations = durations.astype(int)
        Lz = np.zeros(N)
        for offset, duration in zip(offsets, durations):
            try:
                Lz[offset:offset+duration] += 1
            except IndexError:
                break

        if middle_spike:
            Lz[int(N/2)] += 1
        if (not overlapping) and any(Lz > 1):
            continue  # overlapping events: retry
        if overlapping:
            Lz[Lz > 1] = 1
        if middle_spike and any(Lz[center_neighbors] > 0):
            continue  # middle-spike not isolated: retry
        else:
            z = np.append(0, np.diff(Lz))
        x = np.convolve(h, Lz)
        return x, Lz, z

    raise RuntimeError("Failed to produce an activity-inducing signal...")


def add_gaussian_noise(signal, snr, random_state=None):
    """ Add a Gaussian noise as targeted by the given SNR value.
    """
    if isinstance(random_state, int):
        r = np.random.RandomState(random_state)
    elif random_state is None:
        r = np.random
    else:
        raise ValueError("random_state could only be seed-int or None, "
                         "got {0}".format(type(random_state)))

    s_shape = signal.shape
    noise = r.randn(*s_shape)

    true_snr_num = np.linalg.norm(signal)
    true_snr_deno = np.linalg.norm(noise)
    true_snr = true_snr_num / (true_snr_deno + np.finfo(np.float).eps)
    std_dev = (1.0 / np.sqrt(10**(snr/10.0))) * true_snr
    noise = std_dev * noise
    noisy_signal = signal + noise

    return noisy_signal, noise, std_dev


def little_brain(tr=1.0, nx=10, ny=10, nz=10, N=200, snr=1.0, h=None,
                 random_state=None):
    """ Generate a synthetic cubic brain with four regions, within each region
    a specific pattern is generated.
    """
    # regions definition
    if nx < 3 or ny < 3 or nz < 3 or N < 100:
        raise ValueError("nx, ny, nz should be at least 3 and N at least 100.")
    m1 = [slice(0, int(0.3*nx)), slice(0, ny), slice(0, nz)]
    m2 = [slice(int(0.3*nx), nx), slice(0, int(0.3*ny)), slice(0, nz)]
    m3 = [slice(int(0.3*nx), nx), slice(int(0.3*ny), ny),
          slice(0, int(0.5*nz))]
    m4 = [slice(int(0.3*nx), nx), slice(int(0.3*ny), ny),
          slice(int(0.5*nz), nz)]

    # signals generation
    regions = [m1, m2, m3, m4]
    if h is None:
        n_times_atom = 30
        h = spm_hrf(tr, n_times_atom)
    z = np.zeros((nx, ny, nz, N))
    Lz = np.zeros((nx, ny, nz, N))
    x = np.zeros((nx, ny, nz, N + len(h) - 1))
    y = np.zeros((nx, ny, nz, N + len(h) - 1))
    x[tuple(m1)], Lz[tuple(m1)], z[tuple(m1)] = gen_random_events(
                                        N, h, tr=tr, nb_events=21, avg_dur=1,
                                        var_dur=0, random_state=random_state)
    x[tuple(m2)], Lz[tuple(m2)], z[tuple(m2)] = gen_random_events(
                                        N, h, tr=tr, nb_events=7, avg_dur=1,
                                        var_dur=1, random_state=random_state)
    x[tuple(m3)], Lz[tuple(m3)], z[tuple(m3)] = gen_random_events(
                                        N, h, tr=tr, nb_events=4, avg_dur=12,
                                        var_dur=2, random_state=random_state)
    x[tuple(m4)], Lz[tuple(m4)], z[tuple(m4)] = gen_random_events(
                                        N, h, tr=tr, nb_events=5, avg_dur=12,
                                        var_dur=2, middle_spike=True,
                                        random_state=random_state)
    y[tuple(m1)], noise1, _ = add_gaussian_noise(
                              x[tuple(m1)], snr=snr, random_state=random_state)
    y[tuple(m2)], noise2, _ = add_gaussian_noise(
                              x[tuple(m2)], snr=snr, random_state=random_state)
    y[tuple(m3)], noise3, _ = add_gaussian_noise(
                              x[tuple(m3)], snr=snr, random_state=random_state)
    y[tuple(m4)], noise4, _ = add_gaussian_noise(
                              x[tuple(m4)], snr=snr, random_state=random_state)

    # Add description
    descr = ("The software phantom contains 4 regions in a cube that consists "
             "out of {0} x {1} x {2} voxels. The first region (300 voxels) "
             "was simulated as spike train with gradually increasing ISI from "
             "1 s to 12 s and the second region (210 voxels) was "
             "simulated with random events with uniform duration in "
             "(Afshin-Pour et al., 2012; Aguirre et al.,1998) seconds. The "
             "third region (245 voxels) and the fourth region ("
             "245 voxels) were simulated with random events with uniform "
             "duration in (Afshin-Pour et al., 2012; Chang and Glover, 2010) "
             "seconds. A very short event is inserted into region 4 "
             "(around 100 s). The time resolution was chosen as TR = 1 s. The "
             "activity-inducing signals were convolved with HRF to "
             "obtain the BOLD activity for each region. Each voxel time series"
             " was then corrupted with i.i.d. Gaussian noise such that voxel "
             "time series had SNR of 10 dB.").format(nx, ny, nz)
    snr1 = norm_2(x[tuple(m1)]) / (norm_2(noise1) + np.finfo(np.float).eps)
    snr2 = norm_2(x[tuple(m2)]) / (norm_2(noise2) + np.finfo(np.float).eps)
    snr3 = norm_2(x[tuple(m3)]) / (norm_2(noise3) + np.finfo(np.float).eps)
    snr4 = norm_2(x[tuple(m4)]) / (norm_2(noise4) + np.finfo(np.float).eps)
    list_snr = [np.round(20.0 * np.log10(snr1), 3),
                np.round(20.0 * np.log10(snr2), 3),
                np.round(20.0 * np.log10(snr3), 3),
                np.round(20.0 * np.log10(snr4), 3),
                ]
    info = {'descr': descr, 'tr': 1.0, 'regions': regions, 'snr': list_snr}

    # formating
    y = nib.Nifti1Image(y, np.eye(4))
    x = nib.Nifti1Image(x, np.eye(4))
    Lz = nib.Nifti1Image(Lz, np.eye(4))
    z = nib.Nifti1Image(z, np.eye(4))

    return y, x, Lz, z, info
