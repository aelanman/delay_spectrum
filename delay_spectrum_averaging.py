#!/bin/env python

#SBATCH -J delay_spectra
#SBATCH --mem=80G
#SBATCH -n 3
#SBATCH -t 2-00:00:00
#SBATCH -A jpober-condo
#SBATCH --qos=jpober-condo

from __future__ import print_function

import numpy as np
import sys
import os
import glob
import yaml
import argparse
import time
from pyuvdata import UVData
from cosmo_funcs import *
from pyuvsim.utils import check_file_exists_and_increment

# Each file is expected to be the result of a simulation with 
# the same configuration.

parser = argparse.ArgumentParser()

parser.add_argument('files', nargs='+', help='<Required> List of miriad file paths')
parser.add_argument('--steps', type=int, help='Step size for averaging lengths.', default=10)
#parser.add_argument('-a', '--average_baselines', action='store_true', help='Average together visibilities across baselines.')
parser.add_argument('-o', '--out_path', type=str, help='Outfile path', default='results')
parser.add_argument('-p', '--outfile_prefix', type=str, help='Prefix for outfiles', default='rms')
parser.add_argument('--clobber', type=bool, help='Overwrite output npz', default=False)
parser.add_argument('--rms', action='store_true', help='Set true to take sqrt of variances.', default=False)
parser.add_argument('--save_delay', action='store_true', help='Save delay spectra.', default=False)

args = parser.parse_args()

filelist = args.files

Nskies = len(filelist)
print(Nskies)

partitions = {}
averages=[]     # The first partition for each length

uv = UVData()
def varfunc(arr, axis=None):
    if args.rms:
        return np.sqrt(np.var(arr, axis=axis))
    return np.var(arr, axis=axis)

for sky_i in range(Nskies):
    print('Sky: ', sky_i)
    sys.stdout.flush()
    #uv.read_miriad(filelist[sky_i])
    if filelist[sky_i].endswith('uvh5'):
        uv.read_uvh5(filelist[sky_i])
    else:
        uv.read_miriad(filelist[sky_i])
    if uv.phase_type == 'phased':
        uv.unphase_to_drift()
    if sky_i == 0:
        Nfreqs = uv.Nfreqs
        Nbls = uv.Nbls
        Ntimes = uv.Ntimes
        
        baseline_vecs = uv.uvw_array[:uv.Nbls]

        # Define averaging lengths
        avg_lens = np.arange(2, uv.Ntimes, args.steps)
        time_arr = np.unique(uv.time_array)
        avg_lens_time = (time_arr[avg_lens] - time_arr[0]) * 24. # Hours

        # Getting cosmological factors
        Zs = 1420e6/uv.freq_array[0] - 1
        dnu = uv.channel_width   # Hz
        Zmean = np.mean(Zs)
        etas = np.fft.fftfreq(uv.Nfreqs, d=dnu)
        k_parallel = dk_deta(Zmean) * etas

        # Beam integral
        Bandwidth = Nfreqs * dnu
        Opp_I = uv.extra_keywords['bsq_int']
        scalar = X2Y(Zmean) * (Bandwidth / Opp_I)

        # Reference line:
        Nside = uv.extra_keywords['nside']
        skysigma = uv.extra_keywords['skysig']
        om = 4*np.pi/(12.*Nside**2)
        ref_line = X2Y(Zmean) * dnu * om * skysigma**2
#        ref_line_2 = comoving_voxel_volume(Zmean, dnu/1e6, om) * skysigma**2
#        print("Ref line comparison: x2y={:.4f}  comoving_vox_vol={:.4f}".format(ref_line, ref_line_2))

        # Initialize outputs
        partitions = {l : [] for l in avg_lens}
        bl_tuples = zip(uv.ant_1_array[:Nbls], uv.ant_2_array[:Nbls])

    elif not np.all([Nfreqs == uv.Nfreqs, Nbls == uv.Nbls, Ntimes == uv.Ntimes]):
        raise AssertionError("Inconsistency among input data files")

    vis_I = uv.data_array[:,0,:,0]
    vis_I *= jy2Tstr(uv.freq_array[0])
    print("Doing delay transform")
    _visI = np.fft.ifft(vis_I, axis=1)
    dspec_instr = _visI*_visI.conj()
    dspec_instr = dspec_instr.astype(float)
    dspec_instr = dspec_instr.reshape((Ntimes, Nbls, Nfreqs))
#    if args.average_baselines:
#        dspec_instr = np.mean(dspec_instr, axis=1)

    dspec_I =  dspec_instr * scalar

    if args.save_delay:
        ofilename = os.path.basename(filelist[sky_i])
        ofilename = os.path.join(args.out_path, ofilename)
        np.savez(ofilename, k_parallel=k_parallel, dspec_I=dspec_I, ref_line=ref_line)
        continue

    print("Partitioning")
    for li, l in enumerate(avg_lens):
        parts = [dspec_I[j*l:(j+1)*l,...] for j in xrange(int(np.floor(Ntimes/float(l))))]
        parts = [p for p in parts if len(p) > 1]
        parts = np.average(parts, axis=1).tolist()
        partitions[l].extend(parts)     # Extend the list of averaged power spectra by this amount
        if sky_i == 0:
            averages.append(parts[0])

# partitions[length] has shape (Nskies*Nparts, Nbls, Ndelays)

assert all(sorted(partitions.keys()) == avg_lens)

frac_err_per_k_len = np.array([np.mean([pk/ref_line for pk in partitions[l]], axis=0) for l in sorted(partitions.keys())])
#import IPython; IPython.embed()

variances = np.array([varfunc(partitions[l], axis=0) for l in sorted(partitions.keys())])
mean_vars = np.mean(variances, axis=2)
stderrs = np.sqrt(np.var(variances, axis=2))
Nsamples = [len(partitions[l]) for l in avg_lens]

print('Ratio: {:.5f}'.format(np.mean(frac_err_per_k_len[-1][0])))

save_dict = dict(variances=variances, mean_vars=mean_vars, stderrs=stderrs, avg_lens_nsamp = avg_lens,
                    avg_lens_time = avg_lens_time, bls=bl_tuples, frac_err_per_k_len=frac_err_per_k_len,
                    nsamps=Nsamples, ref_line=ref_line,# averaged=args.average_baselines,
                    baseline_vecs = baseline_vecs, averages=averages, k_parallel=k_parallel,
                    file_list=filelist)

# Name should indicate Nbls, Nside, Duration (hours), averaged, Nskies

bl_str = "{:d}bls".format(Nbls)
#if args.average_baselines:
#    bl_str += '-avg'
duration = (time_arr[-1] - time_arr[0]) * 24.   # Hours
ofilename = args.outfile_prefix+"_{}_nside{:d}_{:.2f}hrs_{:d}skies.npz".format(bl_str, Nside, duration, Nskies)

ofilename = os.path.join(args.out_path, ofilename)
if not args.clobber:
        ofilename = check_file_exists_and_increment(ofilename)

np.savez(ofilename, **save_dict)

# Variances dictionary:
#   keys   = averaging length
#   values = list of variance per partition of delay spectra averaged within (up to) Nboot partitions (non-overlapping!)
#   Append to each list for each file. (It's just getting more independent samples for each.)
#   Average and variance of each 
# 
# Averages dictionary:
#   first averaged pspec of each length. Check normalization against expectation.

# Save --- Averages dictionary, variances dictionary, reference line, source file list, averaging lengths in number and time

