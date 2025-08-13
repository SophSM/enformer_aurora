import os
import h5py
import numpy as np
import pandas as pd
import kipoiseq
import argparse
'''
Usage:

python3 -u expand_intervals.py --hdf5_file "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/basenji_data_h5/test_pop_seq.hdf5" \
--out_file "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/expanded_test_intervals.txt"

'''

# ----Arguments----
parser = argparse.ArgumentParser()
parser.add_argument('--hdf5_file', help='dataset with smaller sized-sequences')
parser.add_argument('--out_file')
args = parser.parse_args()

# ---Get regions----

with h5py.File(args.hdf5_file, 'r') as f:
    n_regions = f['query_regions'].shape[0]

with h5py.File(args.hdf5_file, 'r') as f, open(args.out_file, 'w') as out_file:
    for r in range(n_regions):
        small_region = f['query_regions'][r, :]
        if small_region[0] == 0:
            chr = 'X'
        else:
            chr = small_region[0]
        expanded_interval = kipoiseq.Interval(chrom = f"chr{chr}",
                                           start = small_region[1],
                                           end = small_region[2]).resize(393216)
        
        interval_str = f"{expanded_interval.chrom}\t{expanded_interval.start}\t{expanded_interval.end}"

        out_file.write(interval_str + '\n')
        if (r+1) % 200 ==0:
            print(f"[INFO] {r+1}/{n_regions} regions done")