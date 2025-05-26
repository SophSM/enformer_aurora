import numpy as np
import h5py
import os
import pandas as pd
import kipoiseq
import argparse

'''
python get_popseq.py --hdf5_file "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/human_train.h5" \
--dosages_dir "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/EUR_allele_freqs"
'''

SEQUENCE_LENGTH = 393_216

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5_file")
parser.add_argument("--dosages_dir")
args = parser.parse_args()

base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def get_popseq(interval, ref_seq, DIR_freqs, length):
    '''
    start: kipoi expanded interval
    ref_seq: one-hot-encoded reference sequence
    DIR_freqs: directory path where allele dosages per chromosome are stored
    '''
    coordinates = np.arange(interval.start + 1, interval.start + 1 + length) # 393_216 long vector of genomic coordinates

    dosages = pd.read_csv(os.path.join(DIR_freqs, f'{interval.chrom}_EUR_allele_freqs.txt'), sep=' ', header=None)
    dosage_series = pd.Series(dosages[3].values, index=dosages[1])
    aligned_dosages = dosage_series.reindex(coordinates, fill_value=0).values # 393_216 long vector of allele dosages

    dosages[2] = dosages[2].map(base_to_index)
    allele_series = pd.Series(dosages[2].values, index=dosages[1])
    aligned_alleles = allele_series.reindex(coordinates, fill_value=0).values

    result = np.zeros((length, 4), dtype=aligned_dosages.dtype)
    rows = np.arange(length)
    cols = aligned_alleles
    result[rows, cols] = aligned_dosages # zeroes and allele dosages values in correct position
    mask = (result != 0).any(axis=1)
    ref_seq[mask] = result[mask] # replace the non-zero allele dosages on reference one hot encoded sequence
    return ref_seq

# ------Main----------- #

with h5py.File(args.hdf5_file, 'r+') as h5f:
    num_seqs = h5f['sequence'].shape[0]

    # Create dataset if it doesn't exist yet
    if 'pop_sequence' not in h5f:
        h5f.create_dataset(
            'pop_sequence',
            shape=(0, SEQUENCE_LENGTH, 4),
            maxshape=(None, SEQUENCE_LENGTH, 4),
            dtype=np.float32
        )

    for n_seq in range(num_seqs):
        if n_seq % 100 == 0: 
            print(f"{n_seq+1}/{(num_seqs)}")
        chr, start, end = h5f['query_region'][n_seq, :]
        ref_seq = h5f['sequence'][n_seq, :]

        if chr == 0:
            chr = 'X'
        else:
            chr = int(chr)
        expanded_interval = kipoiseq.Interval(
            chrom=f'chr{chr}',
            start=int(start),
            end=int(end)
        ).resize(SEQUENCE_LENGTH)

        pop_seq = get_popseq(expanded_interval, ref_seq=ref_seq, DIR_freqs=args.dosages_dir, length=SEQUENCE_LENGTH)

        # Resize the dataset to fit new entry
        h5f['pop_sequence'].resize((n_seq + 1, SEQUENCE_LENGTH, 4))

        # Write the pop_seq into the dataset
        h5f['pop_sequence'][n_seq, :, :] = pop_seq