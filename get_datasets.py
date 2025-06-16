import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from random import randrange, random
from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import polars as pl
import os
import tensorflow as tf
import json
import functools
import h5py
import argparse

'''
python get_datasets.py --organism "mouse" \
--bed_file "/grand/TFXcan/imlab/data/enformer_training_data/basenji_data_tfr/mouse/sequences.bed" \
--fasta_file "/eagle/AIHPC4Edu/ssalazar/Data/hg_sequences/mm39/mm39.fa" \
--out_file "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/full_393216bp/mouse_validation.h5" \
--data_split "valid" \
--basenji_path "/grand/TFXcan/imlab/data/enformer_training_data/basenji_data_tfr" 
'''

parser = argparse.ArgumentParser()
parser.add_argument('--organism')
parser.add_argument('--bed_file')
parser.add_argument('--fasta_file')
parser.add_argument('--out_file')
parser.add_argument('--data_split')
parser.add_argument('--basenji_path')
args = parser.parse_args()

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5
reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.fromstring(t, dtype = np.uint8), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# augmentations

def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, 'must be one hot encoding with last dimension equal to 4'
    return torch.flip(one_hot, (-1, -2))

# processing bed files

class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    def __call__(self, chr_name, start, end, return_augs = False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = min(max(start + min_shift, 0) - start, 0)
            max_shift = max(min(end + max_shift, chromosome_length) - end, 1)

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)

        should_rc_aug = self.rc_aug and coin_flip()

        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)

            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)

            return seq
        one_hot = str_to_one_hot(seq)

        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        # returns the shift integer as well as the bool (for whether reverse complement was activated)
        # for this particular genomic sequence

        rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return one_hot, rand_shift_tensor, rand_aug_bool_tensor

class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn = identity,
        chr_bed_to_fasta_map = dict(),
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        return_augs = False
    ):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
        df = filter_df_fn(df)
        self.df = df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            context_length = context_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        seq = self.fasta(chr_name, start, end, return_augs = self.return_augs)
        interval_id = f"{chr_name}_{start}_{end}"
        return seq, interval_id
    
bed_file_path = args.bed_file
fasta_file_path = args.fasta_file

organism_intervals = pl.read_csv(str(bed_file_path), separator = '\t', has_header = False)
organism_intervals = organism_intervals.filter(pl.col("column_4") == args.data_split)

def make_interval_filter(loc_row):
    def filter_fn(df):
        return df.filter(
            (pl.col("column_1") == loc_row['column_1']) &
            (pl.col("column_2") == loc_row['column_2']) &
            (pl.col("column_3") == loc_row['column_3'])
        )
    return filter_fn

def organism_path(organism):
  return os.path.join(args.basenji_path, organism)


def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

def deserialize(serialized_example, metadata):
    '''
    Deserialize bytes stored in TFRecordFile.
    '''
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,(metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)
    return target

def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)

def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset

def get_region(locus_str):
  # locus_str format: chr#_start_end
  parts = locus_str.replace("chr", "").split("_")
  if parts[0].upper() == "X":
    parts[0] = 0
  region = np.array(list(map(int, parts)))
  return region

organism_dataset = get_dataset(args.organism, args.data_split).batch(1).repeat()

def write_record(dseq, dtarget, dregion, seq, target, r):
    dseq.resize(dseq.shape[0] + 1, axis=0)
    dseq[-1:] = seq

    dtarget.resize(dtarget.shape[0] + 1, axis=0)
    dtarget[-1:] = target

    dregion.resize(dregion.shape[0] + 1, axis=0)
    dregion[-1:] = r

if args.organism == "human":
    n_tracks = 5313
else:
    n_tracks = 1643


with h5py.File(args.out_file, 'w') as outFile:

    datasets = {
        args.data_split: {
            "sequence": outFile.create_dataset("sequence", shape=(0, 393216, 4), maxshape=(None, 393216, 4), dtype='float32'),
            "target": outFile.create_dataset("target", shape=(0, 896, n_tracks), maxshape=(None, 896, n_tracks), dtype='float32'),
            "region": outFile.create_dataset("query_region", shape=(0, 3), maxshape=(None, 3), dtype='int64')
        }
    }

    for i, records in enumerate(organism_dataset):
        if i % 200 == 0:
            print(f"{i} samples saved")
        loc_row = organism_intervals[i]
        filter_fn = make_interval_filter(loc_row)
        bed_entry = filter_fn(organism_intervals)
        split = bed_entry['column_4'][0]

        # Prepare data
        target = records.numpy().squeeze(0)

        ds = GenomeIntervalDataset(
            bed_file=bed_file_path,
            fasta_file=fasta_file_path,
            filter_df_fn=filter_fn,
            return_seq_indices=False,
            context_length=393_216,
        )

        seq, interval = ds[0]
        if seq.shape != (393216, 4):
            print(f"skipping sequence for {interval}")
            continue


        seq = seq.numpy()
        r = np.array(get_region(interval))

        # Write to appropriate dataset
        write_record(
            datasets[split]["sequence"],
            datasets[split]["target"],
            datasets[split]["region"],
            seq, target, r
        )