import os
import numpy as np
import pandas as pd
import tensorflow as tf
import kipoiseq
import h5py
from tqdm import tqdm
from collections import defaultdict
import pyfaidx
import argparse

'''
Usage:

python3 -u get_datasets.py --out_file "/eagle/AIHPC4Edu/ssalazar/projects/enformer_training/basenji_data_h5/test_mouse.hdf5" \
--bed_file "/grand/TFXcan/imlab/data/enformer_training_data/basenji_data_tfr/mouse/sequences.bed" \
--fasta_file "/eagle/AIHPC4Edu/ssalazar/Data/hg_sequences/mm10/mm10.fa" \
--organism "mouse" \
--tf_records "/eagle/AIHPC4Edu/ssalazar/Data/enformer_training_data/" \
--split "test" 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--out_file')
parser.add_argument('--bed_file')
parser.add_argument('--fasta_file')
parser.add_argument('--tf_records')
parser.add_argument('--organism')
parser.add_argument('--split')
args = parser.parse_args()

class FastaStringExtractor:
    def __init__(self, fasta_file):
        

        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.

        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = kipoiseq.Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                            trimmed_interval.start + 1,
                                            trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
    
def deserialize(serialized_example, metadata):

    feature_map = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence, 'target': target}



def load_tfrecord_to_numpy(tfrecord_path, metadata):
    dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type='ZLIB')
    dataset = dataset.map(lambda x: deserialize(x, metadata))
    sequences = []
    targets = []
    for example in dataset:
        sequences.append(example['sequence'].numpy())
        targets.append(example['target'].numpy())
    sequences = np.stack(sequences)
    targets = np.stack(targets)
    return {'sequence': sequences, 'target': targets}

def parse_bed_row(row):
    if row[0] == 'chrX':
        chrom = 0
    else:
        chrom = int(row[0].replace('chr', ''))
    return np.array([chrom, int(row[1]), int(row[2])])
def expand_sequence_from_bed_row(row, fasta_extractor, size=393216):
    region_interval = kipoiseq.Interval(chrom = row[0], start=int(row[1]), end = int(row[2])).resize(size)
    seq = kipoiseq.transforms.functional.one_hot_dna(fasta_extractor.extract(region_interval))
    return seq

def write_record(dseq, seq):
    dseq.resize(dseq.shape[0] + 1, axis=0)
    dseq[-1:] = seq

fasta_file = args.fasta_file
fasta_extractor = FastaStringExtractor(fasta_file)
out_file = args.out_file
bed_file = args.bed_file
orgamism = args.organism
split = args.split

bed_df = pd.read_csv(bed_file, header=None, sep="\t")
bed_split = bed_df[bed_df[3]==split]
bed_split['row_num'] = range(len(bed_split))

split_observations = bed_split['row_num'].to_list()

grouped_indices = defaultdict(list)
for x in split_observations:
    i, j = x // 256, x % 256
    grouped_indices[i].append(j)

if args.organism == 'human':
    num_targets = 5313
else:
    num_targets = 1643

metadata = { 
    'seq_length': 131072,
    'target_length': 896,
    'num_targets': num_targets
}

if args.organism == 'mouse':
    file_id = 1
else:
    file_id = 0

with h5py.File(out_file, 'w') as outFile:
    new_data = {
        'large_sequence': outFile.create_dataset(
            "large_sequence", shape=(0, 393216, 4), 
            maxshape=(None, 393216, 4), 
            dtype='float32'),
        'target': outFile.create_dataset(
            "target", shape=(0, 896, num_targets), 
            maxshape=(None, 896, num_targets), 
            dtype='float32'),
        'query_regions': outFile.create_dataset(
            "query_regions", shape=(0, 3), 
            maxshape=(None, 3), 
            dtype='int64')

    }
    
    for i in tqdm(sorted(grouped_indices)):
        
        bed_idx = 0 + (i*256) # row number on bed file
        print(bed_idx)
        record = load_tfrecord_to_numpy(
        tfrecord_path=os.path.join(args.tf_records, f"{args.organism}/tfrecords/{split}-{file_id}-{i}.tfr"),
            metadata=metadata
        )['target']
        for j in grouped_indices[i]:
            row = bed_split[bed_split['row_num']==bed_idx].iloc[0]
            region = parse_bed_row(row) # returns array with region
            if j % 20 ==0: print(region)
            expanded_seq = expand_sequence_from_bed_row(row, fasta_extractor)
            target = (record[j])
            assert expanded_seq.shape == (393216, 4) and target.shape== (896, num_targets)

            write_record(new_data['large_sequence'], expanded_seq)
            write_record(new_data['query_regions'], region)
            write_record(new_data['target'], target)
            bed_idx += 1