import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
import random
import numpy as np
import h5py
from torch.utils.data import random_split, Dataset, Subset
from easydict import EasyDict

'''
A = [1,0,0,0], C = [0,1,0,0], G = [0,0,1,0], T = [0,0,0,1], N = [0,0,0,0]
'''

FULL_LENGTH = 393_216
SEQLEN = 196_608
SHIFT_AMPLITUDE = 4

class HDF5Dataset(Dataset):

    def __init__(self, hdf5_file_human, hdf5_file_mouse=None, shift_augmentation=False, complementary_chain_augmentation=False, pop_seq = False):
        """
        Custom PyTorch Dataset for reading an HDF5 file.
        
        :param hdf5_file: Path to the HDF5 file.
        :param dataset_name: Name of the dataset inside the HDF5 file to read.
        """
        self.hdf5_file_human = hdf5_file_human
        self.hdf5_file_mouse = hdf5_file_mouse
        
        self.shift_augmentation = shift_augmentation
        self.complementary_chain_augmentation = complementary_chain_augmentation

        # Open the HDF5 file and check the shape of the dataset
        with h5py.File(hdf5_file_human, 'r') as hdf:
            self.dataset_shape_human = hdf['sequence'].shape

        if hdf5_file_mouse is not None:
            with h5py.File(hdf5_file_mouse, 'r') as hdf:
                self.dataset_shape_mouse = hdf['sequence'].shape
                self.n_mouse_seqs = self.dataset_shape_mouse[0]
        
        self.pop_seq = pop_seq

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.dataset_shape_human[0]


    def __getitem__(self, idx):
        """Get one sample of data from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.pop_seq:
            key = 'pop_sequence'
        else:
            key = 'sequence'
        # Open the HDF5 file and retrieve the data for the given index
        with h5py.File(self.hdf5_file_human, 'r') as hdf:
            sequence_human = hdf[key][idx]
            target_human = hdf['target'][idx]
            region_human = hdf['query_region'][idx]
        
        if self.hdf5_file_mouse is not None:
            with h5py.File(self.hdf5_file_mouse, 'r') as hdf:
                sequence_mouse = hdf['sequence'][idx%self.n_mouse_seqs]
                target_mouse = hdf['target'][idx%self.n_mouse_seqs]
                region_mouse= hdf['query_region'][idx%self.n_mouse_seqs]

        # Crop full sequence:
        ind_min, ind_max = FULL_LENGTH//2-SEQLEN//2, FULL_LENGTH//2+SEQLEN//2

        if self.shift_augmentation:
            shift = np.random.choice(range(-SHIFT_AMPLITUDE, SHIFT_AMPLITUDE+1))
            ind_min += shift
            ind_max += shift
        else:
            shift = 0

        if self.complementary_chain_augmentation:
            reverse_strand = np.random.choice([False, True])
        else:
            reverse_strand = False
        
        # Both the nucleotide and the direction of the strand is inverted
        if reverse_strand:
            sequence_human = sequence_human[::-1, ::-1].copy()
            if self.hdf5_file_mouse is not None:    
                sequence_mouse = sequence_mouse[::-1, ::-1].copy()
                    
        sequence_human = sequence_human[ind_min:ind_max]

        if self.hdf5_file_mouse is not None:
            sequence_mouse = sequence_mouse[ind_min:ind_max]
            # sequence_mouse = sequence_mouse # (131072, 4)

        data_point = {
            'sequence_human': torch.tensor(sequence_human).float(), 
            'target_human': torch.tensor(target_human),            
            'shift': shift,
            'complementary': reverse_strand,
            'region_human': torch.tensor(region_human)
        }

        if self.hdf5_file_mouse is not None:
            data_point['sequence_mouse'] = torch.tensor(sequence_mouse).float()
            data_point['target_mouse']   = torch.tensor(target_mouse)
            data_point['region_mouse'] = torch.tensor(region_mouse)

        return EasyDict(data_point)

def get_datasets(train_human, val_human, train_mouse, val_mouse, pop_seq = False):
    dataset_train = HDF5Dataset(
        hdf5_file_human=train_human,
        hdf5_file_mouse=train_mouse,
        shift_augmentation=True,
        complementary_chain_augmentation=True,
        pop_seq=pop_seq
    )

    dataset_val = HDF5Dataset(
        hdf5_file_human=val_human,
        hdf5_file_mouse=val_mouse,
        shift_augmentation=True,
        complementary_chain_augmentation=True,
        pop_seq=pop_seq
    )
    
    return dataset_train, dataset_val