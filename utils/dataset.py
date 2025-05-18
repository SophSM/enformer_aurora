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

    def __init__(self, hdf5_file_human, hdf5_file_mouse=None, shift_augmentation=False, complementary_chain_augmentation=False):
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


    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.dataset_shape_human[0]


    def __getitem__(self, idx):
        """Get one sample of data from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Open the HDF5 file and retrieve the data for the given index
        with h5py.File(self.hdf5_file_human, 'r') as hdf:
            sequence_human = hdf['sequence'][idx]
            target_human = hdf['target'][idx]
        
        if self.hdf5_file_mouse is not None:
            with h5py.File(self.hdf5_file_mouse, 'r') as hdf:
                sequence_mouse = hdf['sequence'][idx%self.n_mouse_seqs]
                target_mouse = hdf['target'][idx%self.n_mouse_seqs]

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
        print(sequence_human.shape)

        if self.hdf5_file_mouse is not None:
            # sequence_mouse = sequence_mouse[ind_min:ind_max]
            sequence_mouse = sequence_mouse # (131072, 4)
            print(sequence_mouse.shape)

        data_point = {
            'sequence_human': torch.tensor(sequence_human).float(), 
            'target_human': torch.tensor(target_human),            
            'shift': shift,
            'complementary': reverse_strand
        }

        if self.hdf5_file_mouse is not None:
            data_point['sequence_mouse'] = torch.tensor(sequence_mouse).float()
            data_point['target_mouse']   = torch.tensor(target_mouse)

        return EasyDict(data_point)

def get_datasets(hdf5_file_human, hdf5_file_mouse, split_lengths=None):
    dataset_train = HDF5Dataset(
        hdf5_file_human=hdf5_file_human,
        hdf5_file_mouse=hdf5_file_mouse,
        shift_augmentation=True,
        complementary_chain_augmentation=True,
    )


    if split_lengths is None:
        split_lengths = [32000, 1992]
    
    assert len(split_lengths) in {2, 3} and (split_lengths[0] + split_lengths[1]) <= len(dataset_train)
    total_train_samples = len(dataset_train)
    random_indices = random.sample(range(total_train_samples), split_lengths[0]+split_lengths[1])
    dataset_train = Subset(dataset_train, random_indices)
    dataset_train, dataset_val = random_split(dataset_train, split_lengths[:2])
    
    return dataset_train, dataset_val