import h5py
import intel_extension_for_pytorch as ipex
import torch
import random
import numpy as np

from torch.utils.data import random_split, Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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
            sequence_mouse = sequence_mouse[ind_min:ind_max]

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


def get_dataloaders(batch_size, sampler_cfg, split_lengths=None, num_workers=0, pin_memory=True):
         
    dataset_train = HDF5Dataset(
        hdf5_file_human="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/train_human.hdf5",
        hdf5_file_mouse="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/train_mouse.hdf5",
        shift_augmentation=True,
        complementary_chain_augmentation=True,
    )

    dataset_test = HDF5Dataset(
        hdf5_file_human="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/test_human.hdf5",
        hdf5_file_mouse="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/test_mouse.hdf5"
    )

    if split_lengths is None:
        split_lengths = [32000, 1992]
    
    assert len(split_lengths) in {2, 3} and (split_lengths[0] + split_lengths[1]) <= len(dataset_train)

    total_train_samples = len(dataset_train)
    random_indices = random.sample(range(total_train_samples), split_lengths[0]+split_lengths[1])
    dataset_train = Subset(dataset_train, random_indices)

    dataset_train, dataset_val = random_split(dataset_train, split_lengths[:2])

    if len(split_lengths) == 2 or split_lengths[2] is None:
        pass
    else:
        assert ( 
            isinstance(split_lengths[2], int) and 
            split_lengths[2] < len(dataset_test)
        )        
        total_test_samples = len(dataset_test)
        random_indices = random.sample(range(total_test_samples), split_lengths[2])
        dataset_test = Subset(dataset_test, random_indices)

    if sampler_cfg is not None:
        sampler_train = DistributedSampler(dataset_train, **sampler_cfg)
        sampler_val   = DistributedSampler(dataset_val,   **sampler_cfg)
        sampler_test  = DistributedSampler(dataset_test,  **sampler_cfg)
    else:
        sampler_test = sampler_val = sampler_train = None

    common_kwargs = {
        "batch_size" : batch_size, 
        "num_workers": num_workers, 
        "pin_memory" : pin_memory,
    }

    train_dataloader = DataLoader(dataset_train, sampler=sampler_train , **common_kwargs)
    val_dataloader   = DataLoader(dataset_val,   sampler=sampler_val   , **common_kwargs)
    test_dataloader  = DataLoader(dataset_test,  sampler=sampler_test  , **common_kwargs)

    return train_dataloader, val_dataloader, test_dataloader