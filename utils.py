import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
FULL_LENGTH = 393_216
SEQLEN = 196_608
SHIFT_AMPLITUDE = 4
class HDF5Dataset(Dataset):
    def __init__(self, file_paths, organism, mode='train', shift_augmentation=True, complementary_chain_augmentation=True, pop_seq=False):
        """
        file_paths: dictionary {'human': path1, 'mouse': path2}
        organism: 'human' or 'mouse'
        mode: 'train' or 'val'
        """
        self.file_path = file_paths[organism][mode]
        self.h5file = h5py.File(self.file_path, 'r')
        if pop_seq:
            self.inputs = self.h5file['pop_sequence']
        else:
            self.inputs = self.h5file['large_sequence']

        self.targets = self.h5file['target']

        self.shift_augmentation = shift_augmentation
        self.complementary_chain_augmentation = complementary_chain_augmentation

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        if x.shape[0] == 0 or x.shape[1] == 0:
            print(f"Empty for index {idx}")
        y = self.targets[idx]

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
            x = x[::-1, ::-1].copy()

        x = x[ind_min:ind_max] 
        return torch.tensor(x).float(), torch.tensor(y).float()

    def close(self):
        self.h5file.close()

def PearsonR(x, y):
    """
    computed column-wise as cov(x,y) / (sd(x)*sd(y)) for each sample in the batch
    returns tensor of shape [batch_size, 5313]

    r = sum((x - colmeans(x)) * (y - colmeans(y))) / sqrt(sum(x - colmeans(x)**2)* sum(y - colmeans(y)**2))
    """
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered.pow(2)).sum(dim=1) * (y_centered.pow(2)).sum(dim=1))
    cor = numerator / denominator
    return(cor)

def quantile_tensor(tensor, dim=None):
    if dim is None:
        q = torch.quantile(tensor, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device = tensor.device))
    else:
        q = torch.quantile(tensor, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device = tensor.device), dim=dim)
    return q

def summary_per_batch(r_batch):
    '''
    Input: PearsonR() output tensor of size (batch_size, 5313)
    Output: Dictionary of tensors of shape (batch_size) for each statistic
    '''
    q = quantile_tensor(r_batch, dim = 1)
    

    mean = r_batch.mean(dim=1)
    summary = {
        'Min': q[0],
        'Q1':q[1],
        'Mean': mean,
        'Median':q[2],
        'Q3': q[3],
        'Max': q[4]
    }
    return summary

def summary_of_summary(summary):
    '''
    Input: dictionary output from summary_per_batch()
    Output: dictionary with 1 number per statistic
    '''
    summary_s = {
        'Median': quantile_tensor(summary['Median'])[2], # median of medians
        'Mean': summary['Mean'].mean(), # mean of means
    }
    if 'Max' in summary.keys():
        summary_s['Max']=summary['Max'].max() # max of max
    if 'Min' in summary.keys():
        summary_s['Min']=summary['Min'].min() # # min of min
    return summary_s


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': None,
    }
    torch.save(checkpoint, checkpoint_path)

def save_stats(args, global_step, elapsed_times,
               human_train_stats, human_val_stats,
               mouse_train_stats, mouse_val_stats):
    
    save_path = os.path.join(args.stats_dir, f"stats_step_{global_step}.npz")

    np.savez_compressed(
        save_path,
        elapsed_times=elapsed_times,
        human_train_mean_cor=human_train_stats['mean_cor'],
        human_val_mean_cor=human_val_stats['mean_cor'],
        mouse_train_mean_cor=mouse_train_stats['mean_cor'],
        mouse_val_mean_cor=mouse_val_stats['mean_cor'],
        human_train_median_cor=human_train_stats['median_cor'],
        human_val_median_cor=human_val_stats['median_cor'],
        mouse_train_median_cor=mouse_train_stats['median_cor'],
        mouse_val_median_cor=mouse_val_stats['median_cor'],
    )
    print(f"Saved stats to {save_path}")

def save_losses(args, global_step,
                human_train_stats, human_val_stats,
                mouse_train_stats, mouse_val_stats):

    save_path = os.path.join(args.stats_dir, f"losses_step_{global_step}.npz")

    np.savez_compressed(
        save_path,
        human_train_loss=human_train_stats['loss'],
        human_val_loss=human_val_stats['loss'],
        mouse_train_loss=mouse_train_stats['loss'],
        mouse_val_loss=mouse_val_stats['loss'],
    )
    print(f"Saved losses to {save_path}")

def train_step(model, x, y, criterion, optimizer, head, cor=False):
    
    model.train()
    out = model(x)
    loss = criterion(out[head], y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if cor:
        '''
        metrics = MetricDict({
        "pearson": PearsonR(reduce_axis=(0,1))
        }).to(device)
        metrics.update_state(out[head], y)
        res = metrics.result()
        '''
        r = PearsonR(out[head], y) # shape: (batch_size, 5313)
        r_summary = summary_of_summary(summary_per_batch(r))
        # res['pearson'] has shape (n_tracks,), then I take the mean over the tracks
        # r = res['pearson'].detach().cpu().numpy().mean()\
        r_mean, r_median = r_summary['Mean'].view(1), r_summary['Median'].view(1)

    else: 
        r_mean = -1
        r_median = -1
    return loss, r_mean, r_median

def valid_step(model, x, y, criterion, head, cor=False):

    model.eval()
    with torch.no_grad():
        out = model(x) 
        loss = criterion(out[head], y)
        if cor:
            '''
            metrics = MetricDict({
            "pearson": PearsonR(reduce_axis=(0,1))
            }).to(device)
            metrics = MetricDict({
            "pearson": PearsonR(reduce_axis=(0,1))
            }).to(device)
            metrics.update_state(out[head], y)
            res = metrics.result()
            '''
            r = PearsonR(out[head], y) # shape: (batch_size, 5313)
            r_summary = summary_of_summary(summary_per_batch(r))
            # res['pearson'] has shape (n_tracks,), then I take the mean over the tracks
            # r = res['pearson'].detach().cpu().numpy().mean()\
            r_mean, r_median = r_summary['Mean'].view(1), r_summary['Median'].view(1)

        else: 
            r_mean = -1
            r_median = -1
    return loss, r_mean, r_median
