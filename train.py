from mpi4py import MPI
import os, socket
import torch
from torch import nn, einsum
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Literal
from tqdm import tqdm
import os
import re
import h5py
import numpy as np
from easydict import EasyDict
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
        ]
    )

logger = logging.getLogger()


SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)
# print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

# DDP: initialize distributed communication with nccl backend
torch.distributed.init_process_group(backend='ccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

# DDP: pin GPU to local rank.
torch.xpu.set_device(int(LOCAL_RANK))
device = torch.device('xpu')
torch.manual_seed(0)

class MultiHeadAttention(nn.Module):
    """multi-head attention"""

    def __init__(self,
                 input_dim,
                 value_dim,
                 key_dim,
                 num_heads,
                 scaling=True,
                 attention_dropout_rate=0.1,
                 relative_position_symmetric=False,
                 num_relative_position_features=None,
                 positional_dropout_rate=0.1,
                 zero_initialize=True):
        """Args:
            input_dim: the dimension of input embedding
            value_dim: the size of value embedding
            key_dim: the size of key embedding
            num_heads: the number of attention heads
            scaling: whether to scale the attention logits
            attention_dropout_rates: dropout rate for attention logits
            attention relative_position_symmetric: if True, the symmetric
            version of basis function will be used. if False, a symmeetric and
                asymmetric versions will be used.
            num_relative_position_features: number of relative positional
                features to compute. if None, `value_dim * num_heads` is used.
            positional_dropout_rate: dropout rate for the positional encodings
                if relative positions are used
            zero_initialize: if True, the final linear layer will be 0 initialized
        """
        super().__init__()

        self._input_dim = input_dim
        self._value_dim = value_dim
        self._key_dim = key_dim
        self._num_heads = num_heads
        self._scaling = scaling
        self._attention_dropout_rate = attention_dropout_rate
        self._relative_position_symmetric = relative_position_symmetric
        if num_relative_position_features is None:
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_relative_position_features = (
                (self._value_dim // divisible_by) * divisible_by)
        else:
            self._num_relative_position_features = num_relative_position_features
        self._positional_dropout_rate = positional_dropout_rate

        key_proj_size = self._key_dim * self._num_heads
        embedding_size = self._value_dim * self._num_heads

        # query, key, and value weights
        self._q_layer = nn.Linear(input_dim, key_proj_size, bias=False)
        self._k_layer = nn.Linear(input_dim, key_proj_size, bias=False)
        self._v_layer = nn.Linear(input_dim, embedding_size, bias=False)
        self._embedding_layer = nn.Linear(embedding_size, input_dim)
        nn.init.zeros_(self._embedding_layer.weight)
        nn.init.zeros_(self._embedding_layer.bias)

        # relative position weights
        self._rel_pos_layer = nn.Linear(self._num_relative_position_features,
                                        key_proj_size, bias=False)
        self._rel_content_bias = nn.Parameter(
            torch.randn(1, self._num_heads, 1, self._key_dim))
        self._rel_pos_bias = nn.Parameter(
            torch.randn(1, self._num_heads, 1, self._key_dim))
        nn.init.kaiming_normal_(self._rel_content_bias)
        nn.init.kaiming_normal_(self._rel_pos_bias)

        # dropout layers:
        self._pos_dropout_layer = nn.Dropout(self._positional_dropout_rate)
        self._attn_dropout_layer = nn.Dropout(self._attention_dropout_rate)

    def forward(self, inputs):
        embedding_size = self._value_dim * self._num_heads
        seq_length = inputs.shape[-2]

        q = self._q_layer(inputs)
        k = self._k_layer(inputs)
        v = self._v_layer(inputs)
        """
        b: batch
        h: head
        c: channel
        l: length
        """
        q, k, v = map(lambda x: rearrange(
            x, "b l (h c) -> b h l c", h=self._num_heads), (q, k, v))

        if self._scaling:
            q *= self._key_dim ** -0.5

        distances = torch.arange(-seq_length + 1,
                                 seq_length, device=inputs.device)
        positional_encodings = positional_features_all(
            positions=distances,
            feature_size=self._num_relative_position_features,
            seq_length=seq_length,
            symmetric=True)
        # [Batch, 2L - 1, Cr]

        positional_encodings = self._pos_dropout_layer(positional_encodings)
        rel_k = self._rel_pos_layer(positional_encodings)
        rel_k = rearrange(rel_k, "l (h c) -> h l c", h=self._num_heads)

        rel_logits = einsum("b h i c, h j c -> b h i j",
                            q + self._rel_pos_bias, rel_k)  # [B, H, L, 2L-1]
        rel_logits = relative_shift(rel_logits)  # [B, H, L, L]

        content_logits = einsum("b h i c, b h j c -> b h i j",
                                q + self._rel_content_bias, k)  # [B, H, L, L]

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self._attn_dropout_layer(attn)

        output = einsum("b h i j, b h j c -> b h i c", attn, v)  # [B, H, L, V]
        output = rearrange(output, "b h l c -> b l (h c)")
        return self._embedding_layer(output)


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]


def positional_features_exponential(positions, feature_size, seq_length, min_half_life=3.0):
    """
    Create exponentially decaying positional biases

    Args:
        positions: a 1D vector of length 2*N-1 from [-(N-1), -(N-2), ...,N-2, N-1], where N 
        is the length of input sequence.
        feature_size: the number of basis functions
        seq_length: length of input sequence
        min_half_life: smallest half life

    Returns: 
        matrix with dimensions [2*N - 1, feature_size]
    """
    assert seq_length == torch.max(positions) + 1, \
        "seq_length should be max(positions) + 1"
    max_half_life = np.log(seq_length) / np.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_half_life,
                                    feature_size, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    output = torch.exp(-np.log(2.0) / half_life * positions)
    assert ((output.shape[:-1] == positions.shape[:-1]) and
            (output.shape[-1] == feature_size))
    return torch.exp(-np.log(2.0) / half_life * positions)


def positional_features_central_mask(positions, feature_size, seq_length):
    """
    Create positional feature in which central regions are one and other regions are zero
    """
    assert seq_length == torch.max(torch.abs(positions)) + 1, \
        "seq_length should be max(positions) + 1"

    center_widths = 2 ** torch.arange(1, feature_size + 1,
                                      device=positions.device).float()
    center_widths = center_widths - 1
    output = (center_widths[None, ...] > positions.abs()[..., None]).float()
    assert (output.shape[:-1] == positions.shape and
            output.shape[-1] == feature_size)
    return output


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) -
                         concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions, feature_size, seq_length, stddev=None, start_mean=None, eps=1e-8):
    if stddev is None:
        stddev = seq_length / (2 * feature_size)

    if start_mean is None:
        start_mean = seq_length / feature_size

    mean = torch.linspace(start_mean, seq_length, feature_size,
                          device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None],
                              concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities)
    return outputs


def positional_features_all(positions, feature_size, seq_length, symmetric=False):
    """
    Compute relative positional encodings/features.

    Each positional feature function will compute/provide the same fraction of
    features, making up the total of feature_size.

    Args:
    positions: Tensor of relative positions of arbitrary shape.
    feature_size: Total number of basis functions.
    seq_length: Sequence length denoting the characteristic length that
      the individual positional features can use. This is required since the
      parametrization of the input features should be independent of `positions`
      while it could still require to use the total number of features.
    symmetric: If True, the resulting features will be symmetric across the
      relative position of 0 (i.e. only absolute value of positions will
      matter). If false, then both the symmetric and asymmetric version
      (symmetric multiplied by sign(positions)) of the features will be used.

    Returns:
    Tensor of shape: `positions.shape + [feature_size]`.
    """
    assert seq_length == torch.max(positions) + 1, \
        "seq_length should be max(positions) + 1"

    feature_functions = [positional_features_exponential,
                         positional_features_central_mask,
                         positional_features_gamma]

    num_components = len(feature_functions)
    if not symmetric:
        num_components = 2 * num_components

    assert feature_size % num_components == 0, (f"feature_size has "
                                                 "to be divisible by {num_components}")

    num_basis_per_class = feature_size // num_components

    embeddings = [f(torch.abs(positions), num_basis_per_class, seq_length)
                  for f in feature_functions]
    embeddings = torch.cat(embeddings, dim=-1)

    if not symmetric:
        embeddings = torch.cat(embeddings,
                               torch.sign(positions)[..., None] * embeddings,
                               dim=-1)
    return embeddings

TARGET_LENGTH = 896
HEADS_CHANNELS = {"human": 5313 , "mouse": 1643}

class Print(nn.Module):
    def __init__(self, name):
        super(Print, self).__init__()
        self._name = name

    def forward(self, x):
        print(f"{self._name}: {x.shape}")
        return x

class Enformer(nn.Module):
    """Main class"""

    def __init__(self,
                 channels=1536,
                 num_heads=8,
                 num_transformer_layers=11,
                 pooling_type="attention", 
                 prediction_head: Literal["both", "human", "mouse"] = "human"):
        """
        Args:
            channels: number of convolutional filters
            num_heads: number of attention heads
            num_transformer_layers: number of transformer layers
            pooling_type: "attention" or "max"
        """

        super().__init__()

        dropout_rate = 0.4
        num_alphabet = 4
        assert channels % num_heads == 0, ("channels need to be "
                                           "divisible by heads")

        self.prediction_head = prediction_head

        if prediction_head == 'both':
            heads_channels = HEADS_CHANNELS
        elif prediction_head in HEADS_CHANNELS:
            heads_channels = { prediction_head: HEADS_CHANNELS[prediction_head] }


        stem = nn.Sequential(
            # b: batch
            # l: length
            # c: channel
            Rearrange("b l c -> b c l"),
            nn.Conv1d(num_alphabet, channels // 2, 15, padding="same"),
            Residual(conv_block(channels // 2, channels // 2, 1)),
            SoftmaxPooling1D(channels // 2, pool_size=2)
        )

        filter_list = exponential_linspace_int(
            channels // 2, channels, num=6, divisible_by=128)
        filter_list = [channels // 2, *filter_list]

        conv_layers = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    conv_block(in_channels, out_channels, 5),
                    Residual(conv_block(out_channels, out_channels, 1)),
                    SoftmaxPooling1D(out_channels, pool_size=2)
                )
            )
        conv_tower = nn.Sequential(*conv_layers)

        attn_kwargs = {
            "input_dim": channels,
            "value_dim": channels // num_heads,
            "key_dim": 64,
            "num_heads": num_heads,
            "scaling": True,
            "attention_dropout_rate": 0.05,
            "relative_position_symmetric": False,
            "num_relative_position_features": channels // num_heads,
            "positional_dropout_rate": 0.01,
            "zero_initialize": True
        }

        def transformer_mlp():
            return Residual(nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels * 2),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(channels * 2, channels),
                nn.Dropout(dropout_rate)
            ))

        transformer = []
        for _ in range(num_transformer_layers):
            transformer.append(
                nn.Sequential(
                    Residual(nn.Sequential(
                        nn.LayerNorm(channels),
                        MultiHeadAttention(**attn_kwargs),
                        nn.Dropout(dropout_rate)
                    )),
                    transformer_mlp()
                )
            )

        transformer = nn.Sequential(
            Rearrange("b c l -> b l c"),
            *transformer
        )

        crop_final = TargetLengthCrop(TARGET_LENGTH)
        final_pointwise = nn.Sequential(
            nn.Linear(channels, channels * 2, 1),
            nn.Dropout(dropout_rate / 8),
            GELU()
        )

        self._trunk = nn.Sequential(
            stem,
            conv_tower,
            transformer,
            crop_final,
            final_pointwise
        )

        self._heads = nn.ModuleDict({
            head: nn.Sequential(
                nn.Linear(channels * 2, head_channels, 1),
                nn.Softplus())
            for head, head_channels in heads_channels.items()
        })

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def forward(self, inputs):
        
        # x = inputs
        #for func in self.trunk:
        #    x = func(x)
        x = self.trunk(inputs)
        return {head: head_module(x) for
                head, head_module in self.heads.items()}


class TargetLengthCrop(nn.Module):

    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2
        return x[:, -trim:trim, :]


class Residual(nn.Module):
    """residuel block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)


def conv_block(in_channels, out_channels, kernel_size=1, **kwargs):
    return nn.Sequential(
        # nn.SyncBatchNorm(in_channels, momentum = 0.1),
        nn.BatchNorm1d(in_channels),
        GELU(),
        nn.Conv1d(in_channels, out_channels,
                  kernel_size, padding="same", **kwargs)
    )


class SoftmaxPooling1D(nn.Module):

    def __init__(self, channels, pool_size=2, w_init_scale=2.0):
        """
        Args:
            channels: number of channels
            pool_size: pooling size
            w_init_scale: scale on the diagonal element.
        """
        super().__init__()
        self._pool_size = pool_size
        self._w_init_scale = w_init_scale
        self._logit_linear = nn.Linear(channels, channels, bias=False)
        self._logit_linear.weight.data.copy_(
            torch.eye(channels) * self._w_init_scale)

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, ("input length must "
                                              "by divisible by pool_size")
        x = rearrange(x, "b c (l p) -> b l p c", p=self._pool_size)
        x = x * F.softmax(self._logit_linear(x), dim=-2)
        x = torch.sum(x, dim=-2)
        return rearrange(x, "b l c -> b c l")


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value = 0,
                   dtype=np.float32) -> np.ndarray:
    """One-hot encode sequence."""
    def to_uint8(string):
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]

FULL_LENGTH = 393_216
SEQLEN = 196_608
SHIFT_AMPLITUDE = 4

class HDF5Dataset(Dataset):

    def __init__(self, hdf5_file_human, hdf5_file_mouse=None, shift_augmentation=False, complementary_chain_augmentation=False, pop_seq=False):
        """
        Custom PyTorch Dataset for reading an HDF5 file.
        
        :param hdf5_file: Path to the HDF5 file.
        :param dataset_name: Name of the dataset inside the HDF5 file to read.
        """
        self.hdf5_file_human = hdf5_file_human
        self.hdf5_file_mouse = hdf5_file_mouse

        self.shift_augmentation = shift_augmentation
        self.complementary_chain_augmentation = complementary_chain_augmentation
        if pop_seq:
            self.key = 'pop_sequence'
        else:
            self.key = 'sequence'
        # Open the HDF5 file and check the shape of the dataset
        with h5py.File(hdf5_file_human, 'r') as hdf:
            self.dataset_shape_human = hdf[self.key].shape

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
            sequence_human = hdf[self.key][idx]
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
        # print(sequence_human.shape)

        if self.hdf5_file_mouse is not None:
            sequence_mouse = sequence_mouse[ind_min:ind_max]
            # sequence_mouse = sequence_mouse # (131072, 4)
            # print(sequence_mouse.shape)

        data_point = {
            'sequence_human': torch.tensor(sequence_human).float(), 
            'target_human': torch.tensor(target_human),            
            'shift': shift,
            'complementary': reverse_strand
        }
        # print(f"Human sequence {sequence_human.shape}")
        if self.hdf5_file_mouse is not None:
            # print(f"Mouse sequence {sequence_mouse.shape}")
            data_point['sequence_mouse'] = torch.tensor(sequence_mouse).float()
            data_point['target_mouse']   = torch.tensor(target_mouse)

        return EasyDict(data_point)


def build_model_and_optimizer(enformer_params, from_checkpoint, _rank, ckpt_dir, _device):
    model = Enformer(**enformer_params)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-3
    )

    if from_checkpoint == "last":
        regex = re.compile("checkpoint_step_(.*).pth")
        ckpt_files = os.listdir(ckpt_dir)
        last_step = max([int(regex.match(file).group(1)) for file in ckpt_files])
        ckpt_path = f"{ckpt_dir}/checkpoint_step_{str(last_step)}.pth" 
        if _rank == 0:
            logger.info(f"Checkpoint restored from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        assert (
            ckpt is not None
            and isinstance(ckpt, dict)
            and 'optimizer_state_dict' in ckpt
            and 'model_state_dict' in ckpt
        )
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(_device)
            elif isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(_device)

        state_dict = ckpt['model_state_dict']
        unwanted_prefix = 'module.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        step = ckpt['step']
        ckpt = None
    
    elif from_checkpoint is False:
        step = 0
        if _rank == 0:
            logger.info(f"No checkpoint was loaded. Training model from scratch...")
    else:
        raise ValueError(f"Only supported values for from_checkpoint are \"last\" or False, got {from_checkpoint=}")
    
    model.to(_device)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    # model = DDP(model, find_unused_parameters = True, broadcast_buffers=False )
    model = DDP(model, find_unused_parameters = True)
    return model, optimizer, step

    
def train_step(batch, optimizer, head):
    model.train()

    optimizer.zero_grad()

    sequences = batch[f'sequence_{head}'].to(device)
    target = batch[f'target_{head}'].to(device)

    outputs = model(sequences)
    loss = criterion(outputs[head], target)
    loss_mn = loss.mean()
    loss_mn.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
    optimizer.step()
    return loss_mn

def val_step(batch, head):
    val_sequences = batch[f'sequence_{head}'].to(device)
    val_target = batch[f'target_{head}'].to(device)
    val_outputs = model(val_sequences)
    val_loss = criterion(val_outputs[head], val_target)
    val_mn = val_loss.mean()
    return val_mn

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

# --------- Main -----------
'''
If ckpt_frequency is an even number, val_frequency should be an odd number
so the validation is alternated btwn mouse and human heads

'''

parser = argparse.ArgumentParser()
parser.add_argument("--train_human_hdf5", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/human_train.h5") 
parser.add_argument("--train_mouse_hdf5", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/mouse_train.h5") 
parser.add_argument("--val_human_hdf5", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/human_validation.h5") 
parser.add_argument("--val_mouse_hdf5", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/mouse_validation.h5") 
parser.add_argument("--ckpt_dir", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/aurora_checkpoints") 
parser.add_argument("--from_checkpoint", default=False) 
parser.add_argument("--max_steps", type=int, default=150000) 
parser.add_argument("--val_frequency", type=int, default=5) 
parser.add_argument("--ckpt_frequency", type=int, default=10)
parser.add_argument("--num_warmup_steps", type=int, default=5000) 
parser.add_argument("--pop_seq", default=False)
args = parser.parse_args()


dataset_train = HDF5Dataset(hdf5_file_human = args.train_human_hdf5,
                                hdf5_file_mouse = args.train_mouse_hdf5,
                                shift_augmentation=True, 
                                complementary_chain_augmentation=True,
                                pop_seq = args.pop_seq)

dataset_val = HDF5Dataset(hdf5_file_human = args.val_human_hdf5,
                                hdf5_file_mouse = args.val_mouse_hdf5,
                                shift_augmentation=True, 
                                complementary_chain_augmentation=True,
                                pop_seq = args.pop_seq)

# sampler will split the full data between GPUs
sampler = DistributedSampler(dataset_train, shuffle = True,  num_replicas=SIZE, rank=RANK, seed=0)
train_loader = DataLoader(dataset_train, sampler = sampler, batch_size = 1)

val_sampler = DistributedSampler(dataset_val, shuffle = False,  num_replicas=SIZE, rank=RANK, seed=0)
val_loader = DataLoader(dataset_val, sampler = val_sampler, batch_size = 1)

enformer_params = dict(channels= 1536, num_heads=8, num_transformer_layers=11, prediction_head="both")
criterion = nn.PoissonNLLLoss(log_input=False, reduction="none")
criterion = criterion.to(device)

# ---Load model, optimizer, checkpoint
model, optimizer, current_step = build_model_and_optimizer(enformer_params, 
                                                           from_checkpoint = args.from_checkpoint,
                                                           ckpt_dir = args.ckpt_dir,
                                                           _device = device, 
                                                           _rank = RANK)
target_learning_rate = 5e-4
num_warmup_steps = args.num_warmup_steps
max_steps = args.max_steps
val_frequency = args.val_frequency
ckpt_freq = args.ckpt_frequency
data_it = iter(train_loader)
val_it = iter(val_loader)

sampler.set_epoch(current_step)
for _ in tqdm(range(max_steps-current_step)):
    current_step += 1

    # ---Warmup learning rate---
    if current_step >= 1:
        lr_frac = min(1.0, current_step / max(1.0, num_warmup_steps))
        lr = target_learning_rate * lr_frac # * SIZE ??
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # ---Train step---
    try:
        batch = next(data_it)
    except StopIteration:
        sampler.set_epoch(current_step)
        data_it = iter(train_loader)
        batch = next(data_it)
        
    if current_step % 2 ==0:
        step_head = 'human'
    else:
        step_head = 'mouse'
    step_loss = train_step(batch, optimizer, step_head)
    dist.all_reduce(step_loss, op=dist.ReduceOp.SUM) # gather loss across gpu nodes
    # ---validation step---
    if current_step % val_frequency == 0:
        try:
            val_batch = next(data_it)
        except StopIteration:
            val_it = iter(val_loader)
            val_batch = next(val_it)
            
        val_loss = val_step(val_batch, step_head)

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM) # gather loss across gpu nodes
        if RANK == 0: # print the loss only in one gpu to avoid more clutter
            
            logger.info(f"Step: {current_step}, loss_{step_head}: {(step_loss.item()/SIZE):.6f}, learning_rate: {lr:.6f}")
            logger.info(f"Step: {current_step}, val_loss_{step_head}: {(val_loss.item()/SIZE):.6f}, learning_rate: {lr:.6f}")

    if current_step % ckpt_freq == 0:
        if RANK == 0:
            save_checkpoint(model=model, optimizer=optimizer, step=current_step, checkpoint_dir=args.ckpt_dir)
            logger.info(f"Saved checkpoint")
torch.distributed.destroy_process_group()