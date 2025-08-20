
'''
module load frameworks
source /lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale/bin/activate

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale/lib
unset CCL_WORKER_AFFINITY
export OMP_NUM_THREADS=1
export CPU_BIND_SCHEME="--cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79"
cd /flare/GeomicVar/ssalazar/projects/enformer_retraining/scripts

mpiexec -n 12 -ppn 12 ${CPU_BIND_SCHEME} python -u multi_test.py \
--stats_dir "/flare/GeomicVar/ssalazar/projects/enformer_retraining/out" \
--batch_size 4 \
--val_frequency 4 \
--cor_frequency 4 \
--ckpt_frequency 50 \
--max_steps 100 \
--ckpt_dir "/flare/GeomicVar/ssalazar/projects/enformer_retraining/reference_checkpoints2"
'''


from mpi4py import MPI
import os, socket
import torch
import re
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
import time
import h5py
import numpy as np
from scipy.stats import pearsonr
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--max_steps", type=int, default=150000) 
parser.add_argument("--stats_dir", type=str, default="/flare/GeomicVar/ssalazar/projects/enformer_retraining/out") 
parser.add_argument("--batch_size", type=int, default=1 )
parser.add_argument("--val_frequency", type=int, default = 5)
parser.add_argument("--human_train", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/train_pop_seq.hdf5")
parser.add_argument("--human_val", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/val_pop_seq.hdf5")
parser.add_argument("--mouse_train", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/train_mouse.hdf5")
parser.add_argument("--mouse_val", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/valid_mouse.hdf5")
parser.add_argument("--cor_frequency", type=int, default=5)
parser.add_argument("--ckpt_frequency", type=int, default=10)
parser.add_argument("--ckpt_dir", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/reference_checkpoints") 
parser.add_argument("--pop_seq", action="store_true")
parser.add_argument("--from_checkpoint", default=False) 

args = parser.parse_args()

SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)

torch.distributed.init_process_group(backend='ccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))
torch.xpu.set_device(int(LOCAL_RANK))
device = torch.device('xpu')
torch.manual_seed(0)

if args.pop_seq:
    if RANK == 0:
        print(f"Training with population sequences")

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

def build_model_and_optimizer(enformer_params, from_checkpoint, _rank, ckpt_dir, _device):
    model = Enformer(**enformer_params)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-08, 
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
            print(f"Checkpoint restored from {ckpt_path}")
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
            print(f"No checkpoint was loaded. Training model from scratch...")
    else:
        raise ValueError(f"Only supported values for from_checkpoint are \"last\" or False, got {from_checkpoint=}")
    
    model.to(_device)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    model = DDP(model, find_unused_parameters = True)
    return model, optimizer, step

def _reduced_shape(shape, axis):
    '''
    Get the shape of the tensor after reducing over certain axes
    '''
    if axis is None:
        return torch.Size([])
    return torch.Size([d for i, d in enumerate(shape) if i not in axis])

class CorrelationStats(nn.Module):
    '''
    Shared code to compute pearsonR and R2
    '''
    def __init__(self, reduce_axis = None, name = 'pearsonr'):
        super().__init__()
        self._reduce_axis = reduce_axis
        self._shape = None
        self.name = name
    def _initialize(self, input_shape, device=None):
        self._shape = _reduced_shape(input_shape, self._reduce_axis)
        if device is None:
            device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

        def make_buffer():
            return torch.zeros(self._shape, dtype=torch.float32, device=device)
        
        self.register_buffer('_count', make_buffer())
        self.register_buffer('_product_sum', make_buffer())
        self.register_buffer('_true_sum', make_buffer())
        self.register_buffer('_true_squared_sum', make_buffer())
        self.register_buffer('_pred_sum', make_buffer())
        self.register_buffer('_pred_squared_sum', make_buffer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._shape is None:
            self._initialize(y_true.shape, device=y_true.device)
        
        assert y_true.shape == y_pred.shape, "Shapes must match"
        y_true = y_true.float()
        y_pred = y_pred.float()

        reduce_axis = self._reduce_axis
        # PyTorch's sum needs keepdim=False by default
        self._product_sum += y_true.mul(y_pred).sum(dim=reduce_axis)
        self._true_sum += y_true.sum(dim=reduce_axis)
        self._true_squared_sum += y_true.pow(2).sum(dim=reduce_axis)
        self._pred_sum += y_pred.sum(dim=reduce_axis)
        self._pred_squared_sum += y_pred.pow(2).sum(dim=reduce_axis)
        self._count += torch.ones_like(y_true).sum(dim=reduce_axis)
    def result(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def reset_states(self):
        if self._shape is not None:
            for buf_name, buf_val in self._buffers.items():
                self._buffers[buf_name].zero_()

class PearsonR(CorrelationStats):
    """Pearson correlation coefficient."""

    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        super().__init__(reduce_axis=reduce_axis, name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (self._product_sum
                      - true_mean * self._pred_sum
                      - pred_mean * self._true_sum
                      + self._count * true_mean * pred_mean)

        true_var = self._true_squared_sum - self._count * true_mean.pow(2)
        pred_var = self._pred_squared_sum - self._count * pred_mean.pow(2)
        tp_var = (true_var.sqrt() * pred_var.sqrt()).clamp(min=1e-08)
        correlation = covariance / tp_var
        return correlation

class R2(CorrelationStats):
    """R-squared (fraction of explained variance)."""

    def __init__(self, reduce_axis=None, name='R2'):
        super().__init__(reduce_axis=reduce_axis, name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        total = self._true_squared_sum - self._count * true_mean.pow(2)
        residuals = (self._pred_squared_sum - 2 * self._product_sum
                     + self._true_squared_sum)
        return torch.ones_like(residuals) - residuals / total
    
class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def to(self, device):
        for metric in self._metrics.values():
            metric.to(device)
        return self
    
    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}

    def reset_states(self):
        for metric in self._metrics.values():
            metric.reset_states()

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



enformer_params = dict(channels= 1536, num_heads=8, num_transformer_layers=11, prediction_head="both")
criterion = nn.PoissonNLLLoss(log_input=False, reduction="mean")
criterion = criterion.to(device)
model, optimizer, global_step = build_model_and_optimizer(enformer_params,
                                             from_checkpoint = args.from_checkpoint,
                                             ckpt_dir = args.ckpt_dir,
                                             _device = device, 
                                             _rank = RANK)

file_paths = {'human':{'train':args.human_train,
                       'val' :args.human_val },
            'mouse':{'train':args.mouse_train,
                     'val':args.mouse_val}}

human_train_dataset = HDF5Dataset(file_paths, 'human', 'train', shift_augmentation = True, complementary_chain_augmentation=True, pop_seq=args.pop_seq)
human_val_dataset = HDF5Dataset(file_paths, 'human', 'val', shift_augmentation = True, complementary_chain_augmentation=True, pop_seq=args.pop_seq)
mouse_train_dataset = HDF5Dataset(file_paths, 'mouse', 'train', shift_augmentation = True, complementary_chain_augmentation=True)
mouse_val_dataset = HDF5Dataset(file_paths, 'mouse', 'val', shift_augmentation = True, complementary_chain_augmentation=True)

sampler_human_train = DistributedSampler(human_train_dataset, shuffle = True, num_replicas=SIZE, rank=RANK, seed=0)
sampler_human_train.set_epoch(global_step)
human_train_loader = DataLoader(human_train_dataset, sampler = sampler_human_train, batch_size = args.batch_size)

sampler_human_val = DistributedSampler(human_val_dataset, shuffle = False, num_replicas=SIZE, rank=RANK, seed=0)
sampler_human_val.set_epoch(global_step)
human_val_loader = DataLoader(human_val_dataset, sampler = sampler_human_val, batch_size = args.batch_size)

sampler_mouse_train = DistributedSampler(mouse_train_dataset, shuffle = True, num_replicas=SIZE, rank=RANK, seed=0)
mouse_train_loader = DataLoader(mouse_train_dataset, sampler = sampler_mouse_train, batch_size = args.batch_size)

sampler_mouse_val = DistributedSampler(mouse_val_dataset, shuffle = False, num_replicas=SIZE, rank=RANK, seed=0)
mouse_val_loader = DataLoader(mouse_val_dataset, sampler = sampler_mouse_val, batch_size = args.batch_size)


human_train_iter = iter(human_train_loader)
human_val_iter = iter(human_val_loader)
mouse_train_iter = iter(mouse_train_loader)
mouse_val_iter = iter(mouse_val_loader)


def train_step(model, x, y, criterion, head, cor=False):
    
    model.train()
    out = model(x)
    loss = criterion(out[head], y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if cor:
        metrics = MetricDict({
        "pearson": PearsonR(reduce_axis=(0,1))
        }).to(device)
        metrics.update_state(out[head], y)
        res = metrics.result()
        # res['pearson'] has shape (n_tracks,), then I take the mean over the tracks
        # r = res['pearson'].detach().cpu().numpy().mean()\
        r = res['pearson'].detach().cpu().mean().unsqueeze(0)
    else: r = -1
    return loss, r

def valid_step(model, x, y, criterion, head, cor=False):

    model.eval()
    with torch.no_grad():
        out = model(x) 
        loss = criterion(out[head], y)
        if cor:
            metrics = MetricDict({
            "pearson": PearsonR(reduce_axis=(0,1))
            }).to(device)
            metrics = MetricDict({
            "pearson": PearsonR(reduce_axis=(0,1))
            }).to(device)
            metrics.update_state(out[head], y)
            res = metrics.result()
            # res['pearson'] has shape (n_tracks,), then I take the mean over the tracks
            # r = res['pearson'].detach().cpu().numpy().mean()
            r = res['pearson'].detach().cpu().mean().unsqueeze(0)
        else: r = -1
    return loss, r

def set_epoch(current_step, organism=None):
    if organism == 'human':
        sampler_human_train.set_epoch(current_step)

    elif organism == 'mouse':
        sampler_mouse_train.set_epoch(current_step)

    else:
        sampler_human_train.set_epoch(current_step)
        sampler_mouse_train.set_epoch(current_step)

max_steps = args.max_steps
ckpt_freq = args.ckpt_frequency
human_train_stats = {'loss': [], 'cor': []}
human_val_stats = {'loss': [], 'cor': []}
mouse_train_stats = {'loss': [], 'cor': []}
mouse_val_stats = {'loss': [], 'cor': []}

elapsed_times = []
num_warmup_steps = 5000
target_learning_rate = 0.0005
lr = 1e-08
set_epoch(global_step) # set epoch for all samplers


for _ in tqdm(range(max_steps-global_step)):
    if RANK == 0:
        start_time = time.time()
    global_step += 1
    # --Update learning rate--
    if global_step > 1:
        lr_frac = min(1.0, global_step / max(1.0, num_warmup_steps))
        lr = target_learning_rate * lr_frac # * SIZE ??
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if global_step % args.cor_frequency == 0:
        cor = True
    else: cor=False
    
    # --human step--
    # train
    try:
        ht_x, ht_y = next(human_train_iter)
    except StopIteration:
        set_epoch(global_step, 'human')
        human_train_iter = iter(human_train_loader)
        ht_x, ht_y = next(human_train_iter)

    train_loss_human, r_train_human = train_step(model, ht_x.to(device), ht_y.to(device), criterion, head = 'human', cor=cor)
    # gather loss values across ranks
    dist.all_reduce(train_loss_human, op=dist.ReduceOp.SUM)
    train_loss_human = train_loss_human / SIZE
    human_train_stats['loss'].append(train_loss_human.cpu().item())
    if cor:
        if RANK == 0:
            h_t_cor_list = [torch.zeros_like(r_train_human) for _ in range(SIZE)]
        else: h_t_cor_list=None
        dist.gather(r_train_human, gather_list = h_t_cor_list, dst=0)
        
    if global_step % args.val_frequency == 0:

        # validation
        try:
            hv_x, hv_y = next(human_val_iter)

        except StopIteration:
            human_val_iter = iter(human_val_loader)
            hv_x, hv_y = next(human_val_iter)

        val_loss_human, r_val_human = valid_step(model, ht_x.to(device), ht_y.to(device), criterion, head = 'human', cor=cor)
        # gather loss values across ranks
        dist.all_reduce(val_loss_human, op=dist.ReduceOp.SUM)
        val_loss_human = val_loss_human / SIZE
        if cor:
            if RANK == 0:
                h_v_cor_list = [torch.zeros_like(r_val_human) for _ in range(SIZE)]
            else: h_v_cor_list=None
            dist.gather(r_val_human, gather_list = h_v_cor_list, dst=0)
    # --mouse step--
    # train
    try:
        mt_x, mt_y = next(mouse_train_iter)
    except StopIteration:
        set_epoch(global_step, 'mouse')
        mouse_train_iter = iter(mouse_train_loader)
        mt_x, mt_y = next(mouse_train_iter)
    train_loss_mouse, r_train_mouse = train_step(model, mt_x.to(device), mt_y.to(device), criterion, head = 'mouse', cor=cor)
    # gather loss values across ranks
    dist.all_reduce(train_loss_mouse, op=dist.ReduceOp.SUM)
    train_loss_mouse = train_loss_mouse / SIZE
    mouse_train_stats['loss'].append(train_loss_mouse.cpu().item())
    
    if cor:
        if RANK == 0:
            m_t_cor_list = [torch.zeros_like(r_train_mouse) for _ in range(SIZE)]
        else: m_t_cor_list=None
        dist.gather(r_train_mouse, gather_list = m_t_cor_list, dst=0)    
    if global_step % args.val_frequency == 0:
        # validation
        try:
            mv_x, mv_y = next(mouse_val_iter)
        except StopIteration:
            mouse_val_iter = iter(mouse_val_loader)
            mv_x, mv_y = next(mouse_val_iter)
        val_loss_mouse, r_val_mouse = valid_step(model, mv_x.to(device), mv_y.to(device), criterion, head = 'mouse', cor=cor)
        # gather loss values across ranks
        dist.all_reduce(val_loss_mouse, op=dist.ReduceOp.SUM)
        val_loss_mouse = val_loss_mouse / SIZE
        if cor:
            if RANK == 0:
                m_v_cor_list = [torch.zeros_like(r_val_mouse) for _ in range(SIZE)]
            else: m_v_cor_list=None
            dist.gather(r_val_mouse, gather_list = m_v_cor_list, dst=0)


    if RANK == 0:
        
        print(
            f"Step {global_step:<6d} "
            f"| Human train loss: {train_loss_human.item():>8.4f} "
            f"| Mouse train loss: {train_loss_mouse.item():>8.4f} "
            f"| Learning rate: {lr:>10.6f}"
        )
        if cor:
            cor_h_t = torch.cat(h_t_cor_list)
            cor_m_t = torch.cat(m_t_cor_list) # should have SIZE elements
            human_train_stats['cor'].append(cor_h_t.mean().item())
            mouse_train_stats['cor'].append(cor_m_t.mean().item())
            print(
                f"  Human train PearsonR: {cor_h_t.mean().item()}\n"
                f"  Mouse train PearsonR: {cor_m_t.mean().item()}"
            )
        if global_step % args.val_frequency == 0:
            human_val_stats['loss'].append(val_loss_human.cpu().item())
            mouse_val_stats['loss'].append(val_loss_mouse.cpu().item())

            
            print(
                f"   Human val loss: {val_loss_human.item():>8.4f} "
                f"   Mouse val loss: {val_loss_mouse.item():>8.4f} "
            )

            if cor:
                cor_h_v = torch.cat(h_v_cor_list)
                cor_m_v = torch.cat(m_v_cor_list) # should have SIZE elements
                human_val_stats['cor'].append(cor_h_v.mean().item())
                mouse_val_stats['cor'].append(cor_m_v.mean().item())
                print(
                    f"  Human val PearsonR: {cor_h_v.mean().item()}\n"
                    f"  Mouse val PearsonR: {cor_m_v.mean().item()}"
                )
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_times.append(elapsed)
    
    if global_step % ckpt_freq == 0:
        if RANK == 0:
            save_checkpoint(model=model, optimizer=optimizer, step=global_step, checkpoint_dir=args.ckpt_dir)
            print(f"Saved checkpoint")

if RANK == 0:
    np.save(f'{args.stats_dir}/time_{global_step}.npy', elapsed_times)
    np.save(f'{args.stats_dir}/human_train_cor_{global_step}.npy', human_train_stats['cor'])
    np.save(f'{args.stats_dir}/human_val_cor_{global_step}.npy', human_val_stats['cor'])
    np.save(f'{args.stats_dir}/mouse_train_cor_{global_step}.npy', mouse_train_stats['cor'])
    np.save(f'{args.stats_dir}/mouse_val_cor_{global_step}.npy', mouse_val_stats['cor'])

np.save(f'{args.stats_dir}/human_train_loss_{global_step}.npy', human_train_stats['loss'])
np.save(f'{args.stats_dir}/human_val_loss_{global_step}.npy', human_val_stats['loss'])
np.save(f'{args.stats_dir}/mouse_train_loss_{global_step}.npy', mouse_train_stats['loss'])
np.save(f'{args.stats_dir}/mouse_val_loss_{global_step}.npy', mouse_val_stats['loss'])

human_train_dataset.close()
mouse_train_dataset.close()
human_val_dataset.close()
mouse_val_dataset.close()
torch.distributed.destroy_process_group()

