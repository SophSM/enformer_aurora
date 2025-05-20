import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
from mpi4py import MPI
import os, socket
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.model import Enformer
from utils.dataset import get_datasets
import re
from tqdm import tqdm
import argparse
import logging



def init_distributed():
    '''
    Initialize the multi-GPU parallelization
    '''
    _size = MPI.COMM_WORLD.Get_size()
    _rank = MPI.COMM_WORLD.Get_rank()
    _local_rank = os.environ.get('PALS_LOCAL_RANKID')
    os.environ['RANK'] = str(_rank)
    os.environ['WORLD_SIZE'] = str(_size)
    _master_addr = socket.gethostname() if _rank == 0 else None
    _master_addr = MPI.COMM_WORLD.bcast(_master_addr, root=0)
    os.environ['MASTER_ADDR'] = f"{_master_addr}.hsn.cm.aurora.alcf.anl.gov"
    os.environ['MASTER_PORT'] = str(2345)
    # print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")
    return _size, _rank, _local_rank



def build_model_and_optimizer(enformer_params, from_checkpoint, ckpt_dir, _device, _rank):
    model = Enformer(**enformer_params)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0, 
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
            # Check if the state is a tensor and move it to the correct device
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
        
        ckpt = None  # free up memory
    
    elif from_checkpoint is False:
        step = 0
        if _rank == 0:
            print(f"No checkpoint was loaded. Training model from scratch...")
    else:
        raise ValueError(f"Only supported values for from_checkpoint are \"last\" or False, got {from_checkpoint=}")

    model.to(_device)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    # model = DDP(model, find_unused_parameters = True, broadcast_buffers=False )
    model = DDP(model, find_unused_parameters = True)
    return model, optimizer, step

def save_checkpoint(model, optimizer, step, checkpoint_dir):

    """Saves model, optimizer, and scaler states."""
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
    print(f"Checkpoint saved at {checkpoint_path}")

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, sampler,val_sampler, device, checkpoint_dir, _rank, log_freq=2, checkpoint_freq=2, gradient_clip=0.2):
        self.model = model
        self.rank = _rank
        self.sampler = sampler
        self.val_sampler = val_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = optimizer
        self.device = device
        
        self.current_step = 0 # initialize class step counter on zero 
        self._log_freq = log_freq
        self._checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir 

        self.gradient_clip = gradient_clip
        self.initialize()
    
    def initialize(self):
        # runs only once
        self.iter = iter(self.train_dataloader)
        self.val_iter = iter(self.val_dataloader)
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="none")

    def train_step(self, head):
        # forward and backward passes
        try:
            batch = next(self.iter)
        except StopIteration: # if all batches have been consumed, reset iterator
            self.sampler.set_epoch(self.current_step)
            self.iter = iter(self.train_dataloader)
            batch = next(self.iter)
        
        self.optimizer.zero_grad()
        sequences = batch[f'sequence_{head}'].to(self.device)
        target = batch[f'target_{head}'].to(self.device)
        region = batch[f'region_{head}'].to(self.device)

        outputs = self.model(sequences) 
        loss = self.criterion(outputs[head], target)
            
        loss_mn = loss.mean()
        if self.rank == 0:
            print(region)
            print(target)
        loss_mn.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
        return loss_mn
    
    def val_step(self, head):
        try:
            batch = next(self.val_iter)
        except StopIteration: # if all batches have been consumed, reset iterator
            self.val_sampler.set_epoch(self.current_step)
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)

        val_sequences = batch[f'sequence_{head}'].to(self.device)
        val_target = batch[f'target_{head}'].to(self.device)

        val_outputs = self.model(val_sequences)
        val_loss = self.criterion(val_outputs[head], val_target)
        return val_loss

    def train_step_end(self):
        
        if self.rank == 0 and (self.current_step % self._checkpoint_freq) == 0:
            save_checkpoint(self.model, self.optimizer, self.current_step, self.checkpoint_dir)
            print(f"Checkpoint saved")
        self.current_step += 1

    def set_step(self, step):
        self.current_step = step



def main(args):
    '''
    Arguments
        from_checkpoint: start training from a checkpoint
        ckpt_dir: directory of checkpoints
        human_train: human training data hdf5 file path
        mouse_train: mouse training data hdf5 file path
        human_val: human validation data hdf5 file path
        mouse_val: mouse validation data hdf5 file path
        batch_size: number of sequences per batch
        max_steps: maximum number of training steps
        num_warmup_steps: number of steps to warmup the training
        checkpoint_freq: 
        log_path: path to log file
    '''
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(args.log_path)
        ]
    )

    logger = logging.getLogger()


    # ---Set up multi-GPU resource distribution---
    SIZE, RANK, LOCAL_RANK = init_distributed()
    if RANK == 0:
        logger.info("Training started")
    torch.distributed.init_process_group(backend='ccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

    # torch.xpu.set_device(int(LOCAL_RANK)) # pin GPU to local rank
    # device = torch.device('xpu')
    device = torch.device(f"xpu:{LOCAL_RANK}")
    torch.manual_seed(0)

    # ---Load enformer parameters and optimizer---
    enformer_params = dict(channels= 1536, num_heads=8, num_transformer_layers=11, prediction_head="both")
    from_checkpoint = args.from_checkpoint if args.from_checkpoint is not None else False
    model, optimizer, step = build_model_and_optimizer(enformer_params, from_checkpoint, args.ckpt_dir, device, RANK)

    # ---Load data---
    dataset_train, dataset_val = get_datasets(train_human = args.human_train, 
                                              val_human = args.human_val, 
                                              train_mouse = args.mouse_train, 
                                              val_mouse = args.mouse_val,
                                              pop_seq = args.pop_seq)

    # sampler will split the full data between GPUs
    sampler = DistributedSampler(dataset_train, shuffle = True,  num_replicas=SIZE, rank=RANK, seed=0)
    sampler_val = DistributedSampler(dataset_val, shuffle = False,  num_replicas=SIZE, rank=RANK, seed=0)
    # each GPU will recieve batch_size samples at a time
    train_loader = DataLoader(dataset_train, sampler = sampler, batch_size = args.batch_size, num_workers = 1)
    
    if RANK == 0:
        logger.info(f"Length of train loader {len(train_loader)}")

    val_loader = DataLoader(dataset_val, sampler = sampler_val, batch_size = args.batch_size)
    
    # ---Train loop---

    trainer = Trainer(model = model, train_dataloader = train_loader,
                      val_dataloader = val_loader, optimizer = optimizer, 
                      device = device, checkpoint_dir=args.ckpt_dir, _rank = RANK,
                      checkpoint_freq=args.checkpoint_freq, sampler=sampler,val_sampler=sampler_val)
  
    num_warmup_steps = -1 if args.num_warmup_steps is None else args.num_warmup_steps
    target_learning_rate = 5e-4
    
    trainer.set_step(step + 1) # set step from checkpoint if loaded, else is 0

    for _ in tqdm(range(args.max_steps - step)):
        model.train()
        if RANK == 0:
            logger.info(f"Step: {trainer.current_step}")
        
        if trainer.current_step < num_warmup_steps:
            learning_rate_frac = min(1.0, trainer.current_step / max(1.0, num_warmup_steps))                
            current_lr = target_learning_rate * learning_rate_frac
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        if trainer.current_step % 2 == 0:   
            head = 'human'
        else:
            head = 'mouse'
        loss = trainer.train_step(head)  # will reset iterator if samples are exhausted
        
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) # gather loss across gpu nodes

        if RANK == 0:
            logger.info(f"Step: {trainer.current_step}, train_loss_{head}: {loss.item()/SIZE:.6f}, lr: {current_lr}")
        trainer.train_step_end() # does current_step += 1, saves if frequency reached
        
        # validation step
        if trainer.current_step % args.val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss = trainer.val_step(head)
                
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM) # gather loss across gpu nodes
        
            if RANK == 0: # print the loss only in one gpu to avoid more clutter
                logger.info(f"Step: {trainer.current_step}, val_loss_{head}: {val_loss.item()/SIZE:.6f}, learning_rate: {current_lr:.6f}")
        
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_warmup_steps", type=int, default=5000)    
    parser.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=150000)
    parser.add_argument("--val_freq", default=100)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--checkpoint_freq", type = int, default=1)
    parser.add_argument("--from-checkpoint", "--from_checkpoint", "--from-ckpt", "--from_ckpt", dest="from_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt-dir", "--checkpoint-dir", "--ckpt_dir", "--checkpoint_dir", dest="ckpt_dir", 
                        type=str, default="/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/aurora_checkpoints")                        
    parser.add_argument("--human_train", default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/human_train.h5")
    parser.add_argument("--mouse_train", default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/mouse_train.h5")
    parser.add_argument("--human_val", default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/human_validation.h5")
    parser.add_argument("--mouse_val", default="/lus/flare/projects/GeomicVar/ssalazar/enformer_training_data/full_393216bp/mouse_validation.h5")
    parser.add_argument("--pop_seq", default=False)
    parser.add_argument("--log_path", default="/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/logs/training_may20_1530.log")
    args = parser.parse_args()
    main(args)
