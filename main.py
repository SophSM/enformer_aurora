import os, socket
import utils
import modelUtils
from mpi4py import MPI
import torch
from torch import nn
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
from tqdm import tqdm
import torch.distributed as dist

def run(args):
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)
    
    # ----Set up MPI-----
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


    # ----Init model----
    if args.pop_seq:
        if RANK == 0:
            print(f"Training with population sequences")


    enformer_params = dict(channels= 1536, num_heads=8, num_transformer_layers=11, prediction_head="both")
    criterion = nn.PoissonNLLLoss(log_input=False, reduction="mean")
    criterion = criterion.to(device)

    model, optimizer, global_step = modelUtils.build_model_and_optimizer(enformer_params,
                                             from_checkpoint = args.from_checkpoint,
                                             ckpt_dir = args.ckpt_dir,
                                             _device = device, 
                                             _rank = RANK)

    # ---Init datasets----
    file_paths = {'human':{'train':args.human_train,
                       'val' :args.human_val },
            'mouse':{'train':args.mouse_train,
                     'val':args.mouse_val}}

    human_train_dataset = utils.HDF5Dataset(file_paths, 'human', 'train', shift_augmentation = True, complementary_chain_augmentation=True, pop_seq=args.pop_seq)
    human_val_dataset = utils.HDF5Dataset(file_paths, 'human', 'val', shift_augmentation = True, complementary_chain_augmentation=True, pop_seq=args.pop_seq)
    mouse_train_dataset = utils.HDF5Dataset(file_paths, 'mouse', 'train', shift_augmentation = True, complementary_chain_augmentation=True)
    mouse_val_dataset = utils.HDF5Dataset(file_paths, 'mouse', 'val', shift_augmentation = True, complementary_chain_augmentation=True)


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

    def set_epoch(current_step, organism=None):
        if organism == 'human':
            sampler_human_train.set_epoch(current_step)

        elif organism == 'mouse':
            sampler_mouse_train.set_epoch(current_step)

        else:
            sampler_human_train.set_epoch(current_step)
            sampler_mouse_train.set_epoch(current_step)

    # ---Init training---
    max_steps = args.max_steps
    ckpt_freq = args.ckpt_frequency
    human_train_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
    human_val_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
    mouse_train_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
    mouse_val_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}

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

        train_loss_human, r_train_human_mean, r_train_human_median = utils.train_step(model, ht_x.to(device), ht_y.to(device), criterion, optimizer, head = 'human', cor=cor)
        # gather loss values across ranks
        dist.all_reduce(train_loss_human, op=dist.ReduceOp.SUM)
        train_loss_human = train_loss_human / SIZE
        human_train_stats['loss'].append(train_loss_human.cpu().item())
        if cor:
            if RANK == 0:
                h_t_mean_cor_list = [torch.zeros_like(r_train_human_mean) for _ in range(SIZE)]
                h_t_median_cor_list = [torch.zeros_like(r_train_human_median) for _ in range(SIZE)]
            else: 
                h_t_mean_cor_list=None
                h_t_median_cor_list=None
            dist.gather(r_train_human_mean, gather_list = h_t_mean_cor_list, dst=0)
            dist.gather(r_train_human_median, gather_list = h_t_median_cor_list, dst=0)
        if global_step % args.val_frequency == 0:

            # validation
            try:
                hv_x, hv_y = next(human_val_iter)

            except StopIteration:
                human_val_iter = iter(human_val_loader)
                hv_x, hv_y = next(human_val_iter)

            val_loss_human, r_val_human_mean, r_val_human_median = utils.valid_step(model, hv_x.to(device), hv_y.to(device), criterion, head = 'human', cor=cor)
            # gather loss values across ranks
            dist.all_reduce(val_loss_human, op=dist.ReduceOp.SUM)
            val_loss_human = val_loss_human / SIZE
            if cor:
                if RANK == 0:
                    h_v_mean_cor_list = [torch.zeros_like(r_val_human_mean) for _ in range(SIZE)]
                    h_v_median_cor_list = [torch.zeros_like(r_val_human_median) for _ in range(SIZE)]
                
                else: 
                    h_v_mean_cor_list=None
                    h_v_median_cor_list=None
                dist.gather(r_val_human_mean, gather_list = h_v_mean_cor_list, dst=0)
                dist.gather(r_val_human_median, gather_list = h_v_median_cor_list, dst=0)
        # --mouse step--
        # train
        try:
            mt_x, mt_y = next(mouse_train_iter)
        except StopIteration:
            set_epoch(global_step, 'mouse')
            mouse_train_iter = iter(mouse_train_loader)
            mt_x, mt_y = next(mouse_train_iter)
        train_loss_mouse, r_train_mouse_mean, r_train_mouse_median = utils.train_step(model, mt_x.to(device), mt_y.to(device), criterion,optimizer, head = 'mouse', cor=cor)
        # gather loss values across ranks
        dist.all_reduce(train_loss_mouse, op=dist.ReduceOp.SUM)
        train_loss_mouse = train_loss_mouse / SIZE
        mouse_train_stats['loss'].append(train_loss_mouse.cpu().item())
        
        if cor:
            if RANK == 0:
                m_t_mean_cor_list = [torch.zeros_like(r_train_mouse_mean) for _ in range(SIZE)]
                m_t_median_cor_list = [torch.zeros_like(r_train_mouse_median) for _ in range(SIZE)]
            else: 
                m_t_mean_cor_list=None
                m_t_median_cor_list=None
            dist.gather(r_train_mouse_mean, gather_list = m_t_mean_cor_list, dst=0)    
            dist.gather(r_train_mouse_median, gather_list = m_t_median_cor_list, dst=0)    
        if global_step % args.val_frequency == 0:
            # validation
            try:
                mv_x, mv_y = next(mouse_val_iter)
            except StopIteration:
                mouse_val_iter = iter(mouse_val_loader)
                mv_x, mv_y = next(mouse_val_iter)
            val_loss_mouse, r_val_mouse_mean, r_val_mouse_median = utils.valid_step(model, mv_x.to(device), mv_y.to(device), criterion, head = 'mouse', cor=cor)
            # gather loss values across ranks
            dist.all_reduce(val_loss_mouse, op=dist.ReduceOp.SUM)
            val_loss_mouse = val_loss_mouse / SIZE
            if cor:
                if RANK == 0:
                    m_v_mean_cor_list = [torch.zeros_like(r_val_mouse_mean) for _ in range(SIZE)]
                    m_v_median_cor_list = [torch.zeros_like(r_val_mouse_median) for _ in range(SIZE)]
                else: 
                    m_v_mean_cor_list=None
                    m_v_median_cor_list=None
                dist.gather(r_val_mouse_mean, gather_list = m_v_mean_cor_list, dst=0)
                dist.gather(r_val_mouse_median, gather_list = m_v_median_cor_list, dst=0)


        if RANK == 0:
            
            print(
                f"Step {global_step:<6d} "
                f"| Human train loss: {train_loss_human.item():>8.4f} "
                f"| Mouse train loss: {train_loss_mouse.item():>8.4f} "
                f"| Learning rate: {lr:>10.6f}"
            )
            if cor:

                cor_h_t_mean = torch.nan_to_num(torch.cat(h_t_mean_cor_list), nan=0.0)
                cor_h_t_median = torch.nan_to_num(torch.cat(h_t_median_cor_list), nan=0.0)
                cor_m_t_mean = torch.nan_to_num(torch.cat(m_t_mean_cor_list), nan=0.0)
                cor_m_t_median = torch.nan_to_num(torch.cat(m_t_median_cor_list), nan=0.0)
                human_train_stats['mean_cor'].append(cor_h_t_mean.mean().item())
                human_train_stats['median_cor'].append(utils.quantile_tensor(cor_h_t_median)[2].item())
                mouse_train_stats['mean_cor'].append(cor_m_t_mean.mean().item())
                mouse_train_stats['median_cor'].append(utils.quantile_tensor(cor_m_t_mean)[2].item())
                print(
                    f"  Human train mean PearsonR: {cor_h_t_mean.mean().item()}\n"
                    f"  Mouse train mean PearsonR: {cor_m_t_mean.mean().item()}\n"
                )
            if global_step % args.val_frequency == 0:
                human_val_stats['loss'].append(val_loss_human.cpu().item())
                mouse_val_stats['loss'].append(val_loss_mouse.cpu().item())

                
                print(
                    f"   Human val loss: {val_loss_human.item():>8.4f} "
                    f"   Mouse val loss: {val_loss_mouse.item():>8.4f} "
                )

                if cor:
                    cor_h_v_mean = torch.nan_to_num(torch.cat(h_v_mean_cor_list),  nan=0.0)
                    cor_h_v_median = torch.nan_to_num(torch.cat(h_v_median_cor_list), nan=0.0)
                    cor_m_v_mean = torch.nan_to_num(torch.cat(m_v_mean_cor_list), nan=0.0)
                    cor_m_v_median = torch.nan_to_num(torch.cat(m_v_median_cor_list), nan=0.0)
                    human_val_stats['mean_cor'].append(cor_h_v_mean.mean().item())
                    human_val_stats['median_cor'].append(utils.quantile_tensor(cor_h_v_median)[2].item())
                    mouse_val_stats['mean_cor'].append(cor_m_v_mean.mean().item())
                    mouse_val_stats['median_cor'].append(utils.quantile_tensor(cor_m_v_median)[2].item())
                    print(
                        f"  Human val mean PearsonR: {cor_h_v_mean.mean().item()}\n"
                        f"  Mouse val mean PearsonR: {cor_m_v_mean.mean().item()}"
                    )


            end_time = time.time()
            elapsed = end_time - start_time
            elapsed_times.append(elapsed)
        
        if (global_step > 0 and global_step % ckpt_freq == 0) or global_step == max_steps:
            if RANK == 0:
                modelUtils.save_checkpoint(model=model, optimizer=optimizer, step=global_step, checkpoint_dir=args.ckpt_dir)
                print(f"Saved checkpoint")
                utils.save_stats(args, global_step, elapsed_times,
                    human_train_stats, human_val_stats,
                    mouse_train_stats, mouse_val_stats)

                
            utils.save_losses(args, global_step,
                human_train_stats, human_val_stats,
                mouse_train_stats, mouse_val_stats)
            # ---reset stats--- 
            human_train_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
            human_val_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
            mouse_train_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
            mouse_val_stats = {'loss': [], 'mean_cor': [], 'median_cor':[]}
            elapsed_times = []

    human_train_dataset.close()
    mouse_train_dataset.close()
    human_val_dataset.close()
    mouse_val_dataset.close()
    torch.distributed.destroy_process_group()

def add_arguments(parser):
    parser.add_argument("--max_steps", type=int, default=150000) 
    parser.add_argument("--stats_dir", type=str, default="/flare/GeomicVar/ssalazar/projects/enformer_retraining/out") 
    parser.add_argument("--batch_size", type=int, default=1 )
    parser.add_argument("--val_frequency", type=int, default = 5)
    parser.add_argument("--human_train", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/train_pop_seq.hdf5")
    parser.add_argument("--human_val", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/valid_pop_seq.hdf5")
    parser.add_argument("--mouse_train", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/train_mouse.hdf5")
    parser.add_argument("--mouse_val", type=str, default="/flare/GeomicVar/ssalazar/enformer_training_data/basenji_data_h5/valid_mouse.hdf5")
    parser.add_argument("--cor_frequency", type=int, default=5)
    parser.add_argument("--ckpt_frequency", type=int, default=10)
    parser.add_argument("--ckpt_dir", type=str, default="/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/reference_checkpoints") 
    parser.add_argument("--pop_seq", action="store_true")
    parser.add_argument("--from_checkpoint", default=False) 

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train enformer model")
    add_arguments(parser)
    args = parser.parse_args()

    try:
        run(args)
    except Exception as e:
        print(f"Unexpected error: {e}")
