from typing import Optional
import ezpz
import torch

try:
    import intel_extension_for_pytorch as ipex

    HAS_IPEX = True
except Exception:
    ipex = None
    HAS_IPEX = False
try:
    import oneccl_bindings_for_pytorch as torch_ccl
except Exception:
    torch_ccl = None
    HAS_CCL = False

# from mpi4py import MPI
from pathlib import Path
import torch.distributed as dist


from torch import nn

# from torch.utils.data import random_split, Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast
import argparse
import os

# import socket
import time
import logging
import re

try:
    import mlflow
except Exception:
    mlflow = None
    logging.warning("MLflow is not available")


from model import enformer
from dataset import get_dataloaders

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

logger = ezpz.get_logger(__name__)

torch.set_float32_matmul_precision("medium")


# This is needed if using the --compile-model flag (as torch.compile requires a directory to store temporary artifacts)
os.environ["TMPDIR"] = Path(os.getcwd()).joinpath("tmp").as_posix()

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def average_loss_across_gpus(loss, world_size):

    if is_distributed():
        # Convert loss to tensor and move it to the correct device
        loss_tensor = torch.tensor(loss).to(ezpz.get_torch_device_type())
        logger.info("Starting all_reduce operation...")

        # torch.cuda.synchronize()  # Synchronize before the all_reduce
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        logger.info("The all_reduce finished")

        # Synchronize to ensure the operation finishes
        # torch.cuda.synchronize()
        logger.info("The cuda synchronize finished")

        avg_loss = loss_tensor.cpu().item() / world_size
        logger.info(f"Finished all_reduce, average loss: {avg_loss:.4f}")
    else:
        avg_loss = loss
    return avg_loss


def save_checkpoint(model, optimizer, scaler, epoch, checkpoint_dir):
    """Saves model, optimizer, and scaler states."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")


#### NOT WORKING! gather_all and all_reduce times out.
def validate(model, dataloader, criterion, world_size):

    model.eval()
    total_loss = 0.0
    num_batches = 0
    device_type = ezpz.get_torch_device_type()

    with torch.no_grad():
        for inputs, targets in dataloader:
            # inputs = inputs.cuda(non_blocking=True)
            # targets = targets.cuda(non_blocking=True)
            inputs = inputs.to(device_type, non_blocking=True)
            targets = targets.to(device_type, non_blocking=True)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

    # Average the loss across all batches on each GPU
    avg_loss = total_loss / num_batches

    # Average the loss across all GPUs
    avg_loss = average_loss_across_gpus(avg_loss, world_size)

    return avg_loss


####################################################################################################


class Trainer:

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        device,
        checkpoint_dir,
        log_freq=2,
        checkpoint_freq=1,
        precision: str = "half",
        gradient_clip=0.2,
        max_epochs=10,
        dtype: Optional[str | torch.dtype] = None,
    ):

        self.model = model
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.rank = ezpz.get_rank()
        self.world_size = ezpz.get_world_size()
        self.device_id = f"{ezpz.get_torch_device_type()}:{ezpz.get_local_rank()}"
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

        self.species = (
            ["human", "mouse"]
            if self.model.module.prediction_head == "both"
            else [self.model.module.prediction_head]
        )

        self.optimizer = optimizer
        self.device = device

        self.n_step = 0
        self.nval_step = 0
        self.current_epoch = 0
        self.max_epochs = max_epochs

        self._log_freq = log_freq
        self._checkpoint_freq = checkpoint_freq

        self.precision = precision
        assert self.precision in ["half", "single"], (
            f"Invalid precision: {self.precision}"
        )

        self.checkpoint_dir = checkpoint_dir

        self.gradient_clip = gradient_clip
        self.initialize()

    def initialize(self):

        self.iter = iter(self.train_dataloader)
        self.val_iter = iter(self.val_dataloader)
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="none")
        # self.scaler = GradScaler()
        self._train_outputs = []
        self._val_outputs = []

    def train_step(self):

        if self.precision == "half":
            return self.train_step_16()
        elif self.precision == "single":
            return self.train_step_32()

    def train_step_16(self):

        self.n_step += 1

        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.train_dataloader)
            batch = next(self.iter)

        for head in self.species:

            sequences = batch[f"sequence_{head}"].to(self.device)
            targets = batch[f"target_{head}"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=ezpz.get_torch_device_type(), dtype=self.dtype):

                pred = self.model(sequences)
                loss = self.criterion(pred[head], targets)
                loss_mn = loss.mean()
                # self.scaler.scale(loss_mn).backward()
                loss.backward()

            if (self.n_step % self._log_freq) == 0 and ezpz.get_rank() == 0:
                logger.info(
                    f"step: {self.n_step}, head: {head}; loss: {loss_mn.item():03f}"
                )

            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            self.optimizer.step()

    def train_step_32(self):

        self.n_step += 1

        try:
            batch = next(self.iter)
        except StopIteration:
            self.train_epoch_end()
            self.iter = iter(self.train_dataloader)
            batch = next(self.iter)

        loss_mn = 0.0
        for head in self.species:

            sequences = batch[f"sequence_{head}"].to(self.device)
            targets = batch[f"target_{head}"].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(sequences)
            loss = self.criterion(pred[head], targets)
            loss_mn = loss.mean()
            loss_mn.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self._train_outputs.append(loss_mn)

            if (self.rank == 0) and (self.n_step % self._log_freq) == 0:
                # if (self.n_step % self._log_freq) == 0:
                # loss_value = average_loss_across_gpus(loss_mn.item(), world_size=dist.get_world_size())
                # logger.info(f"step: {self.n_step}, head: {head}; loss: {loss_value:03f}")
                logger.info(
                    f"step: {self.n_step}, head: {head}; loss: {loss_mn.item():03f}"
                )

            self.optimizer.step()

        try:
            return loss_mn.item()
        except Exception:
            return loss_mn

    def train_epoch_end(self):
        if self.rank == 0 and (self.current_epoch % self._checkpoint_freq) == 0:
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scaler,
                self.current_epoch,
                self.checkpoint_dir,
            )

        # sum(self._train_outputs.append(loss_mn)) / dist.get_world_size()

        self.n_step = 0
        self.current_epoch += 1

    def validate(self):
        world_size = dist.get_world_size()
        val_loss = validate(self.model, self.val_dataloader, self.criterion, world_size)
        return val_loss

    def validation_step(self):

        self.nval_step += 1
        try:
            batch = next(self.val_iter)
        except StopIteration:
            self.val_epoch_end()
            self.val_iter = iter(self.val_dataloader)
            batch = next(self.val_iter)

        sequences = batch["sequence"].to(self.device)
        targets = batch["target"].to(self.device)

        for head in ["human"]:

            with torch.no_grad():
                pred = self.model(sequences)
                loss = self.criterion(pred[head], targets)
                loss_mn = loss.mean()

                if (self.nval_step % self._log_freq) == 0 and self.rank == 0:
                    logger.info(
                        f"step: {self.nval_step}; head: {head}; loss: {loss_mn.item():03f}"
                    )

        return loss_mn.item()

    def val_epoch_end(self):
        self.nval_step = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch


def configure_optimizer(model, optimizer_params):

    algorithm = optimizer_params["algorithm"]
    algorithm = torch.optim.__dict__[algorithm]
    optimizer = algorithm(model.parameters(), **optimizer_params["parameters"])
    return optimizer


# ModelOptimizerPair = tuple[DDP | torch.nn.Module, torch.optim.Optimizer]

# ModelOptimizerPair = tuple[DistributedDataParallel | Unknown | Enformer, Unknown | Any, Unknown | Literal[0]]


def build_model_and_optimizer(
    enformer_params, from_checkpoint
):  #  -> ModelOptimizerPair:
    rank = ezpz.get_rank()
    device_id = f"{ezpz.get_torch_device_type()}:{ezpz.get_local_rank()}"

    OPTIMIZER_PARAMS = {
        "algorithm": "Adam",
        "parameters": {
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-3,
        },
    }

    model = enformer.Enformer(**enformer_params)

    optimizer = configure_optimizer(model, OPTIMIZER_PARAMS)
    if ipex is not None and HAS_IPEX:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    if from_checkpoint == "last":

        # Start from last checkpoint
        regex = re.compile("checkpoint_epoch_(.*).pth")
        ckpt_files = os.listdir(args.ckpt_dir)
        last_epoch = max([int(regex.match(file).group(1)) for file in ckpt_files])
        ckpt_path = f"{args.ckpt_dir}/checkpoint_epoch_{str(last_epoch)}.pth"

        logger.info(f"Checkpoint restored from {ckpt_path}")
        logger.info(model)

        # ckpt = torch.load(ckpt_path, map_location=DEVICE_ID)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        assert (
            ckpt is not None
            and isinstance(ckpt, dict)
            and "optimizer_state_dict" in ckpt
            and "model_state_dict" in ckpt
        )

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            # Check if the state is a tensor and move it to the correct device
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device_id)
            elif isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id)

        state_dict = ckpt["model_state_dict"]
        unwanted_prefix = "_orig_mod.module."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        epoch = ckpt["epoch"]

        ckpt = None  # free up memory

    elif from_checkpoint is False:
        epoch = 0
        if rank == 0:
            logging.info("No checkpoint was loaded. Training model from scratch...")
    else:
        raise ValueError(
            f'Only supported values for from_checkpoint are "last" or False, got {from_checkpoint=}'
        )

    model.to(device_id)

    logging.debug(f"{model=}")

    if backend.lower() == "ddp":
        if ezpz.get_world_size() > 1:
            model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
    elif backend.lower() in ("ds", "deepspeed"):
        import deepspeed
        import argparse

        parser = argparse.ArgumentParser(description="My training script.")
        parser.add_argument(
            "--local_rank",
            required=False,
            type=int,
            default=-1,
            help="local rank passed from distributed launcher",
        )
        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)
        cmd_args = parser.parse_args()
        logging.info(f"{cmd_args=}")
        model, optimizer, *_ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            optimizer=optimizer,
        )

    return model, optimizer, epoch


def main(args):

    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()
    device_type = ezpz.get_torch_device_type()
    if ezpz.get_rank() == 0 and mlflow is not None:
        mlflow.set_tracking_uri(uri=args.mlflow_uri)
        mlflow.set_experiment(experiment_name=args.mlflow_expname)
        mlflow.start_run()
        mlflow.log_param("n_gpus", world_size)
        mlflow.log_param("parallelization_backend", backend)

    # 3*2**9 == 1536
    enformer_params = dict(
        channels=3 * 2**9,
        num_heads=8,
        num_transformer_layers=11,
        prediction_head="human",
    )
    # enformer_params = dict(channels=3*2**9, num_heads=8, num_transformer_layers=11, prediction_head="both")

    args.from_checkpoint = (
        args.from_checkpoint if args.from_checkpoint is not None else False
    )

    model, optimizer, epoch = build_model_and_optimizer(
        enformer_params, from_checkpoint=args.from_checkpoint
    )

    if args.compile_model:
        model = torch.compile(model)

    distributed_sampler_cfg = {
        "num_replicas": world_size,
        "rank": rank,
    }

    split_lengths = args.split_lengths  # [32000, 1992]

    dataloaders = get_dataloaders(
        sampler_cfg=distributed_sampler_cfg,
        batch_size=args.batch_size,
        split_lengths=split_lengths,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # from torch.profiler import profile, record_function, ProfilerActivity

    dtype = torch.get_default_dtype() if args.dtype is None else args.dtype
    trainer = Trainer(
        model,
        dataloaders,
        optimizer,
        device=device_type,
        dtype=dtype,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.ckpt_dir,
        precision="half",
    )

    trainer.set_epoch(epoch + 1)
    args.num_warmup_steps = (
        -1 if args.num_warmup_steps is None else args.num_warmup_steps
    )
    target_learning_rate = 5e-4

    for n_epoch in range(args.max_epochs):
        if rank == 0:
            logger.info(f"Epoch: {n_epoch}")

        model.train()
        start_time = time.time()
        previous_epoch = 0

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #    with record_function("model_inference"):
        #        loss_trn = trainer.train_step()

        # logger.info(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        def set_lr_on_ramp(optimizer, trainer, num_warmup_steps):
            if trainer.n_step <= args.num_warmup_steps:
                learning_rate_frac = min(
                    1.0, trainer.n_step / max(1.0, args.num_warmup_steps)
                )
                current_lr = target_learning_rate * learning_rate_frac
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            return current_lr

        while True:

            # LR warmup
            if trainer.n_step <= args.num_warmup_steps:
                learning_rate_frac = min(
                    1.0, trainer.n_step / max(1.0, args.num_warmup_steps)
                )
                current_lr = target_learning_rate * learning_rate_frac

                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            loss_trn = trainer.train_step()

            end_time = time.time()
            step_time = end_time - start_time

            if rank == 0 and mlflow is not None:
                mlflow.log_metric("training_step_time", step_time)
                mlflow.log_metric("training_loss", loss_trn)

            if previous_epoch != trainer.current_epoch:

                if rank == 0:
                    logging.info(f"Epoch {trainer.current_epoch}")
                    previous_epoch = trainer.current_epoch
                    start_time = time.time()

                # model.eval()

                # loss_val = trainer.validation_step()
                # end_time = time.time()  # end time
                # step_time = end_time - start_time  # time per step

                # if RANK == 0:
                # mlflow.log_metric("val_step_time", step_time)
                # mlflow.log_metric("val_loss", loss_val)

        trainer.val_epoch_end()
        torch.distributed.destroy_process_group()

    if rank == 0 and mlflow is not None:
        mlflow.end_run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", dest="backend", type=str, default="DDP")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=2)
    parser.add_argument("--num_warmup_steps", type=int, default=None)

    parser.add_argument(
        "--num_workers",
        "--n_workers",
        "--num-workers",
        "--n-workers",
        dest="num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_epochs", "--max-epochs", dest="max_epochs", type=int, default=1000
    )

    parser.add_argument(
        "--mlflow_uri",
        "--mlflow-uri",
        type=str,
        default="mlruns-enformer-test-warmup-16-11-2024",
    )
    parser.add_argument(
        "--mlflow_experiment",
        "--mlflow-experiment",
        "--mlflow-experiment-name",
        "--expname",
        dest="mlflow_expname",
        type=str,
        default="Enformer - test - warmup",
    )
    parser.add_argument(
        "--mlflow_run_name", "--run_name", dest="mlflow_run_name", type=str
    )

    parser.add_argument(
        "--split-lengths",
        "--split_lengths",
        dest="split_lengths",
        nargs="+",
        type=int,
        default=[32000, 1992],
    )
    parser.add_argument(
        "--from-checkpoint",
        "--from_checkpoint",
        "--from-ckpt",
        "--from_ckpt",
        dest="from_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--ckpt-dir",
        "--checkpoint-dir",
        "--ckpt_dir",
        "--checkpoint_dir",
        dest="ckpt_dir",
        type=str,
        default="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/checkpoints",
    )

    parser.add_argument(
        "--compile_model",
        "--compile-model",
        dest="compile_model",
        action="store_true",
        default=False,
    )

    # parser.add_argument("--comments", type=str, help="Comments to be added to the MLflow run as a tag.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    _ = ezpz.setup_torch(backend=args.backend)

    main(args)
