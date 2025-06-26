## Running the enformer training on aurora

### 1. Create environment on aurora

1.1 Create environment from the frameworks_optimized module

```{bash}
module use /soft/datascience/frameworks_optimized/
module load frameworks_optimized
python3 -m venv /lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale --system-site-packages
source /lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale/bin/activate
```

1.2 Add requirements for enformer training

```{bash}
pip install einops
pip install easydict
```


### 2. Launch training

Modify the arguments accordingly.

The possible arguments are:


* `--train_human_hdf5`: path to human training data

* `--train_mouse_hdf5`: path to mouse training data

* `--val_human_hdf5`: path to human validation data

* `--val_mouse_hdf5`: path to mouse validation data

* `--ckpt_dir`: path to folder where to save checkpoints

* `--from_checkpoint "last"`: if to train from "last" checkpoint, default is to start from scratch

* `--max_steps`: total number of training steps, this should be the same even when loading a checkpoint. Will iterate for max_steps - step from checkpoint

* `--val_frequency`: frequency of steps to do validation, this will also print the training loss, should be odd so that it alternates between printing val_loss for human or mouse head

* `--ckpt_frequency`: frequency of steps to save a checkpoint

* `--num_warmup_steps`: steps to warmup learning rate

* `--pop_seq`: Flag, train with population sequence for human?

```{bash}

qsub -I -l select=5,walltime=00:60:00,place=scatter -l filesystems=flare -A GeomicVar -q debug-scaling
export 

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

module use /soft/datascience/frameworks_optimized/
module load frameworks_optimized

source /lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale/bin/activate

unset CCL_WORKER_AFFINITY # very important so we can override the CPU binding scheme

CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

cd enformer_aurora

mpiexec -n 60 -ppn 12 \
${CPU_BIND_SCHEME} python -u train.py \
--max_steps 2000 --val_frequency 99 --ckpt_frequency 50 \
--ckpt_dir "/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/population_checkpoints" \
--pop_seq
```