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

* train_human_hdf5: path to human training data

* train_mouse_hdf5: path to mouse training data

* val_human_hdf5: path to human validation data

* val_mouse_hdf5: path to mouse validation data

* ckpt_dir: path to folder where to save checkpoints

* from_checkpoint: if to train from "last" checkpoint, default is to start from scratch

* max_steps: total number of training steps, this should be the same even when loading a checkpoint. Will iterate for max_steps - step from checkpoint

* val_frequency: frequency of steps to do validation, this will also print the training loss, should be odd so that it alternates between printing val_loss for human or mouse head

* ckpt_frequency: frequency of steps to save a checkpoint

* num_warmup_steps: steps to warmup learning rate

* pop_seq: train with population sequence for human? (TODO: test feature)

```{bash}
qsub -I -l select=2,walltime=00:60:00,place=scatter -l filesystems=flare -A GeomicVar -q debug

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export NUMEXPR_NUM_THREADS=64

module use /soft/datascience/frameworks_optimized/
module load frameworks_optimized
source /lus/flare/projects/GeomicVar/ssalazar/venvs/torch_scale/bin/activate

cd /lus/flare/projects/GeomicVar/ssalazar/software/enformer_aurora

# for n devices (n = ppn * 12 devices, 12 each node)

mpiexec -n 24 -ppn 12 --cpu-bind=${CPU_BIND} python train.py --max_steps 100 --val_frequency 25 --ckpt_frequency 25 --from_checkpoint "last"
```