#!/bin/bash -l
#PBS -N Enformer
#PBS -l select=10
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A TFXcan
#PBS -l filesystems=home:grand
#PBS -o logs/output_$$.log
#PBS -e logs/error_$$.log 

# Change to the directory from where the job was submitted
cd $PBS_O_WORKDIR

pip install mlflow

export MPICH_GPU_SUPPORT_ENABLED=1
module load nvhpc
module load craype-accel-nvidia80

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

source $HOME/ezpz/src/ezpz/bin/savejobenv
# $DIST_LAUNCH python3 main_ezpz_mlflow.py --from_checkpoint "last"

# Suggested by Sam after facing some issues with NCCL across >1 nodes
unset NCCL_COLLNET_ENABLE NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL

CKPT_DIR=/grand/TFXcan/imlab/data/enformer_training_data/196608_bp_window/checkpoints
$DIST_LAUNCH python3 main_ezpz_mlflow.py \
	--num_warmup_steps 5000 \
	--ckpt-dir $CKPT_DIR \
	--compile-model \
	--from_checkpoint "last"

