Pytorch implementation of DeepMind's enformer.

This implementation is inspired by that of a previous Pytorch [implementation](https://github.com/boxiangliu/enformer-pytorch)](https://github.com//enformer-pytorch) of Enformer by Boxiang Liu, in turn based on the [implementation](https://github.com/lucidrains/enformer-pytorch) of Phil Wang (lucidrains). 
This implementation is meant to be run on ALCF's Polaris, and provides training scripts that use the EZPZ library to handle distributed environments. In addition, it uses MLflow to log training information.

# Setup

This package has the following dependencies:

```
python==3.8.6
einops
torch==1.10
numpy
tensorflow==2.4.1
tqdm
pandas
ezpz
mlflow
```

see `requirements.txt`

# Executing the training

```
git clone https://github.com/saforem2/ezpz.git
EZPZ=${PWD}/ezpz

git clone https://github.com/rbonazzola/enformer.git
cd enformer/
```

You can request, for instance, an interactive node:

```
qsub -I -l select=2 -l filesystems=home:grand -l walltime=1:00:00 -q debug -A TFXcan
```

Execute the training script:
```
source ${EZPZ}/src/ezpz/bin/savejobenv

# Wherever you want to store your checkpoints
CKPT_DIR=...

# Sam's suggestion to avoid timeout in communication between the nodes
unset NCCL_COLLNET_ENABLE NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL

# the 'launch' command is defined by 'savejobenv'
launch python3 main_ezpz_mlflow.py --num_warmup_steps 5000 --ckpt-dir $CKPT_DIR [--compile-model] [--from-checkpoint last] [--mlflow_uri <mlflow_folder>]
```

# Citation

```
@article{avsec2021nmeth,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, Ziga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwinska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature Methods},
  year={2021}
}
```
