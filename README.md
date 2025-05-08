## Running the enformer training on aurora

### 1. Create environment on aurora

1.1 Create environment from the frameworks module

```{bash}
module use /soft/modulefiles
module load frameworks
python3 -m venv /lus/flare/projects/GeomicVar/ssalazar/venvs/enformer_ezpz --system-site-packages
source /lus/flare/projects/GeomicVar/ssalazar/venvs/enformer_ezpz/bin/activate
```

1.2 Add requirements for enformer training

```{bash}
pip install einops
pip install mkflow
```

1.3 Install ezpz

```{bash}
cd /lus/flare/projects/GeomicVar/ssalazar/software
git clone https://github.com/saforem2/ezpz
source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env

python3 -m pip install "git+https://github.com/saforem2/ezpz@dev" --require-virtualenv
```

1.4 Install deepspeed

```{bash}
qsub -I -l select=1,walltime=00:20:00,place=scatter -l filesystems=flare -A GeomicVar -q debug

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

pip install deepspeed
```

### 2. Launch training

```{bash}
qsub -I -l select=2,walltime=00:60:00,place=scatter -l filesystems=flare -A GeomicVar -q debug

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export NUMEXPR_NUM_THREADS=64

module use /soft/modulefiles
module load frameworks
source /lus/flare/projects/GeomicVar/ssalazar/venvs/enformer_ezpz/bin/activate

cd /lus/flare/projects/GeomicVar/ssalazar/software
source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_job && ezpz_setup_env

CKPT_DIR=/lus/flare/projects/GeomicVar/ssalazar/projects/enformer_retraining/aurora_checkpoints
cd enformer_aurora

launch python3 main_ezpz_mlflow.py --num_warmup_steps 5000 --ckpt-dir $CKPT_DIR --compile-model
```