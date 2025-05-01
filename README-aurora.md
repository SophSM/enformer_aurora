# 1. Create environment on aurora

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
