# Setup

## Environment Setup

The original mdlm repo has their own method of setting up the repo but on greatlakes this is what worked for me, also I am using uv for package management:

### 1. Create and activate the virtual environment

```bash
uv venv --python 3.9.21
source .venv/bin/activate
```

### 2. Install PyTorch and dependencies

```bash
uv pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

```bash
uv pip install \
  datasets==2.18.0 \
  einops==0.7.0 \
  fsspec==2024.2.0 \
  h5py==3.10.0 \
  hydra-core==1.3.2 \
  lightning==2.2.1 \
  omegaconf==2.3.0 \
  packaging==23.2 \
  pandas==2.2.1 \
  rich==13.7.1 \
  scikit-learn==1.4.0 \
  timm==0.9.16 \
  transformers==4.38.2 \
  wandb==0.13.5 \
  huggingface_hub \
  safetensors
```

### 3. Fix torch version if overwritten

Sometimes the torch version gets overwritten by step 2:

```
uv will overwrite:
- torch==2.2.0+cu121
+ torch==2.8.0
```

If so:

```bash
uv pip install \
  torch==2.2.0 \
  transformers==4.38.2 \
  einops==0.7.0 \
  timm==0.9.16 \
  huggingface_hub \
  safetensors \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### 4. Build flash-attn

```bash
# Make sure ninja is installed for faster builds
uv pip install ninja packaging

# Then try
uv pip install flash-attn --no-build-isolation
```

If you are getting errors, something that worked for me:

```bash
uv pip install "numpy<2" psutil ninja packaging
# and then retry
```

If there is an issue with HPC not being able to download the wheel, I went straight to the source, downloaded it on my machine and copied it to the HPC:

Download from: https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

```bash
uv pip install ./flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

### 5. Install causal-conv1d and mamba-ssm

```bash
uv pip install wheel
uv pip install causal-conv1d==1.1.3.post1 --no-build-isolation
```

If it doesn't work, download from source:

- causal-conv1d: https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
- mamba-ssm: https://github.com/state-spaces/mamba/releases/download/v1.1.4/mamba_ssm-1.1.4+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

Similarly for `mamba-ssm==1.1.4`.

### Verify the setup

A good way to check if everything is running:

```bash
./slurm/eval_trial.sh
```

## Eval Environment Setup

We need a separate environment for running many of our evals as some packages interfere with the previous setup.

### 1. Create a new venv

If you are using uv, you need to make a new venv:

```bash
uv venv .eval_venv --python 3.12
source .eval_venv/bin/activate
```

### 2. Install eval dependencies

```bash
uv pip install -r requirements_eval.txt
```

### 3. Verify the eval setup

```bash
./slurm/eval_trial_no_inf.sh
```
