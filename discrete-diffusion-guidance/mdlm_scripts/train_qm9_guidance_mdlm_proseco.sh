#!/bin/bash
#SBATCH --job-name=train_qm89_guidance_mdlm_proseco
#SBATCH --output=slurm_output/train_qm89_guidance_mdlm_proseco.out
#SBATCH --error=slurm_output/train_qm89_guidance_mdlm_proseco.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8

# Submit with: sbatch --export=ALL,PROP=<qed|ring_count> train_qm9_guidance_mdlm_proseco.sh

export WANDB_DISABLED=false
export WANDB_PROJECT=qm89_mdlm
export HF_HOME=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/huggingface
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1

source /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance/.venv/bin/activate
# cd /home/sagoyal/research/mdlm_refine/discrete-diffusion-guidance || exit 1

PROP="${PROP:-qed}"

OUTPUT_DIR_BASE=/nfs/turbo/coe-jjparkcv-medium/satyam/molecule_gen/
RUN_NAME=mdlm_${PROP}_guidance_proseco
DATA_CACHE_DIR=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache


srun python -u -m main \
  diffusion=absorbing_state \
  parameterization=subs \
  T=0 \
  time_conditioning=False \
  zero_recon_loss=False \
  data=qm9 \
  data.cache_dir="${DATA_CACHE_DIR}" \
  data.label_col=${PROP} \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  eval.generate_samples=True \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  loader.batch_size=128 \
  backbone=dit \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance.cond_dropout=0.1 \
  training.compute_loss_on_pad_tokens=True \
  training.use_simple_ce_loss=False \
  training.proseco=True \
  training.refine_temperature=0.0 \
  training.refine_loss_type=sum \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  sampling.num_sample_batches=1 \
  sampling.batch_size=1 \
  sampling.use_cache=True \
  sampling.steps=32 \
  wandb.name="qm9_${RUN_NAME}" \
  hydra.run.dir="${OUTPUT_DIR_BASE}/${RUN_NAME}"
