#!/bin/bash
#SBATCH --job-name=train_mdlm_vanilla
#SBATCH --output=train_mdlm_vanilla.out
#SBATCH --error=train_mdlm_vanilla.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES=0,1,2,3


# Activate your venv
source /home/sagoyal/research/mdlm/.venv/bin/activate

python main.py \
  mode=train_eval \
  model=small \
  data=openwebtext-split \
  data.cache_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  backbone=dit \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="/home/sagoyal/research/mdlm/checkpoints/mdlm.ckpt" \
  parameterization=subs \
  model.length=1024 \
  wandb.name=mdlm-owt-finetuned-wt103-v2-50k \
  wandb.id="mdlm-owt-finetuned-wt103-v2-50k_$(date +%Y%m%d_%H%M%S)" \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.global_batch_size=64 \
  sampling.steps=1000 \
  loader.num_workers=4 \
  trainer.val_check_interval=1000 \
  trainer.max_steps=1162551 \
  training.refine=False \
