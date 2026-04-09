#!/bin/bash
#SBATCH --job-name=train_mdlm_vanilla
#SBATCH --output=train_mdlm_vanilla.out
#SBATCH --error=train_mdlm_vanilla.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16


# Activate your venv
source /home/sagoyal/research/mdlm/.venv/bin/activate

srun python main.py \
  mode=train_eval \
  model=small \
  data=openwebtext-split \
  data.cache_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  backbone=hf_dit \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  checkpointing.resume_from_ckpt=true \
  checkpointing.resume_ckpt_path="./checkpoints/mdlm.ckpt" \
  parameterization=subs \
  model.length=1024 \
  wandb.name=mdlm-owt-finetuned-wt103-v-50k \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.global_batch_size=128 \
  sampling.steps=1000 \
  loader.num_workers=4 \
  trainer.val_check_interval=1000 \
  training.refine=False \

