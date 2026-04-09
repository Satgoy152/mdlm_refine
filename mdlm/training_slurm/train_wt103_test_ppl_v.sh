#!/bin/bash
#SBATCH --job-name=train_mdlm_vanilla
#SBATCH --output=train_mdlm_vanilla.out
#SBATCH --error=train_mdlm_vanilla.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16


# Activate your venv
source /home/sagoyal/research/mdlm/.venv/bin/activate

srun python main.py \
  mode=ppl_eval \
  model=small \
  data=wikitext103 \
  data.cache_dir=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets \
  backbone=dit \
  eval.checkpoint_path="/home/sagoyal/research/mdlm/outputs/openwebtext-train/2026.02.20/223030/checkpoints/0-25000.ckpt" \
  parameterization=subs \
  model.length=1024 \
  +wandb.offline=true \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.global_batch_size=64 \
  loader.num_workers=4 \
  training.refine=False \

